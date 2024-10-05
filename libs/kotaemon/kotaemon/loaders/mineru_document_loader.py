import os
from io import BytesIO
import base64
import json
import copy
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from PIL import Image
from fsspec import AbstractFileSystem
from llama_index.core.schema import Document

from magic_pdf.pipe.UNIPipe import UNIPipe
from magic_pdf.pipe.OCRPipe import OCRPipe
from magic_pdf.pipe.TXTPipe import TXTPipe
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
import magic_pdf.model as model_config
model_config.__use_inside_model__ = True

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Utility Functions
def convert_image_to_base64(img: Image.Image, format: str = "PNG") -> str:
    """Convert an image to base64 format."""
    img_bytes = BytesIO()
    img.save(img_bytes, format=format)
    img_base64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{img_base64}"


def get_page_thumbnails(file_path: Path, pages: List[int], dpi: int = 80) -> List[str]:
    """Generate base64 thumbnails for specific pages in the PDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("Please install PyMuPDF: 'pip install PyMuPDF'")

    doc = fitz.open(str(file_path))
    thumbnails = []
    for page_number in pages:
        try:
            page = doc.load_page(page_number)
            pixmap = page.get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            thumbnails.append(convert_image_to_base64(img))
        except Exception as e:
            logger.warning(f"Failed to process page {page_number}: {e}")
            thumbnails.append("Image not available")
    doc.close()
    return thumbnails


class MinerUDocumentReader:
    """Wrapper to process PDFs using MinerU and convert the output into `llama_index` Document format."""

    def __init__(self):
        """
        Initialize the MinerUDocumentReader.

        Args:
            output_dir (str): Directory to store intermediate files and parsed outputs.
            generate_thumbnails (bool, optional): Whether to generate page thumbnails. Defaults to False.
        """
        self.output_dir = "./"
        self.generate_thumbnails = True

    def load_data(
        self, pdf_path: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """
        Extract the content of a PDF file using MinerU's extraction pipelines and convert to `Document` format.

        Args:
            pdf_path (str): Path to the PDF file.
            parse_method (str, optional): Parsing method to use ('auto', 'ocr', or 'txt'). Defaults to 'auto'.
            model_json_path (str, optional): Path to the pre-trained model JSON file.
            is_json_md_dump (bool, optional): Whether to output JSON and Markdown results. Defaults to True.

        Returns:
            List[Document]: List of `Document` objects in `llama_index` format.
        """
        parse_method = 'auto'
        model_json_path = None
        is_json_md_dump = False
        
        try:
            output_paths = self._generate_output_paths(pdf_path)
            content_list = self._run_mineru_pipeline(
                pdf_path, parse_method, model_json_path, is_json_md_dump, output_paths
            )

            # Convert content list to llama_index Documents
            documents = self._convert_to_documents(content_list)

            # Optionally generate thumbnails for each page
            if self.generate_thumbnails:
                thumbnails = get_page_thumbnails(Path(pdf_path), [item.get("page_idx", 0) for item in content_list])
                self._append_thumbnails(documents, thumbnails)

            return documents

        except Exception as e:
            logger.exception(f"Error processing PDF '{pdf_path}': {e}")
            return []

    def _run_mineru_pipeline(
        self, pdf_path: str, parse_method: str, model_json_path: str, is_json_md_dump: bool, output_paths: Dict[str, str]
    ) -> tuple:
        """Run the MinerU processing pipeline on a PDF file."""
        pdf_name = os.path.basename(pdf_path).split(".")[0]
        pdf_bytes = open(pdf_path, "rb").read()

        # Prepare output writers
        image_writer = DiskReaderWriter(output_paths['images'])
        md_writer = DiskReaderWriter(output_paths['base'])

        # Select the pipeline based on the parse method
        if parse_method == "auto":
            model_json = self._load_model_json(model_json_path) if model_json_path else []
            pipe = UNIPipe(pdf_bytes, {"_pdf_type": "", "model_list": model_json}, image_writer)
        elif parse_method == "txt":
            pipe = TXTPipe(pdf_bytes, [], image_writer)
        elif parse_method == "ocr":
            pipe = OCRPipe(pdf_bytes, [], image_writer)
        else:
            raise ValueError("Unknown parse method. Allowed values are: 'auto', 'txt', 'ocr'")

        # Classify and parse the document
        pipe.pipe_classify()
        pipe.pipe_parse()

        # Generate content list and optional Markdown outputs
        content_list = pipe.pipe_mk_uni_format(os.path.basename(output_paths['images']), drop_mode="none")
        # if is_json_md_dump:
        md_content = pipe.pipe_mk_markdown(os.path.basename(output_paths['images']), drop_mode="none")
        self.json_md_dump(pipe, md_writer, pdf_name, content_list, md_content)

        return content_list

    def _convert_to_documents(self, content_list: List[Dict[str, Any]]) -> List[Document]:
        """Convert parsed content into `llama_index` Documents."""
        documents = []
        for item in content_list:
            text_content = item.get("text", "")
            page_idx = item.get("page_idx", 0)
            if text_content:
                metadata = {"type": item.get("type", "text"), "page_idx": page_idx, "source": "MinerU"}
                documents.append(Document(text=text_content, metadata=metadata))

        return documents

    def _append_thumbnails(self, documents: List[Document], thumbnails: List[str]):
        """Append thumbnails to the corresponding documents."""
        for doc, thumbnail in zip(documents, thumbnails):
            if doc.metadata.get("type") == "text":
                doc.metadata["thumbnail"] = thumbnail

    def _generate_output_paths(self, pdf_path: str) -> Dict[str, str]:
        """Generate output paths for the extracted content."""
        pdf_name = os.path.basename(pdf_path).split(".")[0]
        base_output_path = os.path.join(self.output_dir, pdf_name)
        image_output_path = os.path.join(base_output_path, 'images')

        os.makedirs(base_output_path, exist_ok=True)
        os.makedirs(image_output_path, exist_ok=True)

        return {"base": base_output_path, "images": image_output_path}

    def _load_model_json(self, model_json_path: str) -> List[Dict[str, Any]]:
        """Load the pre-trained model JSON if provided."""
        if model_json_path and os.path.exists(model_json_path):
            with open(model_json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def json_md_dump(self, pipe, md_writer, pdf_name, content_list, md_content):
        """Save the generated content in JSON and Markdown formats."""
        orig_model_list = copy.deepcopy(pipe.model_list)
        md_writer.write(content=json.dumps(orig_model_list, ensure_ascii=False, indent=4), path=f"{pdf_name}_model.json")
        md_writer.write(content=json.dumps(pipe.pdf_mid_data, ensure_ascii=False, indent=4), path=f"{pdf_name}_middle.json")
        md_writer.write(content=json.dumps(content_list, ensure_ascii=False, indent=4), path=f"{pdf_name}_content_list.json")
        md_writer.write(content=md_content, path=f"{pdf_name}.md")
