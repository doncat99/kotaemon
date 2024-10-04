

import os
from io import BytesIO
import base64
import json
import copy
import logging
from pathlib import Path
from typing import List, Dict, Any

from PIL import Image

from magic_pdf.pipe.UNIPipe import UNIPipe
from magic_pdf.pipe.OCRPipe import OCRPipe
from magic_pdf.pipe.TXTPipe import TXTPipe
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
from llama_index.core.schema import Document

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

    def __init__(self, output_dir: str, generate_thumbnails: bool = False):
        """
        Initialize the MinerUDocumentReader.

        Args:
            output_dir (str): Directory to store intermediate files and parsed outputs.
            generate_thumbnails (bool, optional): Whether to generate page thumbnails. Defaults to False.
        """
        self.output_dir = Path(output_dir)
        self.generate_thumbnails = generate_thumbnails

    def extract_content(
        self, pdf_path: str, parse_method: str = 'auto', model_json_path: str = None, is_json_md_dump: bool = True
    ) -> Dict[str, Any]:
        """
        Extract the content of a PDF file using MinerU's extraction pipelines.

        Args:
            pdf_path (str): Path to the PDF file.
            parse_method (str, optional): Parsing method to use ('auto', 'ocr', or 'txt'). Defaults to 'auto'.
            model_json_path (str, optional): Path to the pre-trained model JSON file.
            is_json_md_dump (bool, optional): Whether to output JSON and Markdown results. Defaults to True.

        Returns:
            Dict[str, Any]: Parsed content list, raw JSON, and image paths.
        """
        try:
            output_paths = self._generate_output_paths(pdf_path)
            content_list, raw_json, pages_image = self._run_mineru_pipeline(
                pdf_path, parse_method, model_json_path, is_json_md_dump, output_paths
            )

            # Generate thumbnails if enabled
            thumbnails = []
            if self.generate_thumbnails:
                pages = [item.get("page_idx", 0) for item in content_list if item.get("type") == "text"]
                thumbnails = get_page_thumbnails(Path(pdf_path), pages)

            return {
                "content_list": content_list,
                "raw_json": raw_json,
                "pages_image": pages_image,
                "thumbnails": thumbnails
            }

        except Exception as e:
            logger.exception(f"Error processing PDF '{pdf_path}': {e}")
            return {}

    def _run_mineru_pipeline(
        self, pdf_path: str, parse_method: str, model_json_path: str, is_json_md_dump: bool, output_paths: Dict[str, str]
    ) -> tuple:
        """
        Run the MinerU processing pipeline on a PDF file.

        Args:
            pdf_path (str): Path to the PDF file.
            parse_method (str): Method to use ('auto', 'ocr', or 'txt').
            model_json_path (str): Path to the pre-trained model JSON file.
            is_json_md_dump (bool): Whether to output JSON and Markdown results.
            output_paths (Dict[str, str]): Generated paths for output storage.

        Returns:
            tuple: content_list, raw_json, pages_image.
        """
        pdf_name = os.path.basename(pdf_path).split(".")[0]
        pdf_bytes = open(pdf_path, "rb").read()

        # Prepare output writers
        image_writer = DiskReaderWriter(output_paths['images'])
        md_writer = DiskReaderWriter(output_paths['base'])

        # Select the pipeline based on the parse method
        if parse_method == "auto":
            model_json = self._load_model_json(model_json_path)
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
        if is_json_md_dump:
            md_content = pipe.pipe_mk_markdown(os.path.basename(output_paths['images']), drop_mode="none")
            self.json_md_dump(pipe, md_writer, pdf_name, content_list, md_content)

        return content_list, json.dumps(pipe.pdf_mid_data, ensure_ascii=False, indent=4), []

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
