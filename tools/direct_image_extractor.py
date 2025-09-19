"""
Direct image extraction service for various document formats.
Uses PyMuPDF for PDFs and other libraries for different document types.
"""

import base64
import io
import logging
import os
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Tuple

import zipfile
from xml.etree import ElementTree as ET


def _load_optional_module(module_name: str, warning: str):
    try:
        return import_module(module_name)
    except ImportError:
        logging.warning(warning)
        return None


fitz = _load_optional_module("fitz", "PyMuPDF not available - PDF image extraction disabled")
PYMUPDF_AVAILABLE = fitz is not None

_pil_image_module = _load_optional_module("PIL.Image", "Pillow not available - image processing disabled")
if _pil_image_module is not None:
    Image = _pil_image_module
    PIL_AVAILABLE = True
else:
    Image = None
    PIL_AVAILABLE = False

DOCX_SUPPORT = True


class DirectImageExtractor:
    """Extracts images directly from various document formats."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def extract_images_from_bytes(self, file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        """
        Extract images from document bytes based on file extension.

        Args:
            file_bytes: Document bytes
            filename: Original filename to determine format

        Returns:
            List of image dictionaries with format:
            {
                'image_id': str,
                'page_number': int,
                'image_data': bytes,
                'format': str,
                'bbox': Optional[Dict],
                'size': Tuple[int, int]
            }
        """
        try:
            file_extension = Path(filename).suffix.lower()

            if file_extension == '.pdf':
                return self._extract_from_pdf(file_bytes)
            elif file_extension in ['.docx', '.doc']:
                return self._extract_from_docx(file_bytes)
            elif file_extension in ['.pptx', '.ppt']:
                return self._extract_from_pptx(file_bytes)
            elif file_extension in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']:
                return self._extract_from_image(file_bytes, filename)
            else:
                self.logger.info(f"No image extraction support for {file_extension}")
                return []

        except Exception as e:
            self.logger.error(f"Error extracting images from {filename}: {str(e)}")
            return []

    def _extract_from_pdf(self, file_bytes: bytes) -> List[Dict[str, Any]]:
        """Extract images from PDF using PyMuPDF."""
        if not PYMUPDF_AVAILABLE:
            self.logger.warning("PyMuPDF not available for PDF image extraction")
            return []

        images = []
        try:
            # Open PDF from bytes
            doc = fitz.open(stream=file_bytes, filetype="pdf")

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images()

                for img_index, img in enumerate(image_list):
                    try:
                        # Get image data
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)

                        # Convert to RGB if CMYK
                        if pix.n - pix.alpha < 4:
                            img_data = pix.tobytes("png")
                        else:
                            pix1 = fitz.Pixmap(fitz.csRGB, pix)
                            img_data = pix1.tobytes("png")
                            pix1 = None

                        # Get image properties
                        image_dict = {
                            'image_id': f"page_{page_num + 1}_img_{img_index + 1}",
                            'page_number': page_num + 1,
                            'image_data': img_data,
                            'format': 'png',
                            'size': (pix.width, pix.height),
                            'bbox': {
                                'x': 0,  # PyMuPDF doesn't provide exact positioning easily
                                'y': 0,
                                'width': pix.width,
                                'height': pix.height
                            }
                        }

                        images.append(image_dict)
                        pix = None

                    except Exception as e:
                        self.logger.warning(f"Failed to extract image {img_index} from page {page_num}: {str(e)}")
                        continue

            doc.close()
            self.logger.info(f"Extracted {len(images)} images from PDF")

        except Exception as e:
            self.logger.error(f"Error processing PDF: {str(e)}")

        return images

    def _extract_from_docx(self, file_bytes: bytes) -> List[Dict[str, Any]]:
        """Extract images from DOCX files."""
        if not DOCX_SUPPORT:
            self.logger.warning("DOCX support not available")
            return []

        images = []
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes), 'r') as zip_file:
                # Find image files in the DOCX
                image_files = [f for f in zip_file.namelist() if f.startswith('word/media/')]

                for idx, img_path in enumerate(image_files):
                    try:
                        img_data = zip_file.read(img_path)
                        img_format = Path(img_path).suffix[1:].lower()  # Remove dot

                        # Get image size if possible
                        size = (0, 0)
                        if PIL_AVAILABLE:
                            try:
                                with Image.open(io.BytesIO(img_data)) as pil_img:
                                    size = pil_img.size
                            except Exception:
                                pass

                        image_dict = {
                            'image_id': f"docx_img_{idx + 1}",
                            'page_number': 1,  # DOCX doesn't have clear page concept
                            'image_data': img_data,
                            'format': img_format,
                            'size': size,
                            'bbox': None
                        }

                        images.append(image_dict)

                    except Exception as e:
                        self.logger.warning(f"Failed to extract image {img_path}: {str(e)}")
                        continue

            self.logger.info(f"Extracted {len(images)} images from DOCX")

        except Exception as e:
            self.logger.error(f"Error processing DOCX: {str(e)}")

        return images

    def _extract_from_pptx(self, file_bytes: bytes) -> List[Dict[str, Any]]:
        """Extract images from PPTX files."""
        if not DOCX_SUPPORT:
            self.logger.warning("PPTX support not available")
            return []

        images = []
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes), 'r') as zip_file:
                slide_files = [
                    name for name in zip_file.namelist()
                    if name.startswith('ppt/slides/slide') and name.endswith('.xml')
                ]

                if not slide_files:
                    self.logger.warning("No slide files found in PPTX")
                    return []

                def slide_sort_key(slide_path: str) -> int:
                    try:
                        stem = Path(slide_path).stem
                        return int(stem.replace('slide', ''))
                    except ValueError:
                        return 0

                slide_files.sort(key=slide_sort_key)

                rel_ns = '{http://schemas.openxmlformats.org/package/2006/relationships}'
                rel_type_image = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/image'
                r_ns = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
                namespaces = {
                    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
                    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
                }

                per_slide_counts: Dict[int, int] = {}

                for slide_index, slide_path in enumerate(slide_files, start=1):
                    try:
                        slide_xml = ET.fromstring(zip_file.read(slide_path))
                    except KeyError:
                        self.logger.warning(f"Slide file {slide_path} missing from PPTX")
                        continue

                    rels_path = slide_path.replace('slides/', 'slides/_rels/').replace('.xml', '.xml.rels')
                    rels_map: Dict[str, str] = {}

                    if rels_path in zip_file.namelist():
                        try:
                            rels_root = ET.fromstring(zip_file.read(rels_path))
                            for rel in rels_root.findall(f'{rel_ns}Relationship'):
                                if rel.attrib.get('Type') != rel_type_image:
                                    continue
                                rel_id = rel.attrib.get('Id')
                                target = rel.attrib.get('Target')
                                if not rel_id or not target:
                                    continue
                                resolved = os.path.normpath(os.path.join(os.path.dirname(slide_path), target))
                                rels_map[rel_id] = resolved.replace('\\', '/')
                        except Exception as rel_error:
                            self.logger.warning(f"Failed to parse relationships for {slide_path}: {rel_error}")

                    slide_images: List[str] = []
                    seen_media: set[str] = set()
                    for blip in slide_xml.findall('.//a:blip', namespaces):
                        rel_id = blip.attrib.get(f'{r_ns}embed')
                        if not rel_id:
                            continue
                        media_path = rels_map.get(rel_id)
                        if media_path and media_path.startswith('ppt/') and media_path not in seen_media:
                            slide_images.append(media_path)
                            seen_media.add(media_path)

                    per_slide_counts.setdefault(slide_index, 0)

                    for media_path in slide_images:
                        try:
                            img_data = zip_file.read(media_path)
                        except KeyError:
                            self.logger.warning(f"Image {media_path} referenced by {slide_path} missing in archive")
                            continue

                        img_format = Path(media_path).suffix[1:].lower() or 'png'

                        size = (0, 0)
                        if PIL_AVAILABLE:
                            try:
                                with Image.open(io.BytesIO(img_data)) as pil_img:
                                    size = pil_img.size
                            except Exception:
                                pass

                        per_slide_counts[slide_index] += 1
                        sequence = per_slide_counts[slide_index]

                        image_dict = {
                            'image_id': f"pptx_slide_{slide_index}_img_{sequence}",
                            'page_number': slide_index,
                            'image_data': img_data,
                            'format': img_format,
                            'size': size,
                            'bbox': None
                        }

                        images.append(image_dict)

            self.logger.info(f"Extracted {len(images)} images from PPTX")

        except Exception as e:
            self.logger.error(f"Error processing PPTX: {str(e)}")

        return images

    def _extract_from_image(self, file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        """Handle single image files."""
        if not PIL_AVAILABLE:
            self.logger.warning("Pillow not available for image processing")
            return []

        try:
            with Image.open(io.BytesIO(file_bytes)) as img:
                # Convert to RGB if needed
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')

                # Save as PNG bytes
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='PNG')
                img_data = img_buffer.getvalue()

                return [{
                    'image_id': f"single_image",
                    'page_number': 1,
                    'image_data': img_data,
                    'format': 'png',
                    'size': img.size,
                    'bbox': {
                        'x': 0,
                        'y': 0,
                        'width': img.size[0],
                        'height': img.size[1]
                    }
                }]

        except Exception as e:
            self.logger.error(f"Error processing image file: {str(e)}")
            return []

    def convert_to_normalized_format(self, extracted_images: List[Dict[str, Any]], source_url: str) -> List[Dict[str, Any]]:
        """
        Convert extracted images to the normalized format expected by ImageDescriptionClient.

        Args:
            extracted_images: List of images from extract_images_from_bytes
            source_url: Source URL for the document

        Returns:
            List of normalized image dictionaries
        """
        normalized_images = []

        for img in extracted_images:
            try:
                # Encode image data to base64
                base64_data = base64.b64encode(img['image_data']).decode('utf-8')

                normalized_img = {
                    'id': img['image_id'],
                    'pageNumber': img['page_number'],
                    'data': base64_data,
                    'contentType': f"image/{img['format']}",
                    'sourceUrl': source_url,
                    'boundingBox': img.get('bbox'),
                    'size': {
                        'width': img['size'][0],
                        'height': img['size'][1]
                    } if img['size'] != (0, 0) else None,
                    'locationMetadata': {
                        'pageNumber': img['page_number'],
                        'ordinalPosition': 1,  # Default since we don't track position within page
                        'boundingPolygons': img.get('bbox', {})
                    }
                }

                normalized_images.append(normalized_img)

            except Exception as e:
                self.logger.warning(f"Failed to normalize image {img.get('image_id', 'unknown')}: {str(e)}")
                continue

        return normalized_images
