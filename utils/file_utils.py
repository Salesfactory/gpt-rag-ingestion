import os
import re
import logging
from typing import Optional


def get_file_extension(file_path: str) -> Optional[str]:
    file_path = os.path.basename(file_path)
    return file_path.split(".")[-1]


def get_filename(file_path: str) -> str:
    match = re.search(r"documents/(.*/)?(.*)", file_path)
    filepath = ""
    if match:
        filepath = (match.group(1) or "") + (match.group(2) or "")
    return filepath


def infer_content_type_from_url(url: str) -> str:
    """
    Infer content type from file extension when blob metadata doesn't include it.

    Args:
        url: Document URL with file extension

    Returns:
        MIME type string
    """
    ext = url.lower().split(".")[-1] if "." in url else ""

    content_type_map = {
        # Text formats
        "txt": "text/plain",
        "html": "text/html",
        "htm": "text/html",
        "json": "application/json",
        "xml": "application/xml",
        "csv": "text/csv",
        "md": "text/markdown",
        "py": "text/x-python",
        # Document formats
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "doc": "application/msword",
        "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "ppt": "application/vnd.ms-powerpoint",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "xls": "application/vnd.ms-excel",
        # Image formats
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "bmp": "image/bmp",
        "tiff": "image/tiff",
        "tif": "image/tiff",
    }

    inferred_type = content_type_map.get(ext, "application/octet-stream")

    if inferred_type == "application/octet-stream" and ext:
        logging.warning(
            f"[infer_content_type] Unknown file extension '.{ext}' for URL: {url}. "
            f"Using default: application/octet-stream"
        )

    return inferred_type
