import html
import io
import logging
from enum import Enum
from typing import AsyncGenerator, Union

import pymupdf
from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import (
    AnalyzeResult,
    DocumentTable,
)
from azure.core.credentials import AzureKeyCredential
from azure.core.credentials_async import AsyncTokenCredential
from PIL import Image
from .page import Page
from .parser import Parser

logger = logging.getLogger("scripts")


class DocumentAnalysisParser(Parser):
    """
    Concrete parser backed by Azure AI Document Intelligence that can parse many document formats into pages
    To learn more, please visit https://learn.microsoft.com/azure/ai-services/document-intelligence/overview
    """

    def __init__(
        self,
        endpoint: str,
        credential: Union[AsyncTokenCredential, AzureKeyCredential],
        model_id="prebuilt-layout",
    ):
        self.model_id = model_id
        self.endpoint = endpoint
        self.credential = credential

    async def parse(self, bytes: bytes, name: str) -> AsyncGenerator[Page, None]:
        logger.info("Extracting text from '%s' using Azure Document Intelligence", name)
        content = io.BytesIO(bytes)

        async with DocumentIntelligenceClient(
            endpoint=self.endpoint, credential=self.credential
        ) as document_intelligence_client:
            poller = await document_intelligence_client.begin_analyze_document(
                model_id=self.model_id,
                analyze_request=content,
                content_type="application/octet-stream",
            )
            analyze_result: AnalyzeResult = await poller.result()
            for page in analyze_result.pages:
                tables_on_page = [
                    table
                    for table in (analyze_result.tables or [])
                    if table.bounding_regions
                    and table.bounding_regions[0].page_number == page.page_number
                ]

                class ObjectType(Enum):
                    NONE = -1
                    TABLE = 0

                page_offset = page.spans[0].offset
                page_length = page.spans[0].length
                mask_chars: list[tuple[ObjectType, Union[int, None]]] = [
                    (ObjectType.NONE, None)
                ] * page_length
                for table_idx, table in enumerate(tables_on_page):
                    for span in table.spans:
                        for i in range(span.length):
                            idx = span.offset - page_offset + i
                            if idx >= 0 and idx < page_length:
                                mask_chars[idx] = (ObjectType.TABLE, table_idx)

                page_text = ""
                added_objects: set[tuple[ObjectType, Union[int, None]]] = set()
                for idx, mask_char in enumerate(mask_chars):
                    object_type, object_idx = mask_char
                    if object_type == ObjectType.NONE:
                        page_text += analyze_result.content[page_offset + idx]
                    elif object_type == ObjectType.TABLE:
                        if object_idx is None:
                            raise ValueError("Expected object_idx to be set")
                        if mask_char not in added_objects:
                            page_text += DocumentAnalysisParser.table_to_html(
                                tables_on_page[object_idx]
                            )
                            added_objects.add(mask_char)
                # We remove these comments since they are not needed and skew the page numbers
                page_text = page_text.replace("<!-- PageBreak -->", "")
                # We remove excess newlines at the beginning and end of the page
                page_text = page_text.strip()
                yield Page(page_num=page.page_number - 1, text=page_text)

    @staticmethod
    def table_to_html(table: DocumentTable):
        table_html = "<figure><table>"
        rows = [
            sorted(
                [cell for cell in table.cells if cell.row_index == i],
                key=lambda cell: cell.column_index,
            )
            for i in range(table.row_count)
        ]
        for row_cells in rows:
            table_html += "<tr>"
            for cell in row_cells:
                tag = (
                    "th"
                    if (cell.kind == "columnHeader" or cell.kind == "rowHeader")
                    else "td"
                )
                cell_spans = ""
                if cell.column_span is not None and cell.column_span > 1:
                    cell_spans += f" colSpan={cell.column_span}"
                if cell.row_span is not None and cell.row_span > 1:
                    cell_spans += f" rowSpan={cell.row_span}"
                table_html += f"<{tag}{cell_spans}>{html.escape(cell.content)}</{tag}>"
            table_html += "</tr>"
        table_html += "</table></figure>"
        return table_html

    @staticmethod
    def crop_image_from_pdf_page(
        doc: pymupdf.Document,
        page_number: int,
        bounding_box: tuple[float, float, float, float],
    ) -> bytes:
        """
        Crops a region from a given page in a PDF and returns it as an image.

        :param doc: PyMuPDF document object.
        :param page_number: The page number to crop from (0-indexed).
        :param bounding_box: A tuple of (x0, y0, x1, y1) coordinates in inches.
        :return: PNG image bytes of the cropped area.
        """
        page = doc.load_page(page_number)

        # Convert bounding box from inches to points (72 points = 1 inch)
        # Document Intelligence returns coordinates in inches for PDFs
        bbx = [x * 72 for x in bounding_box]
        rect = pymupdf.Rect(bbx)

        # Validate the rectangle
        if rect.is_empty or rect.is_infinite:
            logger.error(
                "[crop_image_from_pdf_page] Invalid bounding box on page %d: %s (in points: %s)",
                page_number,
                bounding_box,
                bbx,
            )
            raise ValueError(f"Invalid bounding box: {bounding_box}")

        # Get page dimensions for validation
        page_rect = page.rect
        logger.debug(
            "[crop_image_from_pdf_page] Page %d dimensions: %.2f x %.2f points (%.2f x %.2f inches)",
            page_number,
            page_rect.width,
            page_rect.height,
            page_rect.width / 72,
            page_rect.height / 72,
        )

        # Check if bounding box is within page bounds (with small tolerance)
        if not page_rect.contains(rect) and not page_rect.intersects(rect):
            logger.warning(
                "[crop_image_from_pdf_page] Bounding box %s extends beyond page bounds %s on page %d",
                rect,
                page_rect,
                page_number,
            )

        # Intersect with page bounds to avoid errors
        rect = rect & page_rect

        if rect.is_empty:
            logger.error(
                "[crop_image_from_pdf_page] Bounding box completely outside page bounds on page %d",
                page_number,
            )
            raise ValueError(f"Bounding box {bounding_box} is outside page bounds")

        # Render at 300 DPI for high quality image extraction
        # Matrix scaling: 300/72 ≈ 4.167x enlargement
        pix = page.get_pixmap(matrix=pymupdf.Matrix(300 / 72, 300 / 72), clip=rect)

        logger.debug(
            "[crop_image_from_pdf_page] Rendered pixmap: %d x %d pixels",
            pix.width,
            pix.height,
        )

        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        bytes_io = io.BytesIO()
        img.save(bytes_io, format="PNG")
        return bytes_io.getvalue()
