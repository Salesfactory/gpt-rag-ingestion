import logging
import os
import re

from langchain.text_splitter import MarkdownTextSplitter, RecursiveCharacterTextSplitter

from .base_chunker import BaseChunker
from ..exceptions import UnsupportedFormatError, DocAnalysisError
from tools import DocumentIntelligenceClient, ImageDescriptionClient, MultimodalBlobClient
from tools.blob import BlobStorageClient
from tools.direct_image_extractor import DirectImageExtractor

from azure.core.credentials import AzureKeyCredential
from prepdocslib.pdfparser import DocumentAnalysisParser


class DocAnalysisChunker(BaseChunker):
    """
    DocAnalysisChunker class is responsible for analyzing and splitting document content into chunks
    based on specific format criteria, utilizing the Document Intelligence service for content analysis.

    Format Support:
    ---------------
    The DocAnalysisChunker class leverages the Document Intelligence service to process and analyze
    a wide range of document formats. The class ensures that document content is accurately processed
    and divided into manageable chunks.

    - Supported Formats: The chunker processes document formats supported by the Document Intelligence client.
    - Unsupported Formats: If a document's format is not supported by the client, an `UnsupportedFormatError` is raised.

    Chunking Parameters:
    --------------------
    - max_chunk_size: The maximum size of each chunk in tokens. This value is sourced from the `NUM_TOKENS` 
    environment variable, with a default of 650 tokens.
    - token_overlap: The number of overlapping tokens between consecutive chunks, sourced from the `TOKEN_OVERLAP` 
    environment variable, with a default of 100 tokens.
    - minimum_chunk_size: The minimum size of each chunk in tokens, sourced from the `MIN_CHUNK_SIZE` environment 
    variable, with a default of 100 tokens.

    Document Analysis:
    ------------------
    - The document is analyzed using the Document Intelligence service, extracting its content and structure.
    - The analysis process includes identifying the number of pages and providing a preview of the content.
    - If the document is large, a warning is logged to indicate potential timeout issues during processing.

    Content Chunking:
    -----------------
    - The document content is split into chunks using format-specific strategies.
    - HTML tables in the content are replaced with placeholders during the chunking process to simplify splitting.
    - After chunking, the original content, such as HTML tables, is restored in place of the placeholders.
    - The chunking process also manages page numbering based on the presence of page breaks, ensuring each chunk 
    is correctly associated with its corresponding page.

    Error Handling:
    ---------------
    - The class includes comprehensive error handling during document analysis, such as managing unsupported formats 
    and handling general exceptions.
    - The chunking process's progress and outcomes, including the number of chunks created or skipped, are logged.
    """
    def __init__(self, data, max_chunk_size=None, minimum_chunk_size=None, token_overlap=None):
        super().__init__(data)
        self.max_chunk_size = max_chunk_size or int(os.getenv("NUM_TOKENS", "750"))
        self.minimum_chunk_size = minimum_chunk_size or int(os.getenv("MIN_CHUNK_SIZE", "100"))
        self.token_overlap = token_overlap or int(os.getenv("TOKEN_OVERLAP", "100"))
        self.docint_client = DocumentIntelligenceClient()

        # Multimodal processing setup with direct image extraction
        self.multimodal_enabled = self.docint_client.multimodal_enabled
        if self.multimodal_enabled:
            self.image_description_client = ImageDescriptionClient()
            self.multimodal_blob_client = MultimodalBlobClient()
            self.direct_image_extractor = DirectImageExtractor()
            logging.info(f"[doc_analysis_chunker] Multimodal processing enabled with direct image extraction for {self.filename}")
        else:
            self.image_description_client = None
            self.multimodal_blob_client = None
            self.direct_image_extractor = None

        # Update format handling to ensure consistent format comparison
        self.supported_formats = set(
            f".{ext.lower().lstrip('.')}" for ext in self.docint_client.file_extensions
        )

        # Ensure extension is in the correct format for comparison
        self.extension = f".{self.extension.lower().lstrip('.')}"

        # Add debug logging for format detection
        logging.debug(f"[doc_analysis_chunker] File extension detected: {self.extension}")
        logging.debug(f"[doc_analysis_chunker] Supported formats: {self.supported_formats}")
        logging.debug(f"[doc_analysis_chunker] Multimodal enabled: {self.multimodal_enabled}")

    async def get_chunks(self):
        """
        Analyzes the document and generates content chunks based on the analysis.

        This method supports both traditional text-only processing and multimodal processing
        (text + images) depending on configuration.

        Returns:
            list: A list of dictionaries, each representing a chunk of the document content.
                 For multimodal processing, includes both text and image chunks.

        Raises:
            UnsupportedFormatError: If the document format is not supported.
            Exception: If there is an error during document analysis.
        """
        if self.extension not in self.supported_formats:
            raise UnsupportedFormatError(f"[doc_analysis_chunker] {self.extension} format is not supported")

        logging.info(f"[doc_analysis_chunker][{self.filename}] Running get_chunks (multimodal: {self.multimodal_enabled}).")

        if self.multimodal_enabled:
            return await self._get_multimodal_chunks()
        else:
            return await self._get_traditional_chunks()

    async def _get_traditional_chunks(self):
        """
        Traditional text-only chunk processing using the existing parser approach.
        """
        pages = None

        try:
            doc_int_parser = DocumentAnalysisParser(
                endpoint="https://eastus.api.cognitive.microsoft.com/",
                credential=AzureKeyCredential(os.getenv("COGNITIVE_SERVICES_KEY")),
                use_content_understanding=False,
                content_understanding_endpoint=os.getenv("AZ_COMPUTER_VISION_ENDPOINT"),
            )
            pages = [page async for page in doc_int_parser.parse(bytes=self.document_bytes, name=self.filename)]
        except Exception as e:
            logging.error(f"[doc_analysis_chunker][{self.filename}] Error parsing document: {str(e)}")
            raise

        document_content = ""
        for page in pages:
            document_content += page.text

        document = {
            "content": document_content
        }

        chunks = self._process_document_chunks(document)

        return chunks

    async def _get_multimodal_chunks(self):
        """
        Multimodal chunk processing using Document Intelligence for text + direct image extraction.
        """
        try:
            # Get text sections from Document Intelligence (standard analysis)
            document, analysis_errors = self.docint_client.analyze_document_from_bytes(
                file_bytes=self.document_bytes,
                filename=self.filename
            )

            if analysis_errors:
                logging.warning(f"[doc_analysis_chunker][{self.filename}] Analysis had errors: {analysis_errors}")

            if not document:
                logging.error(f"[doc_analysis_chunker][{self.filename}] No document analysis result")
                # Fallback to traditional processing
                return await self._get_traditional_chunks()

            # Extract text content
            document_content = document.get('content', '')
            if not document_content:
                logging.warning(f"[doc_analysis_chunker][{self.filename}] No text content found")

            # Debug: Check if PageBreaks exist in content
            pagebreak_count = document_content.count('<!-- PageBreak -->')
            logging.debug(f"[doc_analysis_chunker][{self.filename}] Found {pagebreak_count} PageBreak markers in content")

            # Number pagebreaks for proper page tracking
            document_content = self._number_pagebreaks(document_content)

            # Process text chunks using character-based splitting with page info from document
            text_chunks = self._process_text_content_character_based(document_content, document)

            # Extract images directly using PyMuPDF and other libraries
            extracted_images = self.direct_image_extractor.extract_images_from_bytes(
                self.document_bytes,
                self.filename
            )

            logging.info(f"[doc_analysis_chunker][{self.filename}] Found {len(text_chunks)} text chunks and {len(extracted_images)} extracted images")

            # Process image chunks if images exist
            image_chunks = []
            if extracted_images and self.image_description_client:
                # Convert to normalized format
                normalized_images = self.direct_image_extractor.convert_to_normalized_format(
                    extracted_images,
                    self.url or f"file://{self.filename}"
                )
                image_chunks = await self._process_image_sections(normalized_images)

            # Combine all chunks
            all_chunks = text_chunks + image_chunks

            logging.info(f"[doc_analysis_chunker][{self.filename}] Generated {len(text_chunks)} text chunks and {len(image_chunks)} image chunks")

            return all_chunks

        except Exception as e:
            logging.error(f"[doc_analysis_chunker][{self.filename}] Error in multimodal processing: {str(e)}")
            # Fallback to traditional processing
            logging.info(f"[doc_analysis_chunker][{self.filename}] Falling back to traditional processing")
            return await self._get_traditional_chunks()

    def _process_text_content_character_based(self, document_content, document=None):
        """
        Process document content using character-based chunking strategy.
        Converts token-based configuration to approximate character limits.

        Args:
            document_content (str): Full document text content
            document (dict): Document analysis result with page information

        Returns:
            list: Processed text chunks
        """
        chunks = []
        chunk_id = 0

        # Convert token limits to approximate character limits
        # Average ratio: ~4 characters per token for English text
        chars_per_token = 4
        chunk_size = self.max_chunk_size * chars_per_token
        overlap = self.token_overlap * chars_per_token

        # Check if PageBreaks exist, if not use alternative page estimation
        has_pagebreaks = '<!-- PageBreak' in document_content
        logging.debug(f"[doc_analysis_chunker][{self.filename}] PageBreaks available: {has_pagebreaks}")

        # Split content into character-based chunks with page tracking
        start = 0
        content_length = len(document_content)
        current_page = 1

        while start < content_length:
            # Calculate end position for this chunk
            end = min(start + chunk_size, content_length)

            # Extract chunk content
            chunk_content = document_content[start:end]

            # Skip empty chunks
            if not chunk_content.strip():
                start = end - overlap if end < content_length else end
                continue

            # Determine page number
            if has_pagebreaks:
                # Use existing PageBreak tracking
                current_page = self._update_page(chunk_content, current_page)
                chunk_page = self._determine_chunk_page(chunk_content, current_page)
            else:
                # Estimate page based on character position and typical page length
                # Assume ~2000 characters per page (rough estimate)
                estimated_page = max(1, (start // 2000) + 1)
                chunk_page = estimated_page

            # Check minimum size requirement (convert to tokens for consistency)
            num_tokens = self.token_estimator.estimate_tokens(chunk_content)
            if num_tokens >= self.minimum_chunk_size:
                chunk_id += 1
                chunk = self._create_chunk(
                    chunk_id=chunk_id,
                    content=chunk_content.strip(),
                    page=chunk_page,  # Now uses proper page tracking or estimation
                    chunk_type='text',
                    location_metadata={
                        'start_char': start,
                        'end_char': end,
                        'chunk_method': 'character_based',
                        'page_estimation_method': 'pagebreak' if has_pagebreaks else 'character_position'
                    }
                )
                chunks.append(chunk)

            # Move to next chunk with overlap
            if end >= content_length:
                break
            start = end - overlap

        logging.debug(f"[doc_analysis_chunker][{self.filename}] Created {len(chunks)} character-based text chunks")
        return chunks


    async def _process_image_sections(self, normalized_images):
        """
        Process images from Document Layout analysis into chunks.

        Args:
            normalized_images (list): Images from Document Intelligence

        Returns:
            list: Processed image chunks
        """
        if not normalized_images:
            return []

        chunks = []
        organization_id = ""

        try:
            blob_client = BlobStorageClient(self.file_url)
            metadata = blob_client.get_metadata()
            organization_id = metadata.get('organization_id', '') if metadata else ''
            if organization_id:
                logging.debug(
                    f"[doc_analysis_chunker][{self.filename}] Using organization_id from metadata: {organization_id}"
                )
        except Exception as exc:
            logging.warning(
                f"[doc_analysis_chunker][{self.filename}] Unable to retrieve organization_id metadata: {exc}"
            )

        try:
            # Generate descriptions for all images
            described_images = self.image_description_client.describe_normalized_images(normalized_images)

            # Store images in blob storage
            storage_results = self.multimodal_blob_client.store_images_batch(
                described_images,
                self.url or f"file://{self.filename}"
            )

            # Create chunks for each image
            for i, (img, storage_result) in enumerate(zip(described_images, storage_results)):
                location_metadata = img.get('locationMetadata', {})
                description = img.get('description', 'Image description not available')

                # Log image processing for traceability
                logging.debug(f"[doc_analysis_chunker][{self.filename}] Processing image {i+1}: ID='{img.get('id')}', description_length={len(description) if description else 0}")

                # Handle storage result - image chunks are still valuable even without storage
                image_url = None
                if storage_result.get('success'):
                    image_url = storage_result.get('image_url')
                elif storage_result.get('skipped'):
                    # Log but continue - image descriptions are still useful
                    logging.info(f"[doc_analysis_chunker][{self.filename}] Image {img.get('id')} storage skipped but description preserved")
                else:
                    logging.warning(f"[doc_analysis_chunker][{self.filename}] Image {img.get('id')} storage failed: {storage_result.get('error', 'Unknown error')}")

                # Use the original image ID as chunk_id for consistency across pipeline
                original_image_id = img.get('id', f"img_{i+1}")

                # Get page number from multiple possible locations
                page_number = (
                    location_metadata.get('pageNumber') or
                    img.get('pageNumber') or
                    1
                )

                chunk = self._create_chunk(
                    chunk_id=original_image_id,
                    content=description,
                    page=page_number,
                    chunk_type='image',
                    location_metadata=location_metadata,
                    image_url=image_url,
                    image_id=original_image_id
                )
                if organization_id:
                    chunk['organization_id'] = organization_id
                chunks.append(chunk)

            logging.debug(f"[doc_analysis_chunker][{self.filename}] Created {len(chunks)} image chunks")
            return chunks

        except Exception as e:
            logging.error(f"[doc_analysis_chunker][{self.filename}] Error processing images: {e}")
            return []

    def _create_chunk(self, chunk_id, content, embedding_text="", title="", page=0,
                     chunk_type="text", location_metadata=None, image_url=None, image_id=None):
        """
        Enhanced _create_chunk method that supports multimodal chunks.

        Args:
            chunk_id: Sequential number for the chunk
            content: The main content of the chunk (text or image description)
            embedding_text: Text used to generate the embedding
            title: The title of the chunk
            page: The page number where the chunk is located
            chunk_type: Type of chunk ('text' or 'image')
            location_metadata: Location metadata from Document Intelligence
            image_url: URL to stored image (for image chunks)
            image_id: Unique identifier for image (for image chunks)

        Returns:
            dict: Enhanced chunk dictionary with multimodal support
        """
        # Get base chunk from parent class
        base_chunk = super()._create_chunk(
            chunk_id=chunk_id,
            content=content,
            embedding_text=embedding_text,
            title=title,
            page=page
        )

        # Add multimodal enhancements
        enhanced_chunk = base_chunk.copy()
        enhanced_chunk.update({
            "type": chunk_type,
            "location_metadata": location_metadata or {}
        })

        # Add image-specific fields for image chunks
        if chunk_type == "image":
            enhanced_chunk.update({
                "image_url": image_url,
                "image_id": image_id,
                "content_type": "image_description"
            })

        return enhanced_chunk


    def _process_document_chunks(self, document):
        """
        Processes the analyzed document content into manageable chunks.

        Args:
            document (dict): The analyzed document content provided by the Document Intelligence Client.

        Returns:
            list: A list of dictionaries, where each dictionary represents a processed chunk of the document content.

        The method performs the following steps:
        1. Prepares the document content for chunking, including numbering page breaks.
        2. Splits the content into chunks using a chosen splitting strategy.
        3. Iterates through the chunks, determining their page numbers and creating chunk representations.
        4. Skips chunks that do not meet the minimum size requirement.
        5. Logs the number of chunks created and skipped.
        """
        chunks = []
        document_content = document['content']
        document_content = self._number_pagebreaks(document_content)

        text_chunks = self._chunk_content(document_content)
        chunk_id = 0
        skipped_chunks = 0
        current_page = 1

        for text_chunk, num_tokens in text_chunks:
            current_page = self._update_page(text_chunk, current_page)
            chunk_page = self._determine_chunk_page(text_chunk, current_page)
            if num_tokens >= self.minimum_chunk_size:
                chunk_id += 1
                chunk = self._create_chunk(
                    chunk_id=chunk_id,
                    content=text_chunk,
                    page=chunk_page
                )
                chunks.append(chunk)
            else:
                skipped_chunks += 1

        logging.debug(f"[doc_analysis_chunker][{self.filename}] {len(chunks)} chunk(s) created")
        if skipped_chunks > 0:
            logging.debug(f"[doc_analysis_chunker][{self.filename}] {skipped_chunks} chunk(s) skipped")
        return chunks

    def _chunk_content(self, content):
        """
        Splits the document content into chunks based on the specified format and criteria.
        
        Yields:
            tuple: A tuple containing the chunked content and the number of tokens in the chunk.
        """
        content, placeholders, tables = self._replace_html_tables(content)
        splitter = self._choose_splitter()

        chunks = splitter.split_text(content)
        chunks = self._restore_original_tables(chunks, placeholders, tables)

        for chunked_content in chunks:
            chunk_size = self.token_estimator.estimate_tokens(chunked_content)
            if chunk_size > self.max_chunk_size:
                logging.info(f"[doc_analysis_chunker][{self.filename}] truncating {chunk_size} size chunk to fit within {self.max_chunk_size} tokens")
                chunked_content = self._truncate_chunk(chunked_content)

            yield chunked_content, chunk_size

    def _replace_html_tables(self, content):
        """
        Replaces HTML tables in the content with placeholders.
        
        Args:
            content (str): The document content.
        
        Returns:
            tuple: The content with placeholders and a list of the original tables.
        """
        table_pattern = r"(<table[\s\S]*?</table>)"
        tables = re.findall(table_pattern, content, re.IGNORECASE)
        placeholders = [f"__TABLE_{i}__" for i in range(len(tables))]
        for placeholder, table in zip(placeholders, tables):
            content = content.replace(table, placeholder)
        return content, placeholders, tables

    def _restore_original_tables(self, chunks, placeholders, tables):
        """
        Restores original tables in the chunks from placeholders.
        
        Args:
            chunks (list): The list of text chunks.
            placeholders (list): The list of table placeholders.
            tables (list): The list of original tables.
        
        Returns:
            list: The list of chunks with original tables restored.
        """
        for placeholder, table in zip(placeholders, tables):
            chunks = [chunk.replace(placeholder, table) for chunk in chunks]
        return chunks

    def _choose_splitter(self):
        """
        Chooses the appropriate splitter based on document format.
        
        Returns:
            object: The splitter to use for chunking.
        """
        if self.docint_client.output_content_format == "markdown":
            return MarkdownTextSplitter.from_tiktoken_encoder(
                chunk_size=self.max_chunk_size,
                chunk_overlap=self.token_overlap
            )
        else:
            separators = [".", "!", "?"] + [" ", "\n", "\t"]
            return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                separators=separators,
                chunk_size=self.max_chunk_size,
                chunk_overlap=self.token_overlap
            )

    def _number_pagebreaks(self, content):
        """
        Finds and numbers all PageBreaks in the content.
        
        Args:
            content (str): The document content.
        
        Returns:
            str: Content with numbered PageBreaks.
        """
        pagebreaks = re.findall(r'<!-- PageBreak -->', content)
        for i, _ in enumerate(pagebreaks, 1):
            content = content.replace('<!-- PageBreak -->', f'<!-- PageBreak{str(i).zfill(5)} -->', 1)
        return content

    def _update_page(self, content, current_page):
        """
        Updates the current page number based on the content.
        
        Args:
            content (str): The content chunk being processed.
            current_page (int): The current page number.
        
        Returns:
            int: The updated current page number.
        """
        matches = re.findall(r'PageBreak(\d{5})', content)
        if matches:
            page_number = int(matches[-1])
            if page_number >= current_page:
                current_page = page_number + 1
        return current_page

    def _determine_chunk_page(self, content, current_page):
        """
        Determines the chunk page number based on the position of the PageBreak element.
        
        Args:
            content (str): The content chunk being processed.
            current_page (int): The current page number.
        
        Returns:
            int: The page number for the chunk.
        """
        match = re.search(r'PageBreak(\d{5})', content)
        if match:
            page_number = int(match.group(1))
            position = match.start() / len(content)
            # Determine the chunk_page based on the position of the PageBreak element
            if position < 0.5:
                chunk_page = page_number + 1
            else:
                chunk_page = page_number
        else:
            chunk_page = current_page
        return chunk_page

    def _truncate_chunk(self, text):
        """
        Truncates and normalizes the text to ensure it fits within the maximum chunk size.
        
        This method first cleans up the text by removing unnecessary spaces and line breaks. 
        If the text still exceeds the maximum token limit, it iteratively truncates the text 
        until it fits within the limit.

        This method overrides the parent class's method because it includes logic to retain 
        PageBreaks within the truncated text.
        
        Args:
            text (str): The text to be truncated and normalized.
        
        Returns:
            str: The truncated and normalized text.
        """
        # Clean up text (e.g. line breaks)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[\n\r]+', ' ', text).strip()

        page_breaks = re.findall(r'PageBreak\d{5}', text)

        # Truncate if necessary
        if self.token_estimator.estimate_tokens(text) > self.max_chunk_size:
            logging.info(f"[doc_analysis_chunker][{self.filename}] token limit reached, truncating...")
            step_size = 1  # Initial step size
            iteration = 0  # Iteration counter

            while self.token_estimator.estimate_tokens(text) > self.max_chunk_size:
                # Truncate the text
                text = text[:-step_size]
                iteration += 1

                # Increase step size exponentially every 5 iterations
                if iteration % 5 == 0:
                    step_size = min(step_size * 2, 100)

        # Reinsert page breaks and recheck size
        for page_break in page_breaks:
            page_break_text = f" <!-- {page_break} -->"
            if page_break not in text:
                # Calculate the size needed for the page break addition
                needed_size = self.token_estimator.estimate_tokens(page_break_text)

                # Truncate exactly the size needed to accommodate the page break
                while self.token_estimator.estimate_tokens(text) + needed_size > self.max_chunk_size:
                    text = text[:-1]  # Remove one character at a time

                # Now add the page break
                text += page_break_text

        return text
