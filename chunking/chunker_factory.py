import logging
from .chunkers.doc_analysis_chunker import DocAnalysisChunker
from .chunkers.langchain_chunker import LangChainChunker


class ChunkerFactory:
    """Factory class to create appropriate chunker based on file extension."""

    def get_chunker(self, extension, data):
        """
        Get the appropriate chunker based on the file extension.

        Args:
            extension (str): The file extension.
            data (dict): The data containing document information.

        Returns:
            BaseChunker: An instance of a chunker class.
        """
        filename = data["documentUrl"].split("/")[-1]
        logging.info(f"[chunker_factory][{filename}] Creating chunker")

        extension = extension.lower()
        if extension in (
            "pdf",
            "png",
            "jpeg",
            "jpg",
            "bmp",
            "tiff",
            "docx",
            "pptx",
            "html",
        ):
            return DocAnalysisChunker(data)
        else:
            return LangChainChunker(data)

    @staticmethod
    def get_supported_extensions():
        """
        Get a comma-separated list of supported file extensions.

        Returns:
            str: A comma-separated list of supported file extensions.
        """
        extensions = [
            "pdf",
            "png",
            "jpeg",
            "jpg",
            "bmp",
            "tiff",
            "docx",
            "pptx",
            "html",
        ]
        return ", ".join(extensions)
