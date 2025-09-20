# MultimodalBlobClient.py

import os
import logging
import base64
import hashlib
import time
import json
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ResourceExistsError

class MultimodalBlobClient:
    """
    Enhanced blob storage client for multimodal document processing.

    Handles storage of extracted images with consistent naming and metadata,
    replicating Azure AI Search knowledge store projection functionality.
    """

    def __init__(self):
        """
        Initialize the multimodal blob client with managed identity authentication.
        """
        self.images_container = os.getenv('IMAGES_CONTAINER_NAME', 'ragindex-test-images')
        self.blob_service_client = None
        self.credential = None
        self.initialized = False
        self.storage_account_url = None

        # Storage configuration
        account_name = os.getenv('AZURE_STORAGE_ACCOUNT')
        if not account_name:
            logging.warning("[multimodal_blob] AZURE_STORAGE_ACCOUNT not provided. Image storage will be disabled.")
            return

        self.storage_account_url = f"https://{account_name}.blob.core.windows.net"

        try:
            # Initialize credential and blob service client with managed identity
            self.credential = DefaultAzureCredential()
            self.blob_service_client = BlobServiceClient(
                account_url=self.storage_account_url,
                credential=self.credential
            )
            self.initialized = True
            logging.info("[multimodal_blob] Initialized with managed identity authentication.")
        except Exception as e:
            logging.error(f"[multimodal_blob] Failed to initialize blob service client: {e}")
            logging.info("[multimodal_blob] This may be due to DNS resolution or authentication issues in local development.")


    def _ensure_container_exists(self):
        """
        Ensure the images container exists, create if it doesn't.
        """
        if not self.initialized:
            return False

        try:
            container_client = self.blob_service_client.get_container_client(self.images_container)
            container_client.get_container_properties()
            logging.debug(f"[multimodal_blob] Container '{self.images_container}' exists.")
            return True
        except Exception:
            try:
                self.blob_service_client.create_container(self.images_container)
                logging.info(f"[multimodal_blob] Created container '{self.images_container}'.")
                return True
            except ResourceExistsError:
                logging.debug(f"[multimodal_blob] Container '{self.images_container}' already exists.")
                return True
            except Exception as e:
                logging.error(f"[multimodal_blob] Failed to create container '{self.images_container}': {e}")
                self.initialized = False
                return False

    def _generate_image_path(self, document_url: str, image_id: str, image_format: str = 'jpg') -> str:
        """
        Generate a consistent image path following Azure AI Search knowledge store pattern.

        Args:
            document_url (str): Original document URL
            image_id (str): Unique image identifier
            image_format (str): Image format extension

        Returns:
            str: Generated image path
        """
        # Extract document name from URL
        parsed_url = urlparse(document_url)
        document_name = os.path.basename(parsed_url.path)
        document_name_no_ext = os.path.splitext(document_name)[0]

        # Create a hash-based subdirectory for organization
        doc_hash = hashlib.md5(document_url.encode()).hexdigest()[:8]

        # Generate path: document_hash/document_name/image_id.format
        image_path = f"{doc_hash}/{document_name_no_ext}/{image_id}.{image_format}"

        return image_path

    def store_image(
        self,
        image_data: str,
        document_url: str,
        image_id: str,
        location_metadata: Optional[Dict[str, Any]] = None,
        organization_id: str = "",
    ) -> Dict[str, Any]:
        """
        Store an extracted image in blob storage.

        Args:
            image_data (str): Base64 encoded image data
            document_url (str): URL of the source document
            image_id (str): Unique identifier for the image
            location_metadata (Optional[Dict[str, Any]]): Image location metadata from Document Intelligence

        Returns:
            Dict[str, Any]: Storage result with URL and metadata
        """
        # If blob storage is not initialized, skip storage but don't fail
        if not self.initialized:
            logging.warning(f"[multimodal_blob] Blob storage not available, skipping image {image_id} storage")
            return {
                'success': False,
                'error': 'Blob storage not available',
                'image_id': image_id,
                'skipped': True
            }

        try:
            # Decode base64 image data
            image_bytes = base64.b64decode(image_data)

            # Generate image path
            image_path = self._generate_image_path(document_url, image_id)

            # Prepare metadata
            blob_metadata = {
                'source_document': document_url,
                'image_id': image_id,
                'content_type': 'image/jpeg',
                'extraction_timestamp': str(int(time.time()))
            }

            if organization_id:
                blob_metadata['organization_id'] = organization_id

            # Add location metadata if provided
            if location_metadata:
                bounding_polygons = location_metadata.get('boundingPolygons')
                if bounding_polygons is not None:
                    try:
                        bounding_polygons_value = json.dumps(bounding_polygons)
                    except TypeError:
                        bounding_polygons_value = str(bounding_polygons)
                else:
                    bounding_polygons_value = ''

                blob_metadata.update({
                    'page_number': str(location_metadata.get('pageNumber', 0)),
                    'ordinal_position': str(location_metadata.get('ordinalPosition', 0)),
                    'bounding_polygons': bounding_polygons_value
                })

            # Upload blob
            blob_client = self.blob_service_client.get_blob_client(
                container=self.images_container,
                blob=image_path
            )

            blob_client.upload_blob(
                image_bytes,
                content_type='image/jpeg',
                metadata=blob_metadata,
                overwrite=True
            )

            # Generate public URL
            image_url = f"{self.storage_account_url}/{self.images_container}/{image_path}"

            logging.debug(f"[multimodal_blob] Stored image {image_id} at {image_path}")

            return {
                'success': True,
                'image_url': image_url,
                'image_path': image_path,
                'container': self.images_container,
                'size_bytes': len(image_bytes)
            }

        except Exception as e:
            error_msg = f"Failed to store image {image_id}: {e}"
            logging.error(f"[multimodal_blob] {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'image_id': image_id
            }

    def store_images_batch(
        self,
        images: List[Dict[str, Any]],
        document_url: str,
        organization_id: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Store multiple images from normalized_images array.

        Args:
            images (List[Dict[str, Any]]): List of normalized images from Document Intelligence
            document_url (str): URL of the source document

        Returns:
            List[Dict[str, Any]]: List of storage results
        """
        if not images:
            return []

        logging.info(f"[multimodal_blob] Storing {len(images)} images for document {document_url}")

        results = []
        for img in images:
            image_data = img.get('data', '')
            image_id = img.get('id', f"image_{len(results)}")
            location_metadata = img.get('locationMetadata')

            if not image_data:
                logging.warning(f"[multimodal_blob] No image data for {image_id}, skipping")
                results.append({
                    'success': False,
                    'error': 'No image data provided',
                    'image_id': image_id
                })
                continue

            result = self.store_image(
                image_data,
                document_url,
                image_id,
                location_metadata,
                organization_id=organization_id,
            )
            results.append(result)

        success_count = sum(1 for r in results if r.get('success', False))
        logging.info(f"[multimodal_blob] Stored {success_count}/{len(results)} images successfully")

        return results

    def get_image_url(self, document_url: str, image_id: str) -> str:
        """
        Get the URL for a stored image.

        Args:
            document_url (str): Original document URL
            image_id (str): Image identifier

        Returns:
            str: Image URL
        """
        image_path = self._generate_image_path(document_url, image_id)
        return f"{self.storage_account_url}/{self.images_container}/{image_path}"

    def delete_document_images(self, document_url: str) -> Dict[str, Any]:
        """
        Delete all images associated with a document.

        Args:
            document_url (str): Document URL

        Returns:
            Dict[str, Any]: Deletion result
        """
        try:
            # Generate prefix for all images from this document
            parsed_url = urlparse(document_url)
            document_name = os.path.basename(parsed_url.path)
            document_name_no_ext = os.path.splitext(document_name)[0]
            doc_hash = hashlib.md5(document_url.encode()).hexdigest()[:8]
            prefix = f"{doc_hash}/{document_name_no_ext}/"

            # List and delete blobs with this prefix
            container_client = self.blob_service_client.get_container_client(self.images_container)
            blobs = container_client.list_blobs(name_starts_with=prefix)

            deleted_count = 0
            for blob in blobs:
                try:
                    container_client.delete_blob(blob.name)
                    deleted_count += 1
                except Exception as e:
                    logging.warning(f"[multimodal_blob] Failed to delete blob {blob.name}: {e}")

            logging.info(f"[multimodal_blob] Deleted {deleted_count} images for document {document_url}")

            return {
                'success': True,
                'deleted_count': deleted_count
            }

        except Exception as e:
            error_msg = f"Failed to delete images for document {document_url}: {e}"
            logging.error(f"[multimodal_blob] {error_msg}")
            return {
                'success': False,
                'error': error_msg
            }
