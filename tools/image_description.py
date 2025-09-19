# ImageDescriptionClient.py

import os
import base64
import json
import logging
import time
from typing import List, Dict, Any, Optional
import requests
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ClientAuthenticationError
from concurrent.futures import ThreadPoolExecutor, as_completed

class ImageDescriptionClient:
    """
    A client for generating technical descriptions of images using Azure OpenAI GPT-4V/GPT-4o.

    This replaces the Content Understanding service to avoid performance bottlenecks while
    maintaining the same functionality as Azure AI Search multimodal skillsets.
    """

    # System prompt from Azure AI Search skillset
    SYSTEM_PROMPT = """You are tasked with generating concise, accurate descriptions of images, figures, diagrams, or charts in documents. The goal is to capture the key information and meaning conveyed by the image without including extraneous details like style, colors, visual aesthetics, or size.

Instructions:
Content Focus: Describe the core content and relationships depicted in the image.

For diagrams, specify the main elements and how they are connected or interact.
For charts, highlight key data points, trends, comparisons, or conclusions.
For figures or technical illustrations, identify the components and their significance.
Clarity & Precision: Use concise language to ensure clarity and technical accuracy. Avoid subjective or interpretive statements.

Avoid Visual Descriptors: Exclude details about:

Colors, shading, and visual styles.
Image size, layout, or decorative elements.
Fonts, borders, and stylistic embellishments.
Context: If relevant, relate the image to the broader content of the technical document or the topic it supports.

Example Descriptions:
Diagram: "A flowchart showing the four stages of a machine learning pipeline: data collection, preprocessing, model training, and evaluation, with arrows indicating the sequential flow of tasks."

Chart: "A bar chart comparing the performance of four algorithms on three datasets, showing that Algorithm A consistently outperforms the others on Dataset 1."

Figure: "A labeled diagram illustrating the components of a transformer model, including the encoder, decoder, self-attention mechanism, and feedforward layers."
"""

    def __init__(self):
        """
        Initializes the ImageDescriptionClient.
        """
        # Azure OpenAI configuration
        self.azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        self.chat_deployment = os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT', 'Agent')
        self.api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2025-01-01-preview')

        if not self.azure_openai_endpoint:
            logging.error("[image_description] AZURE_OPENAI_ENDPOINT environment variable not set.")
            raise EnvironmentError("AZURE_OPENAI_ENDPOINT environment variable not set.")

        # Processing configuration
        self.batch_size = int(os.getenv('IMAGE_DESCRIPTION_BATCH_SIZE', '5'))
        self.timeout_seconds = int(os.getenv('IMAGE_DESCRIPTION_TIMEOUT', '30'))
        self.max_retries = int(os.getenv('IMAGE_DESCRIPTION_MAX_RETRIES', '3'))
        self.retry_delay = int(os.getenv('IMAGE_DESCRIPTION_RETRY_DELAY', '2'))

        # Authentication always relies on managed identity with CLI fallback.
        try:
            self.credential = DefaultAzureCredential()
            logging.debug("[image_description] Initialized managed identity credential chain for Azure OpenAI.")
        except Exception as e:
            logging.error(f"[image_description] Failed to initialize DefaultAzureCredential: {e}")
            raise

        logging.info(f"[image_description] Initialized with deployment: {self.chat_deployment}, batch size: {self.batch_size}")

    def _get_auth_headers(self) -> Dict[str, str]:
        """
        Get authentication headers for Azure OpenAI requests.

        Returns:
            Dict[str, str]: Headers with authentication
        """
        try:
            token = self.credential.get_token("https://cognitiveservices.azure.com/.default")
            return {
                "Authorization": f"Bearer {token.token}",
                "Content-Type": "application/json"
            }
        except ClientAuthenticationError as e:
            logging.error(f"[image_description] Authentication failed: {e}")
            raise

    def describe_single_image(self, image_data: str, image_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a description for a single image.

        Args:
            image_data (str): Base64 encoded image data
            image_id (Optional[str]): Identifier for the image (for logging)

        Returns:
            Dict[str, Any]: Result containing description or error
        """
        image_ref = image_id or "unknown"

        try:
            # Prepare the request
            url = f"{self.azure_openai_endpoint}/openai/deployments/{self.chat_deployment}/chat/completions?api-version={self.api_version}"

            headers = self._get_auth_headers()

            # Format the image data for Azure OpenAI
            if not image_data.startswith('data:image'):
                # Assume it's raw base64, add the data URL prefix
                image_url = f"data:image/jpeg;base64,{image_data}"
            else:
                image_url = image_data

            payload = {
                "messages": [
                    {
                        "role": "system",
                        "content": self.SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Please describe this image."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.1  # Low temperature for consistent, factual descriptions
            }

            # Make the request with retries
            for attempt in range(self.max_retries):
                try:
                    response = requests.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=self.timeout_seconds
                    )

                    if response.status_code == 200:
                        result = response.json()
                        description = result['choices'][0]['message']['content'].strip()

                        logging.debug(f"[image_description] Successfully described image {image_ref}")
                        return {
                            'success': True,
                            'description': description,
                            'image_id': image_id
                        }

                    elif response.status_code == 429:  # Rate limit
                        wait_time = self.retry_delay * (2 ** attempt)
                        logging.warning(f"[image_description] Rate limited for image {image_ref}, waiting {wait_time}s")
                        time.sleep(wait_time)
                        continue

                    else:
                        error_msg = f"Azure OpenAI request failed with status {response.status_code}: {response.text}"
                        logging.error(f"[image_description] {error_msg}")
                        return {
                            'success': False,
                            'error': error_msg,
                            'image_id': image_id
                        }

                except requests.exceptions.Timeout:
                    if attempt < self.max_retries - 1:
                        logging.warning(f"[image_description] Timeout for image {image_ref}, retrying...")
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        error_msg = f"Request timeout after {self.max_retries} attempts"
                        logging.error(f"[image_description] {error_msg}")
                        return {
                            'success': False,
                            'error': error_msg,
                            'image_id': image_id
                        }

                except Exception as e:
                    if attempt < self.max_retries - 1:
                        logging.warning(f"[image_description] Request error for image {image_ref}: {e}, retrying...")
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        error_msg = f"Request failed after {self.max_retries} attempts: {e}"
                        logging.error(f"[image_description] {error_msg}")
                        return {
                            'success': False,
                            'error': error_msg,
                            'image_id': image_id
                        }

            # Should not reach here, but just in case
            return {
                'success': False,
                'error': 'Unexpected error in request loop',
                'image_id': image_id
            }

        except Exception as e:
            error_msg = f"Unexpected error describing image {image_ref}: {e}"
            logging.error(f"[image_description] {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'image_id': image_id
            }

    def describe_images_batch(self, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate descriptions for multiple images in parallel.

        Args:
            images (List[Dict[str, Any]]): List of image objects with 'data' and optionally 'id'

        Returns:
            List[Dict[str, Any]]: List of results for each image
        """
        if not images:
            return []

        logging.info(f"[image_description] Starting batch description of {len(images)} images")

        results = []

        # Process in batches to avoid overwhelming the API
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]

            logging.debug(f"[image_description] Processing batch {i // self.batch_size + 1} with {len(batch)} images")

            # Use ThreadPoolExecutor for parallel processing while preserving ID mapping
            with ThreadPoolExecutor(max_workers=min(len(batch), 5)) as executor:
                # Submit all tasks
                future_to_image = {
                    executor.submit(
                        self.describe_single_image,
                        img.get('data', ''),
                        img.get('id')
                    ): img for img in batch
                }

                # Collect results and preserve image ID association
                batch_results = []
                for future in as_completed(future_to_image):
                    result = future.result()
                    original_image = future_to_image[future]

                    # Include the original image ID in the result for mapping
                    result['original_image_id'] = original_image.get('id')
                    batch_results.append(result)

                results.extend(batch_results)

            # Small delay between batches to be respectful to the API
            if i + self.batch_size < len(images):
                time.sleep(1)

        success_count = sum(1 for r in results if r.get('success', False))
        failure_count = len(results) - success_count

        logging.info(f"[image_description] Batch completed: {success_count} successful, {failure_count} failed")

        return results

    def describe_normalized_images(self, normalized_images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process normalized images from Document Intelligence Layout skill.

        Args:
            normalized_images (List[Dict[str, Any]]): Images in Document Intelligence format

        Returns:
            List[Dict[str, Any]]: Enhanced images with descriptions
        """
        if not normalized_images:
            logging.info("[image_description] No images to process")
            return []

        logging.info(f"[image_description] Processing {len(normalized_images)} normalized images")

        # Prepare images for batch processing
        images_for_description = []
        for img in normalized_images:
            if 'data' in img:
                images_for_description.append({
                    'data': img['data'],
                    'id': img.get('id', 'unknown')
                })

        # Get descriptions
        description_results = self.describe_images_batch(images_for_description)

        # Create a mapping from image ID to description result for correct matching
        id_to_description = {}
        for result in description_results:
            image_id = result.get('original_image_id')
            if image_id:
                id_to_description[image_id] = result

        # Merge descriptions back into original image objects by ID (not index)
        enhanced_images = []
        for img in normalized_images:
            enhanced_img = img.copy()
            image_id = img.get('id')

            # Find the matching description by image ID
            result = id_to_description.get(image_id)
            if result:
                if result.get('success'):
                    enhanced_img['description'] = result['description']
                    enhanced_img['description_status'] = 'success'
                    logging.debug(f"[image_description] Matched description for {image_id}: {len(result['description'])} chars")
                else:
                    enhanced_img['description'] = f"Error generating description: {result.get('error', 'Unknown error')}"
                    enhanced_img['description_status'] = 'error'
            else:
                enhanced_img['description'] = "Description not generated"
                enhanced_img['description_status'] = 'skipped'
                logging.warning(f"[image_description] No description found for image ID: {image_id}")

            enhanced_images.append(enhanced_img)

        return enhanced_images
