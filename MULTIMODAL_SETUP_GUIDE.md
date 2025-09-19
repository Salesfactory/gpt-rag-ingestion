# Multimodal Document Processing Setup Guide

## Overview

This guide explains how to enable and configure the new multimodal document processing capabilities that extract and process both text and images from documents.

## Features

- **Document Layout Analysis**: Uses Azure Document Intelligence Layout model for structured text and image extraction
- **Image Description**: Generates technical descriptions of charts, diagrams, and figures using Azure OpenAI GPT-4V/GPT-4o
- **Image Storage**: Stores extracted images in Azure Blob Storage with consistent naming
- **Dual Embeddings**: Creates vector embeddings for both text content and image descriptions
- **Location Metadata**: Preserves page numbers and bounding polygon information for precise content location

## Configuration

### Environment Variables

Copy `local.settings.multimodal.json.template` to `local.settings.json` and configure these variables:

#### Document Intelligence
```json
"AZURE_FORMREC_SERVICE": "your-doc-intelligence-service-name",
"FORM_REC_API_VERSION": "2024-11-30",
"MULTIMODAL_PROCESSING_ENABLED": "true",
```

#### Azure OpenAI for Image Description
```json
"AZURE_OPENAI_ENDPOINT": "https://your-openai-service.openai.azure.com/",
"AZURE_OPENAI_CHAT_DEPLOYMENT": "gpt-4o",
"AZURE_OPENAI_API_VERSION": "2025-01-01-preview",
"IMAGE_DESCRIPTION_BATCH_SIZE": "5",
"IMAGE_DESCRIPTION_TIMEOUT": "30"
```

#### Blob Storage for Images
```json
"AZURE_STORAGE_ACCOUNT": "yourstorageaccount",
"IMAGES_CONTAINER_NAME": "extracted-images"
```
The ingestion service automatically builds the endpoint as `https://<account>.blob.core.windows.net`, so only the storage account name is required here.

### Azure Services Required

1. **Azure Document Intelligence**: Standard (S0) tier with Layout model support
2. **Azure OpenAI**: With GPT-4V/GPT-4o deployment for image descriptions
3. **Azure Blob Storage**: For storing extracted images
4. **Azure AI Search**: For indexing multimodal content

### Region Requirements

The multimodal functionality requires specific Azure regions:
- **East US**, **West Europe**, **North Central US**, or **West US 2**

## Output Format

### Text Chunks
```json
{
  "id": "unique-guid",
  "type": "text",
  "content": "extracted text content",
  "vector": [embedding-array],
  "location_metadata": {
    "pageNumber": 1,
    "ordinalPosition": 0,
    "boundingPolygons": "coordinate-data"
  },
  "page": 1,
  "chunk_id": 1
}
```

### Image Chunks
```json
{
  "id": "unique-guid",
  "type": "image",
  "content": "AI-generated image description",
  "vector": [embedding-array],
  "image_url": "https://storage.blob.core.windows.net/images/path/to/image.jpg",
  "image_id": "extracted-image-id",
  "location_metadata": {
    "pageNumber": 1,
    "ordinalPosition": 0,
    "boundingPolygons": "coordinate-data"
  },
  "page": 1,
  "chunk_id": "img_1"
}
```

## Performance Considerations

### Processing Times
- **Document Layout Analysis**: 30-60 seconds for typical documents
- **Image Description**: 2-5 seconds per image
- **Total Processing**: Varies based on document size and image count

### Cost Optimization
- **Batch Processing**: Images are processed in configurable batches
- **Selective Processing**: Can be enabled/disabled via configuration
- **Fallback Mode**: Automatically falls back to text-only processing if multimodal fails

### Limits
- **Document Size**: 500 MB for paid tier, 4 MB for free tier
- **Image Count**: No specific limit, but affects processing time
- **API Rate Limits**: Configurable retry logic with exponential backoff

## Troubleshooting

### Common Issues

1. **"Multimodal processing disabled"**
   - Check `MULTIMODAL_PROCESSING_ENABLED=true`
   - Verify Document Intelligence API version is `2025-05-01-preview` or later

2. **"Image description failed"**
   - Verify Azure OpenAI GPT-4V/GPT-4o deployment
   - Check API keys and endpoint configuration
   - Review rate limiting settings

3. **"Image storage failed"**
   - Confirm `AZURE_STORAGE_ACCOUNT` matches your storage account name
   - Check container permissions and existence (the service will create it if allowed)
   - Ensure managed identity has Storage Blob Data Contributor role

### Logging

Enable debug logging to troubleshoot issues:
```json
"LOG_LEVEL": "DEBUG"
```

### Fallback Behavior

If multimodal processing fails, the system automatically falls back to traditional text-only processing to ensure reliability.

## Testing

### Sample Documents
Test with documents containing:
- **Charts and graphs**: To verify data extraction descriptions
- **Diagrams**: To test technical illustration processing
- **Mixed content**: Documents with both text and images

### Validation
- Check both text and image chunks are generated
- Verify image URLs are accessible
- Confirm embeddings are created for both content types
- Test location metadata preservation

## Monitoring

### Key Metrics
- **Processing time per document**
- **Text vs image chunk ratios**
- **Image description success rates**
- **Storage operation success rates**

### Alerts
- Set up alerts for processing failures
- Monitor storage costs for extracted images
- Track API rate limit violations

## Migration from Content Understanding

If migrating from Azure Content Understanding:
1. Update environment variables as shown above
2. Test with sample documents
3. Compare output quality and performance
4. Gradually roll out to production workloads

The new implementation provides better performance and cost control while maintaining equivalent functionality.
