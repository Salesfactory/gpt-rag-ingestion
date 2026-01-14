import logging
import json
import os
import time
import datetime

import jsonschema
import azure.functions as func

from chunking import DocumentChunker
from azurefunctions.extensions.http.fastapi import Request, Response
from tools import BlobStorageClient, AISearchClient
from utils.file_utils import get_filename, infer_content_type_from_url
from utils.schemas import DateTimeEncoder, get_document_chunking_request_schema
from survey import process_json_to_markdown_in_memory

# -------------------------------
# Logging configuration
# -------------------------------
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level, logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
suppress_loggers = [
    "azure",
    "azure.core",
    "azure.core.pipeline",
    "azure.core.pipeline.policies.http_logging_policy",
    "azsdk-python-search-documents",
    "azsdk-python-identity",
    "azure.ai.openai",  # Assuming 'aoai' refers to Azure OpenAI
    "azure.identity",
    "azure.storage",
    "azure.ai.*",  # Wildcard-like suppression for any azure.ai sub-loggers
    # Add any other specific loggers if necessary
]
for logger_name in suppress_loggers:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.WARNING)
    logger.propagate = False

# -------------------------------
# Azure Functions
# -------------------------------

app = func.FunctionApp()


# -------------------------------
# Health Check Endpoint
# -------------------------------


@app.route(route="health", methods=[func.HttpMethod.GET], auth_level=func.AuthLevel.ANONYMOUS)
async def health_check(req: Request) -> Response:
    """
    Health check endpoint for Azure App Service health monitoring.
    pinged by Azure's health check feature at 1-minute intervals

    Returns:
        200 OK when the application is healthy
    """
    return Response("OK", status_code=200, media_type="text/plain")


# -------------------------------
# Event Grid Trigger Function (for json-intermediate)
# -------------------------------


@app.event_grid_trigger(arg_name="event")
@app.queue_output(
    arg_name="msg", queue_name="survey-processing", connection="AzureWebJobsStorage"
)
def EventGridTrigger(event: func.EventGridEvent, msg: func.Out[str]):
    """
    Handles blob creation events for survey JSON files.
    Queues JSON files from survey-json-intermediate container for long-running processing.
    """
    event_type = event.event_type
    blob_url = event.subject

    logging.info(f"[Ingestion-EventGrid] Event: {event_type}, Subject: {blob_url}")

    if event_type == "Microsoft.Storage.BlobCreated":
        message_data = {
            "blobUrl": blob_url,
            "eventType": event_type,
            "eventTime": event.event_time.isoformat() if event.event_time else None,
        }
        msg.set(json.dumps(message_data))
        logging.info(
            f"[EventGridTrigger] Queued survey JSON for processing: {blob_url}"
        )


# -------------------------------
# Event Grid Trigger for survey-markdown indexing
# -------------------------------


@app.event_grid_trigger(arg_name="event")
async def EventGridTriggerSurveyMarkdownIndexer(
    event: func.EventGridEvent,
):
    """
    Triggers an Azure AI Search indexer run when a new blob arrives in survey-markdown.
    """
    event_type = event.event_type
    blob_subject = event.subject or ""

    if event_type != "Microsoft.Storage.BlobCreated":
        logging.debug(f"[survey-markdown-indexer] Ignoring event type: {event_type}")
        return

    pulse_indexer_name = "pulse-indexer"
    try:
        async with AISearchClient() as client:
            await client.run_indexer(pulse_indexer_name)
            logging.info(
                f"[survey-markdown-indexer] Triggered indexer '{pulse_indexer_name}' for {blob_subject}"
            )
    except Exception as exc:
        logging.error(
            f"[survey-markdown-indexer] Failed to run indexer '{pulse_indexer_name}': {exc}",
            exc_info=True,
        )


# -------------------------------
# Survey JSON Queue Processor (pretty long running ops)
# -------------------------------


@app.queue_trigger(
    arg_name="msg", queue_name="survey-processing", connection="AzureWebJobsStorage"
)
async def process_survey_queue(msg: func.QueueMessage):
    """
    Processes survey JSON files from queue (can take 90+ minutes).
    Downloads JSON, converts to markdown using OpenAI, uploads result.
    """
    start_time = time.time()

    try:
        message_data = json.loads(msg.get_body().decode("utf-8"))
        blob_url = message_data["blobUrl"]

        # event grid Format: /blobServices/default/containers/survey-json-intermediate/blobs/filename.json
        filename = blob_url.split("/blobs/")[-1]
        base_name = filename.replace(".json", "")

        logging.info(f"[process_survey_queue] Processing started: {filename}")

        # Download JSON from blob storage
        storage_account = os.getenv("AZURE_STORAGE_ACCOUNT")
        blob_download_url = f"https://{storage_account}.blob.core.windows.net/survey-json-intermediate/{filename}"
        blob_client = BlobStorageClient(file_url=blob_download_url)
        json_bytes = blob_client.download_blob()
        json_str = json_bytes.decode("utf-8")
        grouped_records = json.loads(json_str)

        # Get source metadata
        source_metadata = blob_client.get_metadata()
        source_file_directory = source_metadata.get("source_file_directory", "")
        source_file_name = source_metadata.get("source_file_name", "")
        source_file_container = source_metadata.get("source_file_container", "")

        logging.info(
            f"[process_survey_queue][{filename}] Loaded {len(grouped_records)} records"
        )

        model = os.getenv("PULSE_SERIALIZATION_MODEL", "gpt-4.1-mini")
        max_concurrent = int(os.getenv("PULSE_MAX_CONCURRENT", "20"))

        markdown_content = await process_json_to_markdown_in_memory(
            grouped_records=grouped_records,
            filename=filename,
            model=model,
            max_concurrent=max_concurrent,
        )

        output_container = "survey-markdown"
        output_filename = f"{base_name}.md"
        output_url = f"https://{storage_account}.blob.core.windows.net/{output_container}/{output_filename}"

        elapsed_time = time.time() - start_time

        # prep metadata for output blob
        output_metadata = {
            "source_file_directory": source_file_directory,
            "source_file_container": source_file_container,
            "source_file_name": source_file_name,
            "duration_seconds": str(round(elapsed_time, 2)),
            "processed_at": datetime.datetime.fromtimestamp(start_time).isoformat(),
        }

        output_blob_client = BlobStorageClient(output_url)
        output_blob_client.upload_blob(
            data=markdown_content.encode("utf-8"),
            overwrite=True,
            content_type="text/markdown",
            metadata=output_metadata,
        )

        logging.info(
            f"[process_survey_queue][{filename}] Completed successfully in {elapsed_time:.2f} seconds. "
            f"Output: {output_filename}"
        )

    except json.JSONDecodeError as e:
        logging.error(f"[process_survey_queue] Invalid JSON format: {e}", exc_info=True)
        raise
    except Exception as e:
        elapsed_time = time.time() - start_time
        logging.error(
            f"[process_survey_queue] Processing failed after {elapsed_time:.2f} seconds: {e}",
            exc_info=True,
        )
        raise


# -------------------------------
# Survey JSON Processing HTTP Trigger (for local development)
# -------------------------------


@app.route(route="process-survey-local", auth_level=func.AuthLevel.FUNCTION)
async def process_survey_http(req: func.HttpRequest) -> func.HttpResponse:
    """
    HTTP trigger for local testing of survey JSON processing.
    Downloads blob from survey-json-intermediate, processes it, and uploads to survey-markdown.

    Request body:
    {
        "blobName": "filename.json"
    }
    """
    start_time = time.time()

    try:
        body = req.get_json()
        blob_name = body.get("blobName")

        if not blob_name:
            return func.HttpResponse(
                json.dumps({"error": "Missing 'blobName' in request body"}),
                status_code=400,
                mimetype="application/json",
            )

        if not blob_name.endswith(".json"):
            blob_name = f"{blob_name}.json"

        base_name = blob_name.replace(".json", "")
        logging.info(f"[process_survey_http] Processing: {blob_name}")

        storage_account = os.getenv("AZURE_STORAGE_ACCOUNT")

        input_url = f"https://{storage_account}.blob.core.windows.net/survey-json-intermediate/{blob_name}"
        input_blob_client = BlobStorageClient(input_url)
        json_bytes = input_blob_client.download_blob()
        json_str = json_bytes.decode("utf-8")
        grouped_records = json.loads(json_str)

        # Get source metadata
        source_metadata = input_blob_client.get_metadata()
        source_file_directory = source_metadata.get("source_file_directory", "")
        source_file_container = source_metadata.get("source_file_container", "")
        source_file_name = source_metadata.get("source_file_name", "")

        logging.info(
            f"[process_survey_http][{blob_name}] Loaded {len(grouped_records)} records"
        )

        model = os.getenv("PULSE_SERIALIZATION_MODEL", "gpt-4.1-mini")
        max_concurrent = int(os.getenv("PULSE_MAX_CONCURRENT", "20"))

        markdown_content = await process_json_to_markdown_in_memory(
            grouped_records=grouped_records,
            filename=blob_name,
            model=model,
            max_concurrent=max_concurrent,
        )

        output_filename = f"{base_name}.md"
        output_url = f"https://{storage_account}.blob.core.windows.net/survey-markdown/{output_filename}"

        elapsed_time = time.time() - start_time

        # prep metadata for output blob
        output_metadata = {
            "source_file_directory": source_file_directory,
            "source_file_container": source_file_container,
            "source_file_name": source_file_name,
            "duration_seconds": str(round(elapsed_time, 2)),
            "processed_at": datetime.datetime.fromtimestamp(start_time).isoformat(),
        }

        output_blob_client = BlobStorageClient(output_url)
        output_blob_client.upload_blob(
            data=markdown_content.encode("utf-8"),
            overwrite=True,
            content_type="text/markdown",
            metadata=output_metadata,
        )

        response = {
            "status": "success",
            "inputFile": blob_name,
            "outputFile": output_filename,
            "recordsProcessed": len(grouped_records),
            "elapsedTimeSeconds": round(elapsed_time, 2),
        }

        logging.info(
            f"[process_survey_http][{blob_name}] Completed in {elapsed_time:.2f} seconds"
        )

        return func.HttpResponse(
            json.dumps(response, indent=2), status_code=200, mimetype="application/json"
        )

    except ValueError as e:
        error_msg = f"Invalid request: {str(e)}"
        logging.error(f"[process_survey_http] {error_msg}", exc_info=True)
        return func.HttpResponse(
            json.dumps({"error": error_msg}),
            status_code=400,
            mimetype="application/json",
        )
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON format: {str(e)}"
        logging.error(f"[process_survey_http] {error_msg}", exc_info=True)
        return func.HttpResponse(
            json.dumps({"error": error_msg}),
            status_code=400,
            mimetype="application/json",
        )
    except Exception as e:
        elapsed_time = time.time() - start_time
        error_msg = f"Processing failed: {str(e)}"
        logging.error(
            f"[process_survey_http] {error_msg} (after {elapsed_time:.2f}s)",
            exc_info=True,
        )
        return func.HttpResponse(
            json.dumps(
                {"error": error_msg, "elapsedTimeSeconds": round(elapsed_time, 2)}
            ),
            status_code=500,
            mimetype="application/json",
        )


# -------------------------------
# Document Chunking Function (HTTP Triggered by AI Search)
# -------------------------------


# Document Chunking Function (HTTP Triggered by AI Search)
@app.route(route="document-chunking", auth_level=func.AuthLevel.FUNCTION)
async def document_chunking(req: func.HttpRequest) -> func.HttpResponse:
    try:
        body = req.get_json()
        jsonschema.validate(body, schema=get_document_chunking_request_schema())

        if body:
            # Log the incoming request
            logging.info(
                f'[document_chunking_function] Invoked document_chunking skill. Number of items: {len(body["values"])}.'
            )

            input_data = {}

            # Processing one item at a time to avoid exceeding the AI Search custom skill timeout (230 seconds)
            # BatchSize should be set to 1 in the Skillset definition, if it is not set, will process just the last item
            count_items = len(body["values"])
            filename = ""
            if count_items > 1:
                logging.warning(
                    "BatchSize should be set to 1 in the Skillset definition. Processing only the last item."
                )
            for _, item in enumerate(body["values"]):
                input_data = item["data"]
                filename = get_filename(input_data["documentUrl"])

                # Handle missing or generic documentContentType by inferring from file extension
                content_type = input_data.get("documentContentType", "")

                if not content_type:
                    # Content type is missing entirely
                    inferred_type = infer_content_type_from_url(
                        input_data["documentUrl"]
                    )
                    input_data["documentContentType"] = inferred_type
                    logging.warning(
                        f"[document_chunking_function] documentContentType missing for {filename}. "
                        f"Inferred from file extension: {inferred_type}"
                    )
                elif content_type == "application/octet-stream":
                    # Content type is generic/unknown - try to infer better type from extension
                    inferred_type = infer_content_type_from_url(
                        input_data["documentUrl"]
                    )
                    if inferred_type != "application/octet-stream":
                        input_data["documentContentType"] = inferred_type
                        logging.warning(
                            f"[document_chunking_function] documentContentType was generic (application/octet-stream) for {filename}. "
                            f"Inferred more specific type from file extension: {inferred_type}"
                        )

                logging.info(
                    f'[document_chunking_function] Chunking document: File {filename}, Content Type {input_data["documentContentType"]}.'
                )

            start_time = time.time()

            # Enrich the input data with the document bytes and file name
            blob_client = BlobStorageClient(input_data["documentUrl"])
            document_bytes = blob_client.download_blob()
            input_data["documentBytes"] = document_bytes
            input_data["fileName"] = filename

            # Chunk the document
            chunks, errors, warnings = await DocumentChunker().chunk_documents(
                input_data
            )

            # Debug logging and multimodal summary
            text_chunks = 0
            image_chunks = 0
            for idx, chunk in enumerate(chunks):
                chunk_type = chunk.get("type", "text")
                if chunk_type == "image":
                    image_chunks += 1
                else:
                    text_chunks += 1

                processed_chunk = chunk.copy()
                processed_chunk.pop("vector", None)
                if "content" in processed_chunk and isinstance(
                    processed_chunk["content"], str
                ):
                    processed_chunk["content"] = processed_chunk["content"][:100]

                # Add multimodal specific info to debug output
                if chunk_type == "image":
                    image_url = chunk.get("image_url", "Not available")
                    if image_url and image_url != "Not available":
                        processed_chunk["image_url_preview"] = image_url[:100]
                    else:
                        processed_chunk["image_url_preview"] = "Not available"

                logging.debug(
                    f"[document_chunking][{filename}] {chunk_type.title()} Chunk {idx + 1}: {json.dumps(processed_chunk, indent=4)}"
                )

            logging.info(
                f"[document_chunking][{filename}] Generated {len(chunks)} total chunks: {text_chunks} text, {image_chunks} image"
            )

            # Filter vectors from response if requested
            include_vectors = input_data.get("includeVectors", True)
            if not include_vectors:
                logging.info(
                    f"[document_chunking][{filename}] Excluding vectors from response (includeVectors=False)"
                )
                chunks = [
                    {k: v for k, v in chunk.items() if k != "vector"}
                    for chunk in chunks
                ]

            # Format results
            values = {
                "recordId": item["recordId"],
                "data": {"chunks": chunks},
                "errors": errors,
                "warnings": warnings,
            }

            results = {"values": [values]}
            result = json.dumps(results, ensure_ascii=False, cls=DateTimeEncoder)

            end_time = time.time()
            elapsed_time = end_time - start_time

            logging.info(
                f"[document_chunking_function] Finished document_chunking skill in {elapsed_time:.2f} seconds."
            )
            return func.HttpResponse(result, mimetype="application/json")
        else:
            error_message = "Invalid body."
            logging.error(
                f"[document_chunking_function] {error_message}", exc_info=True
            )
            return func.HttpResponse(error_message, status_code=400)
    except ValueError as e:
        error_message = f"Invalid body: {e}"
        logging.error(f"[document_chunking_function] {error_message}", exc_info=True)
        return func.HttpResponse(error_message, status_code=400)
    except jsonschema.exceptions.ValidationError as e:
        error_message = f"Invalid request: {e}"
        logging.error(f"[document_chunking_function] {error_message}", exc_info=True)
        return func.HttpResponse(error_message, status_code=400)
    except Exception as e:
        error_message = f"An unexpected error occured: {str(e)}"
        logging.error(f"[document_chunking_function] {error_message}", exc_info=True)
        return func.HttpResponse(error_message, status_code=500)
