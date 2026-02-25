import asyncio
import json
import logging
import time
from io import StringIO
from typing import Any, Dict, List
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def records_to_string(data: Dict[str, Dict[str, float]]) -> str:
    """Convert grouped data to a formatted JSON string."""
    return json.dumps(data, indent=2, ensure_ascii=False)


async def summarize_grouped_data(
    client: AsyncOpenAI,
    section: str,
    parent_category: str,
    column: str,
    data: Dict[str, Dict[str, float]],
    model: str = "gpt-4.1-nano",
) -> str:
    """Generate prose from column-grouped data."""
    data_str = records_to_string(data)

    # Clean up column name for display (by_generation -> Generation)
    column_display = column.replace("by_", "").replace("_", " ").title()

    prompt = f"""Convert the following JSON data into flowing, readable prose.

CONTEXT:
- Section: {section}
- Topic: {parent_category}
- Breakdown by: {column_display}

CRITICAL CONTEXT REQUIREMENT (for RAG/retrieval chunking):
- This text will be chunked and indexed for retrieval. Each sentence may be retrieved independently.
- EVERY sentence MUST include the topic ("{parent_category}") AND the breakdown type ("{column_display}") so any sentence can stand alone.
- Do NOT rely on previous sentences for context.

WRITING STYLE:
- Write in smooth, connected paragraphs, but keep it clear and concise.
- Always write % symbol, never spell out "percent"
- CRITICAL NUMBER FORMATTING:
  * Decimal values between 0 and 1 (like 0.25, 0.3, 0.87) are percentages - multiply by 100 and add % symbol (25%, 30%, 87%)
  * Whole numbers or values greater than 1 (like 5, 42, 150) are counts/integers - report as-is without % symbol
- Repeat context in EVERY sentence using natural phrasing:
  * "For {parent_category} broken down by {column_display}, Gen Z shows..."
  * "Looking at {column_display} data for {parent_category}, Millennials..."
  * "In the {parent_category} analysis by {column_display}..."

STRICT CONTENT RULES:
- Report ONLY the exact values present in the JSON
- DO NOT interpret, analyze, or draw conclusions, be concise.
- DO NOT use evaluative words (strong, weak, notable, significant)
- DO NOT identify patterns or trends beyond stating the numbers

Data:
<data>
{data_str}
</data>

Write prose where EVERY sentence explicitly mentions both the topic and breakdown type for retrieval context."""

    response = await client.responses.create(
        model=model,
        instructions="You are a prose writer who converts data into smooth, readable paragraphs. Present all data values accurately. Never analyze or use evaluative language. Just state the facts in flowing prose.",
        input=prompt,
        temperature=0.2,
        max_output_tokens=5000,
    )

    return response.output_text


async def process_grouped_record_in_memory(
    client: AsyncOpenAI,
    record: Dict[str, Any],
    output: Any,
    lock: asyncio.Lock,
    model: str = "gpt-4.1-nano",
    filename: str = "",
) -> None:
    """Process a single grouped record and append result to StringIO buffer."""
    section = record["section"]
    parent_category = record["parent_category"]
    column = record["column"]
    data = record["data"]

    column_display = column.replace("by_", "").replace("_", " ").title()
    logger.info(f"[{filename}] Processing: {parent_category} - {column_display}...")

    try:
        summary = await summarize_grouped_data(
            client, section, parent_category, column, data, model
        )

        async with lock:
            output.write(f"\n## {parent_category} - {column_display}\n\n")
            output.write(summary)
            output.write("\n\n---\n")

        logger.info(f"[{filename}] Completed: {parent_category} - {column_display}")
    except Exception as e:
        logger.error(
            f"[{filename}] Error processing {parent_category} - {column_display}: {e}",
            exc_info=True,
        )


async def process_grouped_record(
    client: AsyncOpenAI,
    record: Dict[str, Any],
    output_file: str,
    lock: asyncio.Lock,
    model: str = "gpt-4.1-nano",
) -> None:
    """Process a single grouped record and append result to file."""
    section = record["section"]
    parent_category = record["parent_category"]
    column = record["column"]
    data = record["data"]

    column_display = column.replace("by_", "").replace("_", " ").title()
    logger.info(f"Processing: {parent_category} - {column_display}...")

    try:
        summary = await summarize_grouped_data(
            client, section, parent_category, column, data, model
        )

        async with lock:
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(f"\n## {parent_category} - {column_display}\n\n")
                f.write(summary)
                f.write("\n\n---\n")

        logger.info(f"Completed: {parent_category} - {column_display}")
    except Exception as e:
        logger.error(
            f"Error processing {parent_category} - {column_display}: {e}", exc_info=True
        )


async def process_json_to_markdown_in_memory(
    grouped_records: List[Dict[str, Any]],
    filename: str,
    model: str = "gpt-4.1-nano",
    max_concurrent: int = 5,
) -> str:
    """Process grouped JSON data in memory and return markdown string.

    Args:
        grouped_records: Pre-grouped JSON from excel_serializer.group_by_column()
        filename: Source filename for logging purposes
        model: OpenAI model to use
        max_concurrent: Maximum concurrent API calls

    Returns:
        Markdown string with all processed content
    """
    start_time = time.time()
    logger.info(f"[{filename}] Processing {len(grouped_records)} grouped records")

    # Initialize markdown content with header
    section = (
        grouped_records[0].get("section", "Unknown") if grouped_records else "Unknown"
    )
    output = StringIO()
    output.write(f"# {section} Analysis\n\n")

    client = AsyncOpenAI()
    lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_process(record: Dict[str, Any]):
        async with semaphore:
            await process_grouped_record_in_memory(
                client, record, output, lock, model, filename
            )

    tasks = [limited_process(record) for record in grouped_records]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Check if all tasks failed
    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        logger.warning(f"[{filename}] {len(errors)} tasks failed during processing")
        for i, error in enumerate(errors[:5]):  # Log first 5 errors
            logger.error(f"[{filename}] Task error {i+1}: {error}", exc_info=error)

    elapsed_time = time.time() - start_time
    result = output.getvalue()
    output.close()

    logger.info(f"[{filename}] Processing complete in {elapsed_time:.2f} seconds")

    # Warn if only header was generated (indicates all processing failed)
    if len(result.strip()) <= len(f"# {section} Analysis") + 10:
        logger.error(
            f"[{filename}] WARNING: Output contains only header! "
            f"All {len(grouped_records)} records may have failed processing."
        )

    return result


async def process_json_to_markdown(
    input_json_path: str,
    output_md_path: str,
    model: str = "gpt-4.1-nano",
    max_concurrent: int = 5,
) -> None:
    """Main processing function. Expects pre-grouped JSON from excel_serializer.group_by_column()."""
    start_time = time.time()

    with open(input_json_path, "r", encoding="utf-8") as f:
        grouped_records = json.load(f)

    logger.info(f"Loaded {len(grouped_records)} grouped records")

    # Initialize output file with header
    section = (
        grouped_records[0].get("section", "Unknown") if grouped_records else "Unknown"
    )
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(f"# {section} Analysis\n\n")

    client = AsyncOpenAI()
    lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_process(record: Dict[str, Any]):
        async with semaphore:
            await process_grouped_record(client, record, output_md_path, lock, model)

    tasks = [limited_process(record) for record in grouped_records]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Check if all tasks failed
    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        logger.warning(f"{len(errors)} tasks failed during processing")
        for i, error in enumerate(errors[:5]):
            logger.error(f"Task error {i+1}: {error}", exc_info=error)

    elapsed_time = time.time() - start_time
    logger.info(f"\nDone! Output written to: {output_md_path}")
    logger.info(f"Total time: {elapsed_time:.2f} seconds")


def main():
    """Entry point for command line usage."""
    asyncio.run(
        process_json_to_markdown(
            "subsection.json",  # Pre-grouped JSON from excel_serializer.group_by_column()
            "water_consideration.md",
            "gpt-4.1-mini",
            20,
        )
    )


if __name__ == "__main__":
    main()
