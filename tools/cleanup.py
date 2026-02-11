from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import unquote, urlsplit, urlunsplit

from azure.core.exceptions import AzureError
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.storage.blob import BlobServiceClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CleanupTarget:
    """Represents a cleanup target pair: one index backed by one blob container."""

    index_name: str
    container_name: str


def _normalize_url(url: str) -> str:
    """
    Normalize URLs for reliable comparison between blob URLs and indexed URLs.
    """
    parsed = urlsplit(url)
    normalized_path = unquote(parsed.path or "")
    if normalized_path.endswith("/") and normalized_path != "/":
        normalized_path = normalized_path.rstrip("/")
    return urlunsplit(
        (parsed.scheme.lower(), parsed.netloc.lower(), normalized_path, "", "")
    )


def load_cleanup_targets_from_env() -> List[CleanupTarget]:
    """
    Loads cleanup targets from environment variables.

    Required:
    CLEANUP_INDEX_CONTAINER_MAP as JSON list:
    [{"index":"idx-a","container":"cont-a"}, ...]
    """
    raw_map = os.getenv("CLEANUP_INDEX_CONTAINER_MAP", "").strip()
    if not raw_map:
        raise ValueError(
            "Missing required environment variable: CLEANUP_INDEX_CONTAINER_MAP"
        )

    try:
        parsed = json.loads(raw_map)
    except json.JSONDecodeError as exc:
        raise ValueError("CLEANUP_INDEX_CONTAINER_MAP must be valid JSON.") from exc

    if not isinstance(parsed, list) or not parsed:
        raise ValueError("CLEANUP_INDEX_CONTAINER_MAP must be a non-empty JSON array.")

    targets: List[CleanupTarget] = []
    for idx, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError(f"CLEANUP_INDEX_CONTAINER_MAP[{idx}] must be an object.")
        index_name = str(item.get("index", "")).strip()
        container_name = str(item.get("container", "")).strip()
        if not index_name or not container_name:
            raise ValueError(
                f"CLEANUP_INDEX_CONTAINER_MAP[{idx}] requires non-empty "
                "'index' and 'container'."
            )
        targets.append(
            CleanupTarget(index_name=index_name, container_name=container_name)
        )
    return targets


def load_cleanup_runtime_settings() -> Dict[str, Any]:
    """
    Validates and loads shared cleanup runtime settings from env.
    """
    search_service = os.getenv("AZURE_SEARCH_SERVICE", "").strip()
    storage_account = os.getenv("AZURE_STORAGE_ACCOUNT", "").strip()

    if not search_service:
        raise ValueError("Missing required environment variable: AZURE_SEARCH_SERVICE")
    if not storage_account:
        raise ValueError("Missing required environment variable: AZURE_STORAGE_ACCOUNT")

    dry_run = os.getenv("CLEANUP_INDEX_DRY_RUN", "true")

    raw_batch_size = os.getenv("CLEANUP_INDEX_BATCH_SIZE", "1000").strip()

    delete_batch_size = int(raw_batch_size)

    if delete_batch_size < 1 or delete_batch_size > 1000:
        raise ValueError("CLEANUP_INDEX_BATCH_SIZE must be between 1 and 1000.")

    targets = load_cleanup_targets_from_env()

    return {
        "search_service": search_service,
        "storage_account": storage_account,
        "dry_run": dry_run,
        "delete_batch_size": delete_batch_size,
        "targets": targets,
    }


class CleanupCoordinator:
    """
    Coordinates orphaned-document cleanup across one or more index/container targets.
    """

    def __init__(
        self,
        search_service: str,
        storage_account: str,
        credential: DefaultAzureCredential,
    ):
        self.search_service = search_service
        self.storage_account = storage_account
        self.credential = credential
        self.search_endpoint = f"https://{self.search_service}.search.windows.net"
        self.storage_account_url = (
            f"https://{self.storage_account}.blob.core.windows.net"
        )

        self.blob_service_client = BlobServiceClient(
            account_url=self.storage_account_url, credential=self.credential
        )
        self._search_clients: Dict[str, SearchClient] = {}

    def _get_search_client(self, index_name: str) -> SearchClient:
        if index_name not in self._search_clients:
            self._search_clients[index_name] = SearchClient(
                endpoint=self.search_endpoint,
                index_name=index_name,
                credential=self.credential,
            )
        return self._search_clients[index_name]

    def _list_active_blob_urls(self, container_name: str) -> set[str]:
        logger.info(
            "[cleanup][list_active_blob_urls] scanning container=%s", container_name
        )
        container_client = self.blob_service_client.get_container_client(container_name)
        active_urls: set[str] = set()
        blob_count = 0

        for blob in container_client.list_blobs():
            raw_url = f"{container_client.url}/{blob.name}"
            active_urls.add(_normalize_url(raw_url))
            blob_count += 1

        logger.info(
            "[cleanup][list_active_blob_urls] container=%s active_blobs=%s",
            container_name,
            blob_count,
        )
        return active_urls

    def _scan_index_documents(self, index_name: str) -> Dict[str, List[str]]:
        logger.info("[cleanup][scan_index_documents] scanning index=%s", index_name)
        search_client = self._get_search_client(index_name)
        url_to_ids: Dict[str, List[str]] = defaultdict(list)
        scanned_docs = 0

        results = search_client.search(search_text="*", select=["id", "url"])
        for row in results:
            doc_id = row.get("id")
            url = row.get("url")
            if not doc_id or not url:
                continue
            url_to_ids[_normalize_url(url)].append(doc_id)
            scanned_docs += 1

        logger.info(
            "[cleanup][scan_index_documents] index=%s scanned_docs=%s unique_urls=%s",
            index_name,
            scanned_docs,
            len(url_to_ids),
        )
        return url_to_ids

    def _delete_chunk_ids(
        self, index_name: str, chunk_ids: List[str], dry_run: bool, batch_size: int
    ) -> tuple[int, int]:
        if not chunk_ids:
            return 0, 0

        if dry_run:
            logger.info(
                "[cleanup][delete_chunk_ids] index=%s dry_run=True would_delete=%s",
                index_name,
                len(chunk_ids),
            )
            return len(chunk_ids), 0

        search_client = self._get_search_client(index_name)
        total_success = 0
        total_failed = 0

        for start in range(0, len(chunk_ids), batch_size):
            batch = chunk_ids[start : start + batch_size]
            actions = [{"id": chunk_id} for chunk_id in batch]
            result = search_client.delete_documents(documents=actions)
            batch_success = sum(1 for item in result if item.succeeded)
            batch_failed = len(batch) - batch_success
            total_success += batch_success
            total_failed += batch_failed

            logger.info(
                "[cleanup][delete_chunk_ids] index=%s batch=%s deleted=%s failed=%s",
                index_name,
                (start // batch_size) + 1,
                batch_success,
                batch_failed,
            )

        return total_success, total_failed

    def cleanup_target(
        self, target: CleanupTarget, dry_run: bool, delete_batch_size: int
    ) -> Dict[str, Any]:
        active_urls = self._list_active_blob_urls(target.container_name)
        indexed_url_to_ids = self._scan_index_documents(target.index_name)

        orphaned_urls = sorted(set(indexed_url_to_ids.keys()) - active_urls)
        orphaned_chunk_ids = [
            chunk_id
            for orphaned_url in orphaned_urls
            for chunk_id in indexed_url_to_ids[orphaned_url]
        ]

        deleted_chunks, failed_deletes = self._delete_chunk_ids(
            target.index_name,
            orphaned_chunk_ids,
            dry_run=dry_run,
            batch_size=delete_batch_size,
        )

        return {
            "status": "success",
            "index": target.index_name,
            "container": target.container_name,
            "orphaned_documents": len(orphaned_urls),
            "orphaned_chunks": len(orphaned_chunk_ids),
            "deleted_chunks": deleted_chunks,
            "failed_deletes": failed_deletes,
            "dry_run": dry_run,
        }

    def cleanup_all(
        self,
        targets: Iterable[CleanupTarget],
        dry_run: bool,
        delete_batch_size: int,
    ) -> Dict[str, Any]:
        target_results: List[Dict[str, Any]] = []
        failed_targets = 0

        for target in targets:
            logger.info(
                "[cleanup][cleanup_all] start index=%s container=%s dry_run=%s",
                target.index_name,
                target.container_name,
                dry_run,
            )
            try:
                target_result = self.cleanup_target(
                    target=target,
                    dry_run=dry_run,
                    delete_batch_size=delete_batch_size,
                )
                target_results.append(target_result)
            except Exception as exc:
                failed_targets += 1
                logger.error(
                    "[cleanup][cleanup_all] failed index=%s container=%s error=%s",
                    target.index_name,
                    target.container_name,
                    exc,
                    exc_info=True,
                )
                target_results.append(
                    {
                        "status": "error",
                        "index": target.index_name,
                        "container": target.container_name,
                        "error": str(exc),
                        "orphaned_documents": 0,
                        "orphaned_chunks": 0,
                        "deleted_chunks": 0,
                        "failed_deletes": 0,
                        "dry_run": dry_run,
                    }
                )

        totals = {
            "orphaned_documents": sum(
                item.get("orphaned_documents", 0) for item in target_results
            ),
            "orphaned_chunks": sum(
                item.get("orphaned_chunks", 0) for item in target_results
            ),
            "deleted_chunks": sum(
                item.get("deleted_chunks", 0) for item in target_results
            ),
            "failed_deletes": sum(
                item.get("failed_deletes", 0) for item in target_results
            ),
        }

        if failed_targets == 0:
            status = "success"
        elif failed_targets < len(target_results):
            status = "partial_success"
        else:
            status = "error"

        return {
            "status": status,
            "dry_run": dry_run,
            "targets_processed": len(target_results),
            "failed_targets": failed_targets,
            "totals": totals,
            "results": target_results,
        }

    def close(self) -> None:
        for client in self._search_clients.values():
            close_client = getattr(client, "close", None)
            if callable(close_client):
                close_client()
        self._search_clients.clear()

        close_credential = getattr(self.credential, "close", None)
        if callable(close_credential):
            close_credential()


def run_search_index_cleanup(dry_run_override: Optional[bool] = None) -> Dict[str, Any]:
    """
    Entry helper used by function triggers.
    """
    settings = load_cleanup_runtime_settings()
    dry_run = settings["dry_run"] if dry_run_override is None else dry_run_override

    credential = DefaultAzureCredential(exclude_interactive_browser_credential=True)
    coordinator = CleanupCoordinator(
        search_service=settings["search_service"],
        storage_account=settings["storage_account"],
        credential=credential,
    )

    try:
        result = coordinator.cleanup_all(
            targets=settings["targets"],
            dry_run=dry_run,
            delete_batch_size=settings["delete_batch_size"],
        )
        return result
    except AzureError as exc:
        logger.error(
            "[cleanup][run_search_index_cleanup] Azure error: %s", exc, exc_info=True
        )
        raise
    finally:
        coordinator.close()
