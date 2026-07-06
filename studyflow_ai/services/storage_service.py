"""
storage_service.py
------------------
Storage Service wrapper.
Manages raw document uploads (S3/Cloud Storage) and document parser integration.
"""
import asyncio
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class StorageService:
    def __init__(self):
        logger.info("StorageService initialized.")

    def _validate_uri(self, uri: str) -> None:
        """Validates that a string is a properly formatted URI or path."""
        if not uri:
            raise ValueError("URI cannot be empty")
            
        parsed = urlparse(uri)
        # We allow local paths (no scheme) or valid cloud schemes
        if parsed.scheme and parsed.scheme not in ["gs", "s3", "http", "https", "file"]:
            raise ValueError(f"Unsupported URI scheme: {parsed.scheme}")

    async def upload_file(self, file_path: str, destination_bucket: str) -> str:
        """
        Uploads files to Cloud Storage.
        """
        logger.debug("Uploading file_path=%s to bucket=%s", file_path, destination_bucket)
        
        if not file_path or not destination_bucket:
            raise ValueError("file_path and destination_bucket cannot be empty")

        try:
            # TODO: Interface with google-cloud-storage library.
            await asyncio.sleep(0.1)
            filename = file_path.split('/')[-1] or "unnamed_file"
            destination_uri = f"gs://{destination_bucket}/{filename}"
            logger.info("Successfully uploaded file to %s", destination_uri)
            return destination_uri
        except Exception as e:
            logger.exception("File upload failed for path=%s: %s", file_path, e)
            raise RuntimeError(f"Storage upload failed: {e}")

    async def parse_document(self, file_uri: str) -> dict:
        """
        Decodes PDF text segments.
        """
        logger.debug("Parsing document at uri=%s", file_uri)
        
        self._validate_uri(file_uri)

        try:
            # TODO: Implement PyPDF2/Apache Tika extractor pipelines.
            await asyncio.sleep(0.1)
            logger.info("Successfully parsed document from %s", file_uri)
            return {
                "text_content": "Extracted course syllabus description text. Contains structured module lists and deadlines.", 
                "page_count": 5,
                "metadata": {
                    "source_uri": file_uri,
                    "document_type": "PDF"
                }
            }
        except Exception as e:
            logger.exception("Document parsing failed for uri=%s: %s", file_uri, e)
            raise RuntimeError(f"Document parsing failed: {e}")
