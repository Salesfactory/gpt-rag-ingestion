import datetime
from json import JSONEncoder


class DateTimeEncoder(JSONEncoder):
    """Custom JSON encoder for datetime objects."""

    def default(self, obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        return super().default(obj)


def get_document_chunking_request_schema():
    """
    Returns JSON schema for document chunking custom skill requests.

    Expected format from Azure AI Search custom skill:
    {
        "values": [
            {
                "recordId": "string",
                "data": {
                    "documentUrl": "string",
                    "documentSasToken": "string (optional)",
                    "documentContentType": "string",
                    "includeVectors": "boolean (optional)"
                }
            }
        ]
    }
    """
    return {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "type": "object",
        "properties": {
            "values": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "recordId": {"type": "string"},
                        "data": {
                            "type": "object",
                            "properties": {
                                "documentUrl": {"type": "string", "minLength": 1},
                                "documentSasToken": {"type": "string", "minLength": 0},
                                "documentContentType": {
                                    "type": "string",
                                    "minLength": 1,
                                },
                                "includeVectors": {"type": "boolean"},
                            },
                            "required": ["documentUrl"],
                        },
                    },
                    "required": ["recordId", "data"],
                },
            }
        },
        "required": ["values"],
    }
