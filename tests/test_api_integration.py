from function_app import document_chunking
from unittest.mock import Mock

def test_api_schema_error():

    # Create a mock request with valid JSON body
    req = Mock()
    req.get_json.return_value = {
        "data": {
            "content": "This is a test document.",
            "documentUrl": "https://example.com/test-document.txt",
        }
}
    
    # Call the function
    response = document_chunking(req)

    # Check the response
    assert response.status_code == 400

