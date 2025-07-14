
from function_app import get_request_schema

def test_get_request_schema():
    schema = get_request_schema()
    
    # Check if the schema is a dictionary
    assert isinstance(schema, dict)
    
    # Check if the schema has the required properties
    assert "$schema" in schema
    assert "type" in schema
    assert "properties" in schema
    assert "required" in schema
    
    # Check if the properties contain 'values'
    assert "values" in schema["properties"]
    
    # Check if 'values' is an array with items of type object
    values = schema["properties"]["values"]
    assert values["type"] == "array"
    assert "items" in values
    assert values["items"]["type"] == "object"

