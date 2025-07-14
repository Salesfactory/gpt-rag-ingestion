from utils.file_utils import get_filename, get_file_extension

def test_get_filename():

    file_path = "documents/folder/subfolder/test_document.txt"
    filename = get_filename(file_path)
    assert filename == "folder/subfolder/test_document.txt"
    

def test_get_file_extension():
    file_path = "documents/folder/subfolder/test_document.txt"
    extension = get_file_extension(file_path)
    assert extension == "txt"

    file_path = "chunked_doc.pdf"
    extension = get_file_extension(file_path)
    assert extension == "pdf"
