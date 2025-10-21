class Page:
    """
    A single page from a document

    Attributes:
        page_num (int): Page number (0-indexed)
        text (str): The text of the page
    """

    def __init__(self, page_num: int, text: str):
        self.page_num = page_num
        self.text = text
