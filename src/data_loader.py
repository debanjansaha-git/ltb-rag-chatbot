import json
import re


class DataCleaner:
    """
    A class for cleaning text data by removing HTML tags, function patterns, and non-ASCII characters.

    Methods:
    - clean_text(text): Cleans the input text by removing HTML tags, function patterns, and non-ASCII characters.

    Args:
    - text (str): The text data to be cleaned.

    Returns:
    - str: The cleaned text data without HTML tags, function patterns, and non-ASCII characters.
    """

    def __init__(self):
        self.html_pattern = re.compile(r"\{[^}]*\}")
        self.function_pattern = re.compile(r"function .*?\(\) \{.*?\}", flags=re.DOTALL)

    def clean_text(self, text):
        text = re.sub(self.html_pattern, "", text)
        text = re.sub(self.function_pattern, "", text)
        text = text.encode("ascii", "ignore").decode()
        return text


class DataLoader:
    """
    A class for loading data from a JSON file.

    Methods:
    - load_data(filepath): Loads data from the specified JSON file.

    Args:
    - filepath (str): The path to the JSON file to load.

    Returns:
    - dict: The data loaded from the JSON file.
    """

    def __init__(self):
        pass

    def load_data(self, filepath):
        with open(filepath, "r") as file:
            data = json.load(file)
        return data
