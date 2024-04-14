import json
import re


class DataCleaner:
    def __init__(self):
        self.html_pattern = re.compile(r"\{[^}]*\}")
        self.function_pattern = re.compile(r"function .*?\(\) \{.*?\}", flags=re.DOTALL)

    def clean_text(self, text):
        text = re.sub(self.html_pattern, "", text)
        text = re.sub(self.function_pattern, "", text)
        text = text.encode("ascii", "ignore").decode()
        return text


class DataLoader:
    def __init__(self):
        pass

    def load_data(self, filepath):
        with open(filepath, "r") as file:
            data = json.load(file)
        return data
