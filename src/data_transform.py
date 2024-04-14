import re
from utils import save_json


class DataTransformer:
    """
    A class for aggregating context data based on specific rules and types.

    Methods:
    - aggregate_context(data, outfilepath=None): Aggregates context data based on rules and types.

    Args:
    - data (dict): The input data to aggregate.
    - outfilepath (str, optional): The file path to save the aggregated data.

    Returns:
    - dict: The aggregated context data.
    """

    def __init__(self, cleaner):
        self.cleaner = cleaner

    def aggregate_context(self, data, outfilepath=None):
        aggregated = {}
        for key, value in data.items():
            if key not in aggregated:
                aggregated[key] = {}

            type_form = value.get("__Type__", "")
            for rule in value.get("__Rules__", []):
                context = rule.get("Context")
                text = rule.get("text").strip()
                if key == "0050000107510004":
                    pattern = r"^(.*?):\s*(.*)$"
                    if match := re.search(pattern, text.strip(), re.DOTALL):
                        context = match.group(1).strip()
                        text = match.group(2).strip()

                if (
                    (
                        key != "000000010002003603620001"
                        and '\ndocument.getElementById("form1").onclick' not in text
                    )
                    and (
                        key != "000000010002000300430001"
                        and '\ndocument.getElementById("step1").onclick' not in text
                    )
                    and (
                        key != "000000010002006200701184"
                        and "\n\r\n    Table 1: Appointees" not in text
                    )
                    and type_form == "link"
                ):
                    cleaned_text = self.cleaner.clean_text(text)

                    if context in aggregated[key]:
                        aggregated[key][context] += f" {cleaned_text}"
                    else:
                        aggregated[key][context] = cleaned_text

        if outfilepath is not None:
            save_json(aggregated, outfilepath)
        return aggregated
