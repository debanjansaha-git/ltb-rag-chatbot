import re
import json
import jq
from transformers import GPT2TokenizerFast
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from ltb import logger


class DataProcessor:
    def __init__(self, chunk_size=300, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.DEPTH_LIMIT = 4
        self.visited = set()  # To keep track of visited links
        self.documents = []
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    def read_data(self, json_file):
        # Read data from JSON file
        with open(json_file) as json_file:
            raw_data = json.load(json_file)
        logger.info(f"Read complete for JSON: {json_file.name}")
        data_keys = list(raw_data.keys())
        return raw_data, data_keys

    def write_json(self, output_file):
        # Write json data
        with open(output_file, "w") as file:
            for nested_dict in self.documents:
                json.dump(nested_dict, file)
                file.write("\n")
        logger.info("Pre-processed data written to disk!!!")

    def clean_text(self, text):
        # Define a regular expression pattern to match all special characters and Unicode character escape sequences
        pattern = r"[^\w\s]|\\u[0-9a-fA-F]{4}"
        cleaned_text = re.sub(pattern, "", text)
        return cleaned_text

    def extend_data(self, text, context):
        if text or context:
            self.documents.append({"text": text, "context": context})
        else:
            logger.warning("No information present")

    def process_context(self, data):
        # Extract text and context from data
        text_str, context_str = "", ""
        logger.info("Processing Rules for Key: %s", data["__ID__"])
        if "__Rules__" in data:
            for rule in data["__Rules__"]:
                rule_text = rule.get("text", "")
                rule_context = rule.get("Context", "")
                text_str += self.clean_text(rule_text) + "\n"
                context_str += self.clean_text(rule_context) + "\n"

        return text_str, context_str

    def split_text_into_blocks(self, text, chunk_size=500, chunk_overlap=75):
        """
        Split the text into blocks with the specified size and overlap.
        #"""
        # text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
        #     self.tokenizer, chunk_size=100, chunk_overlap=0
        # )
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        texts = text_splitter.split_text(text)
        return texts

    def split_documents(self, data, depth=0):
        content_type = data.get("__Type__", "")
        id = data.get("__ID__", "")
        url = data.get("__URL__", "")
        sub_links = data.get("__SUB_Link__", [])
        # logger.debug(f"Type: {content_type}, Key: {id}, URL: {url}")
        logger.info(f"Processing Key={id}")
        if url in self.visited:
            logger.warning(f"Skipping - Already Indexed: {url}.")

        else:
            if content_type == "link":
                text, context = self.process_context(data)
                print(f"Len Text: {len(text)}, Len Context: {len(context)}")
                # self.extend_data(text, context)
                self.visited.add(url)
                if len(text) > 0:
                    if len(text) < 500:
                        print("Inside smaller block")
                        # Split text into smaller units with overlap for lower depths
                        blocks = self.split_text_into_blocks(
                            text, chunk_size=100, chunk_overlap=20
                        )
                        print("Num Blocks: ", len(blocks))
                        for block in blocks:
                            self.extend_data(block, context)
                    else:
                        print("Inside larger block")
                        text_blocks = self.split_text_into_blocks(text)
                        context_blocks = self.split_text_into_blocks(context)
                        for text_block, context_block in zip(
                            text_blocks, context_blocks
                        ):
                            self.extend_data(text_block, context_block)

            elif content_type == "pdf":
                for page in data.get("data", []):
                    forms_data = data.get("form", "")
                    forms_topics = list(forms_data.keys())
                    for topic in forms_topics:
                        topic_text = " ".join(forms_data[topic])
                        forms = self.clean_text(topic_text)
                        self.extend_data(forms, topic)
            else:
                logger.warning(f"Invalid content type: {content_type}")

        # if depth == 0:
        #     self.visited.clear()  # Reset visited links at the root level

        # docs = self.text_splitter.split_documents(self.documents)

        # if depth < self.DEPTH_LIMIT:
        #     # Include related content referenced by sub-links
        #     for sub_link in sub_links:
        #         if sub_link in data:
        #             sub_data = data[sub_link]
        #             if depth < 3:
        #                 # Split text into smaller units with overlap for lower depths
        #                 blocks = self.split_text_into_blocks(
        #                     text, block_size=100, overlap=25
        #                 )
        #                 for block in blocks:
        #                     self.extend_data(block, context)
        #             else:
        #                 # At depth 3 and above, heavily fragment the text
        #                 text_blocks = self.split_text_into_blocks(text)
        #                 context_blocks = self.split_text_into_blocks(context)
        #                 for text_block, context_block in zip(
        #                     text_blocks, context_blocks
        #                 ):
        #                     self.extend_data(text_block, context_block)
        #         else:
        #             logger.warning(f"Invalid sub-link type for {sub_link}")
