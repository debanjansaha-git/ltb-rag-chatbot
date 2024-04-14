import hashlib
import pandas as pd
import json


def save_json(data, path):
    with open(path, "w") as file:
        json.dump(data, file)


def load_json(path):
    with open(path, "r") as file:
        json.loads(file)


def encode_context_columns(context_aggregated_text):
    """
    Encodes context columns into one-hot vectors based on aggregated text data.

    Args:
    - context_aggregated_text (dict): Aggregated text data containing contexts and corresponding text.

    Returns:
    - tuple: A tuple containing the encoded DataFrame and a hashmap mapping hash values to original context strings.
    """

    # @title Hashmap context to one hot vectors
    rows = []
    context_columns = set()
    hash_to_context = {}  # Map hash values to original context strings

    for key, contexts in context_aggregated_text.items():
        for context, text in contexts.items():
            context_hash = hashlib.md5(context.encode()).hexdigest()[:8]
            context_columns.add(context_hash)

            # Map the hash back to its original context
            hash_to_context[context_hash] = context

            row = {"text": text, context_hash: 1}
            rows.append(row)

    df = pd.DataFrame(rows)
    for column in context_columns:
        if column not in df.columns:
            df[column] = 0

    # @title Clean Columns
    context_to_column = {
        hashvalue: f"c{i+1}" for i, hashvalue in enumerate(sorted(context_columns))
    }
    df.rename(columns=context_to_column, inplace=True)
    columns_order = ["text"] + sorted(
        context_to_column.values(), key=lambda x: int(x[1:])
    )
    df = df.loc[:, columns_order]
    df.fillna(0, inplace=True)

    return (df, hash_to_context)
