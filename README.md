# Landlord Tenant Board ChatBot

The Landlord Tenant Board ChatBot is an AI-powered assistant designed to provide information and assistance to both landlords and tenants regarding housing regulations in Ontario, Canada. The chatbot utilizes advanced natural language processing techniques to understand user queries and retrieve relevant information from a large corpus of documents extracted from the Ontario Tribunals Landlord Tenant Board website.

## Table of Contents

- [Landlord Tenant Board ChatBot](#landlord-tenant-board-chatbot)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [License](#license)

## Overview

The Landlord Tenant Board ChatBot leverages state-of-the-art technologies including OpenAI's text embedding models and FAISS (Facebook AI Similarity Search) for document retrieval. By indexing the corpus of documents and generating embeddings for efficient similarity search, the chatbot provides fast and accurate responses to user queries, covering a wide range of topics related to landlord-tenant disputes, regulations, rights, and responsibilities.

## Features

- **Document Indexing**: The chatbot indexes a large corpus of documents extracted from the Ontario Tribunals Landlord Tenant Board website, including rules, regulations, obligations, and other relevant information.

- **Natural Language Understanding**: Using OpenAI's text embedding models, the chatbot comprehends user queries and identifies relevant documents for retrieval.

- **Document Retrieval**: Leveraging FAISS for efficient similarity search, the chatbot retrieves relevant documents based on user queries, ensuring fast and accurate responses.

- **Context Expansion**: The chatbot incorporates adjacent text and related content referenced by sub-links to provide comprehensive responses, preserving sequence information and context.

- **Context Generation**: The chatbot generates responses to user queries after processing the sub-context from the FAISS store and passing the query parameters to the corresponding LLM for generating responses.

## Installation

To install and run the Landlord Tenant Board ChatBot, follow these steps:

1. Clone the repository to your local machine:

```bash
git clone https://github.com/debanjansaha-git/ltb-rag-chatbot.git
```

2. Navigate to the project directory:

```bash
cd ltb-rag-chatbot
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Set up the necessary environment variables, including your OpenAI API key, FAISS index name, and other configuration parameters.

5. Run the chatbot:

```bash
python main.py
```

## Usage

Once the chatbot is up and running, you can interact with it by entering your queries regarding landlord-tenant disputes, regulations, rights, and responsibilities. The chatbot will provide responses based on the indexed documents and the relevance of the information to your query.

## Contributing

Contributions to the Landlord Tenant Board ChatBot project are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository on GitHub.

2. Create a new branch with a descriptive name for your feature or fix:

```bash
git checkout -b feature/new-feature
```

3. Make your changes and commit them with clear and concise commit messages.

4. Push your changes to your fork:

```bash
git push origin feature/new-feature
```

5. Create a pull request against the main branch of the original repository.

6. Your pull request will be reviewed, and once approved, it will be merged into the main branch.

## License

This project is licensed under the [MIT License](LICENSE).

