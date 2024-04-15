# Legal Aid Chatbot for Landlord and Tenant Board Ontario

## Overview
The Legal Aid Chatbot is an AI-driven platform designed to provide immediate legal assistance tailored to the Landlord and Tenant Board of Ontario. Utilizing a sophisticated blend of Natural Language Processing (NLP), Machine Learning (ML), and Retrieval-Augmented Generation (RAG) technologies, the chatbot delivers precise, accurate legal advice in real-time. The project's aim is to democratize access to legal information, enhancing the public's ability to address landlord-tenant disputes effectively.


## Features
- **Advanced Natural Language Understanding**: Interprets and processes complex legal language to grasp user inquiries fully.
- **Retrieval-Augmented Generation (RAG)**: Leverages RAG to source relevant legal information from a comprehensive vector database, ensuring responses are both accurate and contextually relevant.
- **FAISS Vector Database**: Employs Facebook AI Similarity Search (FAISS) for efficient and scalable retrieval of document embeddings, significantly enhancing the chatbot's performance.
- **Reranking Mechanism**: Integrates advanced reranking algorithms to refine document selection, prioritizing the most pertinent information for response generation.


## Project Presentation
Watch a project presentation video of the Legal Aid Chatbot: 
[![Watch the video](https://img.youtube.com/vi/Vavf0isOISY/maxresdefault.jpg)](https://youtu.be/Vavf0isOISY?si=h31R_-lJB3JI8MNB).

## Installation

To set up the project on your local machine, follow these steps:

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Libraries Installation
Install all required Python libraries using pip:

```bash
pip install -r requirements.txt
```
This command will install all necessary packages, including transformers, faiss-gpu, nltk, and others as specified in the requirements.txt file of the project.

## Usage
To execute the module, run the following command in the terminal:

```
python main.py
```

This will trigger the entire pipeline as a batch invocation, and you can also interact with it by sending requests as well.

## Documentation
For more detailed information about the project's architecture, technologies used, and methodology, please refer to the docs folder.

## Contributing
Contributions to the project are welcome! To contribute, please follow these steps:

- Fork the repository.
- Create a new branch (git checkout -b feature-branch).
- Make your changes and commit them (git commit -am 'Add some feature').
- Push to the branch (git push origin feature-branch).
- Create a new Pull Request.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Authors
- [Debanjan Saha](https://github.com/debanjansaha-git) - Northeastern University
- [Dr. Uzair Ahmad](https://github.com/DrUzair) - PI, Khoury College of Computer Science, Northeastern University

## Acknowledgments
Thanks to the contributors who have invested their time in improving this project.
Special thanks to Atharva Pandkar & Tarun Reddy for assisting in data collection and providing the necessary legal databases and resources.