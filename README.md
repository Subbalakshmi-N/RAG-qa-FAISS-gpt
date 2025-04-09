# Document-QA-System

This project implements a Document Question-Answering (QA) System. It allows users to input a question and retrieve relevant answers from a Word document using a combination of Natural Language Processing (NLP) techniques, sentence embeddings, and a vector search index (FAISS). The system processes the document, divides it into chunks, and uses the FAISS index to retrieve the most relevant chunks based on the input question. The system then uses GPT-3.5 Turbo to generate a coherent and contextually accurate response.

## Features

- **Text Preprocessing**: The document is preprocessed to remove stopwords, punctuation, and perform stemming/lemmatization.
- **Document Chunking**: Large documents are split into smaller chunks to enable efficient processing and search.
- **FAISS Integration**: The system uses the FAISS library to build an index of document embeddings, enabling fast and scalable nearest neighbor search.
- **GPT-3.5 Turbo Integration**: After retrieving the top relevant chunks from the document, GPT-3.5 Turbo generates an answer based on the context provided.

## Prerequisites

- Python 3.6+
- OpenAI API Key
- FAISS library
- NLTK library
- Sentence-Transformers library
- Python-docx library

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/Document-QA-System.git
   cd Document-QA-System

2. Install the required dependencies:

pip install -r requirements.txt

3. Make sure you have an OpenAI API key set up.

## Usage
1. Prepare a Word document (e.g., .docx) and place it in the same directory as the script.

2. Run the script:
  python model-distillation.py

3. Ask questions, and the system will provide relevant answers based on the document's content.

4. To exit the system, simply type 'exit'.

## Acknowledgements
OpenAI's GPT-3.5 Turbo for natural language processing.

Sentence-Transformers for embedding-based document search.

FAISS for efficient similarity search.

NLTK for text preprocessing tasks.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

This summary provides a clear overview of the project's functionality, setup instructions, and usage examples, making it easier for anyone to understand and use the system.



