Project Name: Document-based Question Answering System

Description:
A Python-based system that answers questions from a Word document (.docx) using:

FAISS for efficient similarity search

Sentence Transformers for text embeddings

OpenAI GPT-3.5 Turbo for answer generation

NLTK for text preprocessing (tokenization, stemming, lemmatization)

Key Features:

Loads and preprocesses .docx files.

Splits text into chunks for efficient retrieval.

Uses FAISS to find the most relevant text passages for a given question.

Generates accurate answers using GPT-3.5 Turbo with retrieved context.

Use Cases:

Quickly extract answers from long documents.

Build a knowledge base assistant.

Educational or research tool for document analysis.

Dependencies:

openai, python-docx, sentence-transformers, faiss-cpu, nltk

How to Run:

Install dependencies:

bash
Copy
pip install openai python-docx sentence-transformers faiss-cpu nltk  
Add your OpenAI API key and Word document path.

Run python main.py and ask questions interactively.

Example Workflow:

plaintext
Copy
Ask a question (or type 'exit' to quit): What is the main topic of the document?  
Answer: The document discusses... [generated answer]  
