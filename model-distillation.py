from openai import OpenAI
from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import string

# Download NLTK resources (run once)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

client = OpenAI(api_key="your api key here")


def preprocess_text(text):
    # Tokenization: Split into sentences and words
    sentences = sent_tokenize(text)

    # Initialize stemmer, lemmatizer, and stopwords
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)

    processed_sentences = []
    for sentence in sentences:
        # Tokenize words
        words = word_tokenize(sentence.lower())

        # Remove stopwords and punctuation
        words = [word for word in words if word not in stop_words and word not in punctuation]

        # Apply stemming and lemmatization
        words = [lemmatizer.lemmatize(stemmer.stem(word)) for word in words]

        processed_sentences.append(" ".join(words))

    return " ".join(processed_sentences)


def load_docx_text(path):
    doc = Document(path)
    raw_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    return preprocess_text(raw_text)  # Apply preprocessing


def split_into_chunks(text, max_words=200):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def initialize_faiss(text_file_path):
    # Load embedding model
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # Load, preprocess, and chunk the document
    text = load_docx_text(text_file_path)  # Now includes preprocessing
    chunks = split_into_chunks(text)

    # Encode the chunks
    embeddings = embedder.encode(chunks, convert_to_numpy=True)

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return embedder, index, chunks

# Function to retrieve top-k relevant chunks for the given question
def get_top_k_chunks(question, embedder, index, chunks, k=10):
    # Convert the question to an embedding
    question_embedding = embedder.encode([question], convert_to_numpy=True)

    # Use FAISS to find the top-k closest chunks based on the question embedding
    distances, indices = index.search(question_embedding, k)

    return [chunks[i] for i in indices[0]]

def answer_with_gpt(question, context):
    # Create the chat-based prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant answering questions based on provided context."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
    ]

    # Call GPT-3.5 Turbo using the new API
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.0,
        max_tokens=300
    )

    answer = response.choices[0].message.content.strip()
    return answer

# Main function to interact with the user
def main():
    # File path for your Word document
    text_file_path = "final_document_2.docx"

    # Initialize FAISS and chunk the document
    embedder, index, chunks = initialize_faiss(text_file_path)

    while True:
        question = input("\nAsk a question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            print("Exiting the system. Goodbye!")
            break

        # Retrieve the top-k relevant chunks
        top_chunks = get_top_k_chunks(question, embedder, index, chunks)

        # Use GPT-3.5 Turbo to generate an answer from the context
        answer = answer_with_gpt(question, "\n".join(top_chunks))
        print("Answer:", answer)


# Run the main function
if __name__ == "__main__":
    main()
