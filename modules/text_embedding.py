from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams

def read_text_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {file_path} was not found.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while reading the file: {e}")

def summarize_text(text, max_length=100, min_length=30):
    if not text.strip():
        raise ValueError("The text is empty or contains only whitespace.")

    try:
        summarizer = pipeline("summarization", model="t5-small", device=0 if torch.cuda.is_available() else -1)
        # Truncate text to a maximum of 2000 characters
        text = text[:2000] if len(text) > 2000 else text
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        raise RuntimeError(f"Error occurred during summarization: {e}")

def embed_text(text):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    try:
        if isinstance(text, str):
            text_segments = text.split("\n")
        else:
            raise ValueError("Input text must be a string.")
        
        print(f"Embedding {len(text_segments)} segments of text.")  # Debugging
        embeddings = model.encode(text_segments)
        
        print(f"Generated embeddings: {embeddings[:5]}")  # Debugging
        return embeddings
    except Exception as e:
        raise RuntimeError(f"Error occurred during embedding: {e}")

def process_text_for_embedding(file_path, summary_max_length=100, summary_min_length=25):
    print("Reading text from file...")
    extracted_text = read_text_from_file(file_path)
    print("Checking if summarization is needed...")
    summarized_text = summarize_text(extracted_text, max_length=summary_max_length, min_length=summary_min_length)
    print("Text processing complete. Proceeding to embedding...")
    embeddings = embed_text(summarized_text)
    print("Embedding complete.")
    return summarized_text, embeddings

def store_embeddings_in_qdrant(embeddings, metadata, collection_name="my_collection"):
    # Initialize the Qdrant client
    client = QdrantClient(host="localhost", port=6333)  # Update with your Qdrant server details

    # Ensure the collection exists with proper vector configuration
    vector_dim = len(embeddings[0])  # Determine vector dimensions
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_dim, distance="Cosine")  # Set vector size and distance metric
    )

    # Prepare data points
    points = [
        PointStruct(
            id=i,
            vector=embedding.tolist(),
            payload={"metadata": metadata}  # Attach metadata
        )
        for i, embedding in enumerate(embeddings)
    ]

    # Upload data to Qdrant
    client.upsert(collection_name=collection_name, points=points)
    print(f"Stored {len(points)} embeddings into Qdrant collection '{collection_name}'.")

if __name__ == "__main__":
    file_path = "extracted_text.txt"  # Replace with your text file path
    summarized_text, embeddings = process_text_for_embedding(file_path)

    # Store the embeddings in Qdrant
    metadata = {"source": file_path, "summary": summarized_text}
    store_embeddings_in_qdrant(embeddings, metadata, collection_name="text_embeddings")
