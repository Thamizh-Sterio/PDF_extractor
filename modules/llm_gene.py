import os
from PyPDF2 import PdfReader
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
import torch


# Function to extract text from a PDF file
def extract_text_from_pdf(file_path):
    """
    Extract text from a PDF file.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.lower().endswith(".pdf"):
        raise ValueError("The provided file is not a PDF.")

    try:
        reader = PdfReader(file_path)
        text = ""
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            else:
                print(f"Warning: Page {page_num} has no extractable text.")
        return text
    except Exception as e:
        raise RuntimeError(f"Error occurred while processing the PDF: {e}")


# Function to summarize text using GPT-2
def summarize_text(text, max_new_tokens=50, min_length=25):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    summarizer = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
    )
    # Truncate text to a maximum of 1024 tokens for GPT-2
    input_text = text[:1024] if len(text) > 1024 else text
    # Generate summary
    summary = summarizer(input_text, max_new_tokens=max_new_tokens, min_length=min_length, num_return_sequences=1)
    return summary[0]["generated_text"]


# Function to embed text using SentenceTransformers
def embed_text(text):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    text_segments = text.split("\n")
    embeddings = model.encode(text_segments)
    return embeddings


# Function to store embeddings in Qdrant
def store_embeddings_in_qdrant(embeddings, metadata, collection_name="text_embeddings"):
    client = QdrantClient(host="localhost", port=6333)
    vector_dim = len(embeddings[0])  # Dimension of the embeddings

    # Check and create the Qdrant collection if it doesn't exist
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_dim, distance="Cosine"),
        )

    # Prepare points for insertion
    points = [
        PointStruct(
            id=i,
            vector=embedding.tolist(),
            payload={"metadata": metadata},
        )
        for i, embedding in enumerate(embeddings)
    ]

    client.upsert(collection_name=collection_name, points=points)
    print(f"Stored {len(points)} embeddings into Qdrant collection '{collection_name}'.")


def truncate_input(input_text, max_length=1024):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenized_input = tokenizer(input_text, truncation=True, max_length=max_length, return_tensors="pt")
    return tokenizer.decode(tokenized_input["input_ids"][0], skip_special_tokens=True)


# Function to create the chat interface using GPT-2
def create_chat_interface():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    chat_model = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
    )

    client = QdrantClient(host="localhost", port=6333)
    collection_name = "text_embeddings"

    def chat(input_message):
        # Retrieve summaries from Qdrant
        retrieved_points = client.scroll(collection_name=collection_name, limit=10)
        responses = [point.payload["metadata"]["summary"] for point in retrieved_points[0]]

        # Combine and summarize the retrieved context
        combined_context = "\n".join(responses)
        summarized_context = summarize_text(combined_context, max_new_tokens=50)

        # Truncate summarized context to fit within token limits
        truncated_context = truncate_input(summarized_context)
        prompt = f"Context:\n{truncated_context}\n\nUser: {input_message}\nAI:"

        # Generate response
        response = chat_model(prompt, max_new_tokens=50, num_return_sequences=1)
        return response[0]["generated_text"]

    return chat


# Main script
if __name__ == "__main__":
    pdf_path = "test_files\\tamil_resume.pdf"  # Replace with your PDF file path
    extracted_text = extract_text_from_pdf(pdf_path)

    # Summarize and embed the text
    summarized_text = summarize_text(extracted_text, max_new_tokens=50, min_length=25)
    embeddings = embed_text(summarized_text)

    # Store embeddings in Qdrant
    metadata = {"source": pdf_path, "summary": summarized_text}
    store_embeddings_in_qdrant(embeddings, metadata, collection_name="text_embeddings")

    # Create the chat interface
    chat_interface = create_chat_interface()

    # Start a conversation
    user_message = "What is the summary of the data in the PDF?"
    response = chat_interface(user_message)

    print("Response from model:", response)
