# PDF_extractor

# Chat with Your PDF

This project allows you to upload a PDF, extract its content, summarize it using GPT-2, store text embeddings in a Qdrant vector database, and interact with the content via a chat interface. The chat model uses the uploaded PDF's content as a context for generating responses.

---

## **Setup Instructions**

### 1. **Install Dependencies**
Make sure you have the required libraries installed. Use the following command to install them:
```bash
pip install torch transformers PyPDF2 sentence-transformers qdrant-client streamlit
```

---

### 2. **Qdrant Setup**
1. **Qdrant Account**: Ensure you have a Qdrant cloud account. 
2. **API Key**: Replace the `api_key` in the code with your Qdrant API key.
3. **URL**: Use the appropriate Qdrant URL for your instance.

---

### 3. **Run the Application**
Run the Streamlit app using:
```bash
streamlit run app.py
```
Here, `app.py` is the file where the provided code is saved.

---

## **How to Use**

### **Step 1: Upload a PDF**
- Use the file uploader in the app to upload a PDF file.
- The app will extract and summarize the text from the PDF.

### **Step 2: Index the Content**
- The summarized text is converted into embeddings using the SentenceTransformer model and stored in a Qdrant collection.
- Once indexed, you will see a success message indicating that the PDF is ready for interaction.

### **Step 3: Start Chatting**
- Enter a question or message in the text input box.
- The app will use the stored context from the PDF to generate a response.

---

## **Code Workflow**

1. **Extract Text**:
   - The `extract_text_from_pdf()` function reads and extracts text from the uploaded PDF file.

2. **Summarize Content**:
   - The `summarize_text()` function generates a concise summary using GPT-2.

3. **Embed and Store**:
   - The `embed_text()` function creates embeddings using the SentenceTransformer model.
   - These embeddings are stored in a Qdrant collection via the `store_embeddings_in_qdrant()` function.

4. **Chat Interface**:
   - The chat interface uses Qdrant to retrieve relevant context and GPT-2 to generate responses.

---

## **Key Features**

1. **PDF Text Extraction**:
   - Extracts content from uploaded PDF files.

2. **Summarization**:
   - Condenses long PDF content into shorter summaries.

3. **Vector Database Integration**:
   - Stores embeddings in a Qdrant collection for efficient retrieval.

4. **Interactive Chat**:
   - Enables conversations based on the uploaded PDF's content.

---

## **Troubleshooting**

1. **Missing Dependencies**:
   - Ensure all required libraries are installed using the `pip install` command.

2. **Qdrant Connection Issues**:
   - Verify the API key and URL in the code.
   - Ensure your Qdrant instance is running.

3. **PDF Processing Errors**:
   - Ensure the uploaded file is a valid PDF.
   - If a page has no text, a warning is shown but the process continues.

---

## **Customizations**

1. **Change the Summarization Model**:
   - Replace `"gpt2"` with another supported text generation model from Hugging Face.

2. **Modify Qdrant Collection Name**:
   - Update the `collection_name` parameter in `store_embeddings_in_qdrant()`.

3. **Adjust Token Limits**:
   - Modify the `truncate_input()` function to change token limits.

---

## **Future Improvements**
- Support for multi-page retrieval.
- Fine-tuned models for better summarization and chat responses.
- Additional visualization for stored embeddings.

---

### **Contact**
For any issues or feedback, please contact the developer. Happy chatting!
