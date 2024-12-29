import os
from PyPDF2 import PdfReader

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

def save_text_to_file(text, output_file_path):
    """
    Save extracted text to a text file.
    
    Args:
        text (str): The extracted text to save.
        output_file_path (str): Path to the output text file.
    """
    try:
        with open(output_file_path, 'w', encoding='utf-8') as file:
            file.write(text)
        print(f"Text successfully saved to {output_file_path}")
    except Exception as e:
        raise RuntimeError(f"Error occurred while saving text to file: {e}")

def test_pdf_extraction(input_pdf_path, output_text_path):
    """
    Test function to extract text from a PDF and save it to a text file.
    
    Args:
        input_pdf_path (str): Path to the input PDF file.
        output_text_path (str): Path to the output text file.
    """
    try:
        print(f"Starting text extraction for: {input_pdf_path}")
        extracted_text = extract_text_from_pdf(input_pdf_path)
        save_text_to_file(extracted_text, output_text_path)
        print("Test completed successfully.")
    except Exception as e:
        print(f"Test failed: {e}")

# Example Usage
if __name__ == "__main__":
    input_pdf = "test_files\\tamil_resume.pdf"  # Replace with your PDF file path
    output_text = "extracted_text.txt"  # Replace with your desired text file path
    test_pdf_extraction(input_pdf, output_text)
