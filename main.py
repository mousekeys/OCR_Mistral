from mistralai import Mistral
import os
from dotenv import load_dotenv
import argparse

# Load environment variables from .env file
load_dotenv()

# --- Argument Parsing Setup ---
def validate_pdf_path(path):
    """Checks if the provided path is a valid PDF file."""
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f"Error: File not found at '{path}'")
    if not path.lower().endswith(".pdf"):
        raise argparse.ArgumentTypeError(f"Error: File '{path}' is not a PDF.")
    return path

parser = argparse.ArgumentParser(description="Process a PDF file using Mistral OCR.")
parser.add_argument(
    "--pdf_path",
    dest="pdf_path",
    type=validate_pdf_path,
    required=True,
    help="The absolute path to the PDF file you want to process.",
)

args = parser.parse_args()


api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)


pdf_to_upload_path = args.pdf_path

print(f"Uploading PDF from: {pdf_to_upload_path}")

try:
    with open(pdf_to_upload_path, "rb") as pdf_content_file:
        uploaded_pdf = client.files.upload(
            file={
                "file_name": os.path.basename(pdf_to_upload_path), # Get just the filename
                "content": pdf_content_file,
            },
            purpose="ocr"
        )
    print(f"File uploaded successfully with ID: {uploaded_pdf.id}")

    retrieved_file = client.files.retrieve(file_id=uploaded_pdf.id)
    print(f"Retrieved file details: {retrieved_file.filename}")

    signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)
    print(f"Obtained signed URL: {signed_url.url}")

    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": signed_url.url,
        },
        include_image_base64=True
    )
    print("OCR processing complete.")

    if ocr_response and ocr_response.pages:
        output_file_name = "output.txt"
        with open(output_file_name, 'w', encoding='utf-8') as file:
            for page in ocr_response.pages:
                file.write(page.markdown)
                file.write('\n\n---\n\n')
        print(f"Text has been successfully written to {output_file_name}")
    else:
        print("OCR response is empty or invalid. No text to write.")

except FileNotFoundError:
    print(f"Error: The PDF file '{pdf_to_upload_path}' was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")