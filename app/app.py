import logging
import argparse
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def split_pdf_to_text(pdf_path, output_file="output.txt", chunk_size=1000, chunk_overlap=200):
    try:
        # Load the PDF document
        logging.info(f"Loading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
    except Exception as e:
        logging.error(f"Error loading PDF: {e}")
        return

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Split the document into texts
    logging.info("Splitting PDF into text chunks...")
    texts = text_splitter.split_documents(documents)

    # Save the split texts to a file
    with open(output_file, "w") as f:
        for i, text in enumerate(texts):
            f.write(f"Chunk {i+1} (Page {text.metadata['page']}):\n{text.page_content}\n\n")

    logging.info(f"Text chunks saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a PDF into text chunks.")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--output", help="Output file name", default="output.txt")
    parser.add_argument("--chunk_size", type=int, help="Size of each text chunk", default=1000)
    parser.add_argument("--chunk_overlap", type=int, help="Overlap between chunks", default=200)
    args = parser.parse_args()

    split_pdf_to_text(args.pdf_path, args.output, args.chunk_size, args.chunk_overlap)