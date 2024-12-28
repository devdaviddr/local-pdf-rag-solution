import logging
import argparse
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import ollama

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def split_pdf_to_text(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    Load a PDF and split it into text chunks, including metadata like page number and document name.
    """
    try:
        # Load the PDF document
        logging.info(f"Loading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        logging.info(f"Loaded {len(documents)} pages from the PDF.")
    except Exception as e:
        logging.error(f"Error loading PDF: {e}")
        return None

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Split the document into texts
    logging.info("Splitting PDF into text chunks...")
    texts = text_splitter.split_documents(documents)
    logging.info(f"Split the PDF into {len(texts)} chunks.")

    # Add metadata (page number and document name) to each chunk
    for text in texts:
        if not hasattr(text, 'metadata'):
            text.metadata = {}
        text.metadata['source'] = pdf_path  # Document name (file path)
        if 'page' not in text.metadata:  # Ensure page number is included
            text.metadata['page'] = text.metadata.get('page', 'Unknown')

    # Debug: Print the first chunk with metadata
    if texts:
        logging.info(f"First chunk:\n{texts[0].page_content}\nMetadata: {texts[0].metadata}\n")

    return texts

def store_in_chromadb(texts, persist_directory="my_chroma_db"):
    """
    Generate embeddings and store the text chunks in ChromaDB.
    """
    try:
        # Initialize the embedding model
        logging.info("Initializing Hugging Face embedding model...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Lightweight embedding model

        # Store the texts and embeddings in ChromaDB
        logging.info("Storing text chunks in ChromaDB...")
        db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
        logging.info(f"Text chunks stored in ChromaDB at {persist_directory}")
        return db
    except Exception as e:
        logging.error(f"Error storing in ChromaDB: {e}")
        return None

def query_chromadb(query, persist_directory="my_chroma_db", k=5):  # Increased k to 5
    """
    Query ChromaDB for relevant text chunks.
    """
    try:
        # Initialize the embedding model
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Load ChromaDB
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

        # Retrieve relevant chunks
        results = db.similarity_search(query, k=k)
        logging.info(f"Retrieved {len(results)} relevant chunks from ChromaDB.")
        
        # Debug: Print the retrieved chunks
        for i, chunk in enumerate(results):
            logging.info(f"Chunk {i+1}:\n{chunk.page_content}\n")
        
        return results
    except Exception as e:
        logging.error(f"Error querying ChromaDB: {e}")
        return None

def generate_response_with_llama(query, relevant_chunks):
    """
    Generate a response using Llama 3 via Ollama.
    """
    try:
        # Combine the query and relevant chunks into a prompt
        context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
        prompt = (
            f"You are a helpful assistant. Use the following context to answer the question.\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )

        # Generate the response using Llama 3
        logging.info("Generating response with Llama 3...")
        response = ollama.generate(model="llama3.2", prompt=prompt)  # Corrected model name
        return response["response"]
    except Exception as e:
        logging.error(f"Error generating response with Llama 3: {e}")
        return None

def main(pdf_path=None, persist_directory="my_chroma_db", chunk_size=1000, chunk_overlap=200):
    """
    Main function to handle embedding PDFs and querying ChromaDB with Llama 3.
    """
    if pdf_path:
        # Step 1: Load the PDF and split it into chunks
        texts = split_pdf_to_text(pdf_path, chunk_size, chunk_overlap)
        if not texts:
            logging.error("Failed to load or split the PDF. Exiting.")
            return

        # Step 2: Store the chunks in ChromaDB
        if not store_in_chromadb(texts, persist_directory):
            logging.error("Failed to store chunks in ChromaDB. Exiting.")
            return

    # Step 3: Query ChromaDB and generate responses with Llama 3
    logging.info("Starting RAG pipeline with ChromaDB and Llama 3.")
    while True:
        # Get user input
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            logging.info("Exiting the RAG pipeline.")
            break

        # Step 4: Query ChromaDB for relevant chunks
        relevant_chunks = query_chromadb(query, persist_directory)
        if not relevant_chunks:
            print("No relevant chunks found. Try a different query.")
            continue

        # Step 5: Generate a response using Llama 3
        response = generate_response_with_llama(query, relevant_chunks)
        if response:
            print(f"Assistant: {response}")
        else:
            print("Failed to generate a response. Please try again.")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Embed a PDF into ChromaDB and query it with Llama 3.")
    parser.add_argument("--pdf_path", help="Path to the PDF file (optional)", default=None)
    parser.add_argument("--persist_directory", help="Directory to store ChromaDB", default="my_chroma_db")
    parser.add_argument("--chunk_size", type=int, help="Size of each text chunk", default=1000)
    parser.add_argument("--chunk_overlap", type=int, help="Overlap between chunks", default=200)
    args = parser.parse_args()

    # Run the pipeline
    main(args.pdf_path, args.persist_directory, args.chunk_size, args.chunk_overlap)