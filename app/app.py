import logging
import argparse
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import ollama

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class PDFProcessor:
    """Handles loading and splitting PDF documents."""

    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_pdf_to_text(self, pdf_path):
        """Load a PDF and split it into text chunks, including metadata."""
        try:
            logging.info(f"Loading PDF: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            logging.info(f"Loaded {len(documents)} pages from the PDF.")
        except Exception as e:
            logging.error(f"Error loading PDF: {e}")
            return None

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        logging.info("Splitting PDF into text chunks...")
        texts = text_splitter.split_documents(documents)
        logging.info(f"Split the PDF into {len(texts)} chunks.")

        for text in texts:
            if not hasattr(text, 'metadata'):
                text.metadata = {}
            text.metadata['source'] = pdf_path
            if 'page' not in text.metadata:
                text.metadata['page'] = text.metadata.get('page', 'Unknown')

        if texts:
            logging.info(f"First chunk:\n{texts[0].page_content}\nMetadata: {texts[0].metadata}\n")

        return texts

class ChromaDBManager:
    """Manages ChromaDB operations like storing and querying."""

    def __init__(self, persist_directory="my_chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def store_in_chromadb(self, texts):
        """Store text chunks and embeddings in ChromaDB."""
        try:
            logging.info("Storing text chunks in ChromaDB...")
            db = Chroma.from_documents(texts, self.embeddings, persist_directory=self.persist_directory)
            logging.info(f"Text chunks stored in ChromaDB at {self.persist_directory}")
            return db
        except Exception as e:
            logging.error(f"Error storing in ChromaDB: {e}")
            return None

    def query_chromadb(self, query, k=5):
        """Query ChromaDB for relevant text chunks."""
        try:
            db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
            results = db.similarity_search(query, k=k)
            logging.info(f"Retrieved {len(results)} relevant chunks from ChromaDB.")
            for i, chunk in enumerate(results):
                logging.info(f"Chunk {i+1}:\n{chunk.page_content}\n")
            return results
        except Exception as e:
            logging.error(f"Error querying ChromaDB: {e}")
            return None

class LLMResponseGenerator:
    """Generates responses using Llama 3 via Ollama."""

    def generate_response(self, query, relevant_chunks):
        """Generate a response using Llama 3."""
        try:
            context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
            prompt = (
                f"You are a helpful assistant. Use the following context to answer the question.\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n"
                f"Answer:"
            )
            logging.info("Generating response with Llama 3...")
            response = ollama.generate(model="llama3.2", prompt=prompt)
            return response["response"]
        except Exception as e:
            logging.error(f"Error generating response with Llama 3: {e}")
            return None

class RAGPipeline:
    """Orchestrates the RAG pipeline."""

    def __init__(self, pdf_path=None, persist_directory="my_chroma_db", chunk_size=1000, chunk_overlap=200):
        self.pdf_path = pdf_path
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.pdf_processor = PDFProcessor(chunk_size, chunk_overlap)
        self.chroma_manager = ChromaDBManager(persist_directory)
        self.llm_generator = LLMResponseGenerator()

    def run(self):
        """Run the RAG pipeline."""
        if self.pdf_path:
            texts = self.pdf_processor.split_pdf_to_text(self.pdf_path)
            if not texts:
                logging.error("Failed to load or split the PDF. Exiting.")
                return

            if not self.chroma_manager.store_in_chromadb(texts):
                logging.error("Failed to store chunks in ChromaDB. Exiting.")
                return

        logging.info("Starting RAG pipeline with ChromaDB and Llama 3.")
        while True:
            query = input("\nYou: ")
            if query.lower() in ["exit", "quit"]:
                logging.info("Exiting the RAG pipeline.")
                break

            relevant_chunks = self.chroma_manager.query_chromadb(query)
            if not relevant_chunks:
                print("No relevant chunks found. Try a different query.")
                continue

            response = self.llm_generator.generate_response(query, relevant_chunks)
            if response:
                print(f"Assistant: {response}")
            else:
                print("Failed to generate a response. Please try again.")

def main():
    """Parse command-line arguments and run the RAG pipeline."""
    parser = argparse.ArgumentParser(description="Embed a PDF into ChromaDB and query it with Llama 3.")
    parser.add_argument("--pdf_path", help="Path to the PDF file (optional)", default=None)
    parser.add_argument("--persist_directory", help="Directory to store ChromaDB", default="my_chroma_db")
    parser.add_argument("--chunk_size", type=int, help="Size of each text chunk", default=1000)
    parser.add_argument("--chunk_overlap", type=int, help="Overlap between chunks", default=200)
    args = parser.parse_args()

    pipeline = RAGPipeline(args.pdf_path, args.persist_directory, args.chunk_size, args.chunk_overlap)
    pipeline.run()

if __name__ == "__main__":
    main()