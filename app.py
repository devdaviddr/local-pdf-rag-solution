import logging
import argparse
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
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
    """Manages ChromaDB operations like storing, querying, and reindexing."""

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

    def add_to_chromadb(self, texts):
        """Add new text chunks to an existing ChromaDB collection."""
        try:
            logging.info("Adding text chunks to ChromaDB...")
            db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
            db.add_documents(texts)
            logging.info(f"Added {len(texts)} chunks to ChromaDB.")
            return db
        except Exception as e:
            logging.error(f"Error adding to ChromaDB: {e}")
            return None

    def reindex_chromadb(self, texts):
        """Reindex ChromaDB by deleting existing documents and adding new ones."""
        try:
            logging.info("Reindexing ChromaDB...")
            db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
            db.delete_collection()  # Delete the existing collection
            db = Chroma.from_documents(texts, self.embeddings, persist_directory=self.persist_directory)
            logging.info(f"Reindexed ChromaDB with {len(texts)} chunks.")
            return db
        except Exception as e:
            logging.error(f"Error reindexing ChromaDB: {e}")
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

    def __init__(self, data_directory=None, persist_directory="my_chroma_db", chunk_size=1000, chunk_overlap=200, reindex=False):
        self.data_directory = data_directory
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.reindex = reindex
        self.pdf_processor = PDFProcessor(chunk_size, chunk_overlap)
        self.chroma_manager = ChromaDBManager(persist_directory)
        self.llm_generator = LLMResponseGenerator()

    def process_pdfs_in_directory(self):
        """Process all PDFs in the specified directory."""
        if not self.data_directory:
            logging.error("No data directory provided. Exiting.")
            return None

        if not os.path.exists(self.data_directory):
            logging.error(f"Data directory does not exist: {self.data_directory}")
            return None

        all_texts = []
        for filename in os.listdir(self.data_directory):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(self.data_directory, filename)
                texts = self.pdf_processor.split_pdf_to_text(pdf_path)
                if texts:
                    all_texts.extend(texts)
                else:
                    logging.error(f"Failed to process PDF: {pdf_path}")

        return all_texts

    def run(self):
        """Run the RAG pipeline."""
        if self.data_directory:
            all_texts = self.process_pdfs_in_directory()
            if not all_texts:
                logging.error("No valid PDFs found in the data directory. Exiting.")
                return

            if self.reindex:
                if not self.chroma_manager.reindex_chromadb(all_texts):
                    logging.error("Failed to reindex ChromaDB. Exiting.")
                    return
            else:
                if not self.chroma_manager.add_to_chromadb(all_texts):
                    logging.error("Failed to add chunks to ChromaDB. Exiting.")
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
    parser = argparse.ArgumentParser(description="Embed PDFs into ChromaDB and query it with Llama 3.")
    parser.add_argument("--data_directory", help="Directory containing PDF files (optional)", default=None)
    parser.add_argument("--persist_directory", help="Directory to store ChromaDB", default="my_chroma_db")
    parser.add_argument("--chunk_size", type=int, help="Size of each text chunk", default=1000)
    parser.add_argument("--chunk_overlap", type=int, help="Overlap between chunks", default=200)
    parser.add_argument("--reindex", action="store_true", help="Reindex the ChromaDB collection")
    args = parser.parse_args()

    pipeline = RAGPipeline(
        data_directory=args.data_directory,
        persist_directory=args.persist_directory,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        reindex=args.reindex
    )
    pipeline.run()

if __name__ == "__main__":
    main()