import argparse
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def inspect_chromadb(persist_directory):
    """
    Inspect the ChromaDB collection to view documents and metadata.
    """
    try:
        # Initialize the embedding model
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Load ChromaDB
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

        # Get the collection
        collection = db._collection

        # Fetch all documents and metadata
        documents = collection.get(include=["metadatas", "documents"])

        # Print the schema and metadata
        print("ChromaDB Schema and Metadata:")
        for i, (doc, metadata) in enumerate(zip(documents["documents"], documents["metadatas"])):
            print(f"Document {i+1}:")
            print(f"Text: {doc}")
            print(f"Metadata: {metadata}")
            print("-" * 50)

    except Exception as e:
        print(f"Error inspecting ChromaDB: {e}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Inspect ChromaDB collection.")
    parser.add_argument("--persist_directory", help="Directory where ChromaDB is stored", required=True)
    args = parser.parse_args()

    # Inspect the ChromaDB collection
    inspect_chromadb(args.persist_directory)