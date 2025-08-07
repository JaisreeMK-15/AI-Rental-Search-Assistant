import os
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, CSVLoader

# Define the path where the pre-computed index will be stored
FAISS_INDEX_PATH = "faiss_index"

def get_or_create_vector_store(folder_path, embeddings):
    """
    Loads a pre-existing FAISS vector store if it exists. If not, it creates
    the vector store from the data folder and saves it for future use.
    """
    # Check if the saved index path already exists
    if os.path.exists(FAISS_INDEX_PATH):
        # If it exists, load the local index directly
        print("Loading existing FAISS index...")
        # Note: allow_dangerous_deserialization is required for loading local FAISS with langchain
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH, 
            embeddings,
            allow_dangerous_deserialization=True 
        )
        print("FAISS index loaded successfully.")
        return vector_store
    else:
        # If it doesn't exist, create it from scratch
        print("FAISS index not found. Creating a new one...")
        if not os.path.exists(folder_path):
            print(f"Error: Data folder '{folder_path}' not found.")
            return None
        try:
            # Load all .txt and .csv files from the data directory
            txt_loader = DirectoryLoader(folder_path, glob="**/*.txt", show_progress=True)
            txt_documents = txt_loader.load()

            csv_file_path = os.path.join(folder_path, 'bengaluru_rentals.csv')
            csv_documents = []
            if os.path.exists(csv_file_path):
                csv_loader = CSVLoader(file_path=csv_file_path, encoding='utf-8')
                csv_documents = csv_loader.load()
            
            all_documents = txt_documents + csv_documents
            if not all_documents:
                print(f"Warning: No documents found in '{folder_path}'.")
                return None

            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            texts = text_splitter.split_documents(all_documents)

            # Create the vector store
            print("Creating vector store from all documents...")
            vector_store = FAISS.from_documents(texts, embeddings)
            
            # --- SAVE THE NEW INDEX ---
            # Save the newly created index to the specified path for future runs
            print(f"Saving new FAISS index to {FAISS_INDEX_PATH}...")
            vector_store.save_local(FAISS_INDEX_PATH)
            print("FAISS index saved successfully.")
            return vector_store
        except Exception as e:
            print(f"Error creating vector store: {e}")
            raise RuntimeError(f"Failed to create vector store: {e}")


def get_context_from_rag(vector_store, query, k=5):
    """
    Performs a similarity search on the vector store to find relevant document chunks.
    (This function remains unchanged)
    """
    if not vector_store:
        return "Vector store is not available."
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        relevant_docs = retriever.get_relevant_documents(query)
        context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
        return context
    except Exception as e:
        print(f"Error retrieving context from RAG: {e}")
        return "Could not retrieve context due to an error."
