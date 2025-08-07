from langchain_community.embeddings import HuggingFaceEmbeddings

def get_huggingface_embeddings(model_name="all-MiniLM-L6-v2"):
    """
    Loads a pre-trained sentence-transformer model from HuggingFace.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        HuggingFaceEmbeddings: The loaded embedding model object.
    """
    try:
        # Initialize the embedding model. It will be downloaded automatically if not cached.
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'} # Use CPU for broad compatibility
        )
        print("Embedding model loaded successfully.")
        return embeddings
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        # Propagate the error to be handled by the main app
        raise RuntimeError(f"Failed to load embedding model: {e}")

