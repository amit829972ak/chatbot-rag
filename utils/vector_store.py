import logging
import faiss
import numpy as np
import json
from utils.db_utils import get_all_knowledge_items, add_knowledge_item
from utils.openai_utils import get_embedding as openai_get_embedding
from utils.gemini_utils import get_embedding as gemini_get_embedding

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_vector_store(model_version=None):
    """
    Initialize the FAISS vector store with sample knowledge.
    
    Returns:
        tuple: (FAISS index, list of documents)
    """
    # Load existing knowledge items from database
    items = get_all_knowledge_items()
    
    # If no items, add some sample knowledge
    if not items:
        logger.info("No knowledge items found. Attempting to add sample knowledge.")
        
        try:
            # Try to use OpenAI for embeddings
            try:
                embedder = openai_get_embedding
                api_key = None
                logger.info("Using OpenAI for embeddings.")
            except:
                # Fall back to Gemini
                embedder = gemini_get_embedding
                api_key = None
                logger.info("Using Gemini for embeddings.")
            
            # Sample knowledge items
            sample_knowledge = [
                "Retrieval-Augmented Generation (RAG) is an AI framework that enhances LLM responses by retrieving relevant information from external sources.",
                "FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors.",
                "Vector embeddings transform text, images, or other data into numerical vectors capturing semantic meaning.",
                "Multimodal AI systems can process and relate multiple types of inputs such as text, images, audio, and video.",
                "Streamlit is a Python library that makes it easy to create web apps for data science and machine learning projects."
            ]
            
            # Add sample knowledge to database with embeddings
            for item in sample_knowledge:
                embedding = embedder(item, api_key)
                add_knowledge_item(item, embedding)
            
            # Reload items
            items = get_all_knowledge_items()
            
        except Exception as e:
            logger.warning(f"Failed to add sample knowledge: {str(e)}")
            logger.warning("No API key available. Skipping sample knowledge creation.")
            print("No API key available. Skipping sample knowledge creation.")
    
    # Create FAISS index if we have items
    if items:
        # Extract embeddings and documents
        embeddings = [item["embedding"] for item in items]
        documents = [item["content"] for item in items]
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)
        
        return (index, documents)
    else:
        # Create empty index
        logger.warning("No knowledge items could be loaded. Creating empty index.")
        print("No knowledge items could be loaded. Creating empty index.")
        dimension = 768  # Default embedding dimension
        index = faiss.IndexFlatL2(dimension)
        return (index, [])

def search_vector_store(vector_store, query_embedding, k=3):
    """
    Search the vector store for documents similar to the query.
    
    Args:
        vector_store (tuple): (FAISS index, list of documents)
        query_embedding (list): The query embedding
        k (int): Number of results to return
        
    Returns:
        list: The most relevant documents
    """
    try:
        # Unpack vector store
        index, documents = vector_store
        
        # If empty index or empty documents, return empty list
        if index.ntotal == 0 or not documents:
            return []
        
        # Convert query embedding to numpy array
        query_embedding_array = np.array([query_embedding]).astype('float32')
        
        # Search index
        distances, indices = index.search(query_embedding_array, min(k, index.ntotal))
        
        # Get results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(documents):  # Ensure the index is valid
                results.append(documents[idx])
        
        return results
    except Exception as e:
        logger.error(f"Error searching vector store: {str(e)}")
        return []
