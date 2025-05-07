from utils.gemini_utils import get_ai_response as gemini_get_ai_response, analyze_image_content as gemini_analyze_image, get_embedding as gemini_get_embedding
from utils.openai_utils import get_ai_response as openai_get_ai_response, analyze_image_content as openai_analyze_image, get_embedding as openai_get_embedding
from utils.anthropic_utils import get_ai_response as anthropic_get_ai_response, analyze_image_content as anthropic_analyze_image, get_embedding as anthropic_get_embedding
from utils.gemini_langchain_utils import get_rag_response as gemini_get_rag_response
from utils.gemini_langchain_utils import get_multimodal_response as gemini_get_multimodal_response
from utils.langchain_utils import get_rag_response as openai_get_rag_response
from utils.langchain_utils import get_multimodal_response as openai_get_multimodal_response
from utils.anthropic_langchain_utils import get_rag_response as anthropic_get_rag_response
from utils.anthropic_langchain_utils import get_multimodal_response as anthropic_get_multimodal_response
from utils.db_utils import add_message_to_db, get_conversation_messages
from utils.vector_store import search_vector_store

class Agent:
    """
    Agent class to manage conversation flow and determine the appropriate response strategy.
    """
    
    def __init__(self):
        """
        Initialize the agent with empty conversation history.
        """
        self.conversation_id = None
        self.history = []
        self.selected_model = "gemini"  # default model
        self.api_key = None
        self.model_version = None
    
    def set_conversation_id(self, conversation_id):
        """
        Set the active conversation ID and load its history.
        
        Args:
            conversation_id (int): The conversation ID to set.
        """
        self.conversation_id = conversation_id
        self.load_history_from_db()
    
    def load_history_from_db(self):
        """
        Load conversation history from the database.
        """
        if not self.conversation_id:
            return
        
        # Get messages from database
        messages = get_conversation_messages(self.conversation_id)
        
        # Convert to agent history format
        self.history = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
        ]
    
    def set_model(self, model_name, api_key=None, model_version=None):
        """
        Set which AI model to use.
        
        Args:
            model_name (str): 'gemini', 'openai', or 'anthropic'
            api_key (str, optional): API key for the selected model
            model_version (str, optional): Specific version of the model to use
        """
        self.selected_model = model_name
        self.api_key = api_key
        self.model_version = model_version
    
    def add_to_history(self, role, content):
        """
        Add a message to the conversation history and database.
        
        Args:
            role (str): The role of the message sender ('user' or 'assistant').
            content (str): The message content.
        """
        # Add to memory
        self.history.append({"role": role, "content": content})
        
        # Add to database if conversation_id is set
        if self.conversation_id:
            add_message_to_db(self.conversation_id, role, content)
    
    def get_conversation_history(self, limit=20):
        """
        Get the conversation history.
        
        Args:
            limit (int): Maximum number of messages to retrieve
            
        Returns:
            list: List of message dictionaries
        """
        if not self.conversation_id:
            return []
        
        return get_conversation_messages(self.conversation_id, limit)
    
    def get_chatbot_format_history(self, history=None):
        """
        Format the conversation history for Streamlit chat display.
        
        Args:
            history (list, optional): List of message dictionaries from the database
            
        Returns:
            list: List of (role, content) tuples for Streamlit chat messages
        """
        formatted_messages = []
        
        if history:
            # Use provided history (usually from database)
            for msg in history:
                role = "user" if msg["role"] == "user" else "assistant"
                content = msg["content"]
                
                # Add the message content
                formatted_messages.append((role, content))
        else:
            # Use in-memory history
            for msg in self.history:
                role = "user" if msg["role"] == "user" else "assistant"
                content = msg["content"]
                
                # Add the message content
                formatted_messages.append((role, content))
            
        return formatted_messages
    
    def process_query(self, query, vector_store=None, image=None, document_content=None):
        """
        Process a user query and determine the appropriate response strategy.
        
        Args:
            query (str): The user's text query
            vector_store (tuple, optional): FAISS vector store for RAG
            image (Image, optional): PIL Image if an image was uploaded
            document_content (str, optional): Extracted text content from a document
            
        Returns:
            str: The AI's response.
        """
        try:
            # Validate API key
            if not self.api_key:
                return "Please enter your API key in the sidebar to use this chatbot."
            
            # Determine which model's functions to use
            if self.selected_model == "gemini":
                get_ai_response = gemini_get_ai_response
                get_rag_response = gemini_get_rag_response
                get_multimodal_response = gemini_get_multimodal_response
                analyze_image = gemini_analyze_image
                get_embedding = gemini_get_embedding
            elif self.selected_model == "anthropic":
                get_ai_response = anthropic_get_ai_response
                get_rag_response = anthropic_get_rag_response
                get_multimodal_response = anthropic_get_multimodal_response
                analyze_image = anthropic_analyze_image
                get_embedding = anthropic_get_embedding
            else:  # OpenAI
                get_ai_response = openai_get_ai_response
                get_rag_response = openai_get_rag_response
                get_multimodal_response = openai_get_multimodal_response
                analyze_image = openai_analyze_image
                get_embedding = openai_get_embedding
            
            # Add user query to history
            self.add_to_history("user", query)
            
            # Extract searchable content from user query
            searchable_query = query
            
            # Various response strategies based on inputs
            response = None
            relevant_info = None
            
            # Case 1: Image + Query - Use multimodal processing
            if image and query:
                # First analyze the image
                image_analysis = analyze_image(image, self.api_key, self.model_version)
                
                # Then combine with the query for a response
                response = get_multimodal_response(query, image_analysis, self.api_key, self.model_version)
            
            # Case 2: Document + Query - Use RAG approach
            elif document_content and query:
                # Make the document content searchable in the query
                searchable_query = f"{query}\n\nDocument content: {document_content[:1000]}"
                
                # Get an embedding for the query
                query_embedding = get_embedding(query, self.api_key, self.model_version)
                
                # Prepare document chunks for searching (simple approach)
                document_chunks = []
                chunk_size = 2000
                for i in range(0, len(document_content), chunk_size):
                    chunk = document_content[i:i+chunk_size]
                    document_chunks.append({"content": chunk, "embedding": get_embedding(chunk, self.api_key, self.model_version)})
                
                # Find relevant chunks
                relevant_chunks = []
                for chunk in document_chunks:
                    # Add the content as relevant info
                    relevant_chunks.append(chunk["content"])
                
                # Use RAG to generate a response
                if relevant_chunks:
                    response = get_rag_response(query, relevant_chunks, self.api_key, self.model_version)
                    relevant_info = relevant_chunks
            
            # Case 3: Vector store search + Query - Use RAG approach
            elif vector_store and query:
                # Get an embedding for the query
                query_embedding = get_embedding(query, self.api_key, self.model_version)
                
                # Search the vector store for relevant information
                relevant_info = search_vector_store(vector_store, query_embedding)
                
                # Use RAG to generate a response if relevant info found
                if relevant_info:
                    response = get_rag_response(query, relevant_info, self.api_key, self.model_version)
            
            # Default case: Simple query-response
            if not response:
                response = get_ai_response(query, context=self.history[-10:] if len(self.history) > 0 else None, api_key=self.api_key, model_version=self.model_version)
            
            # Add the response to conversation history
            self.add_to_history("assistant", response)
            
            return response
            
        except Exception as e:
            error_message = f"Error processing your query: {str(e)}"
            # Don't add errors to conversation history
            return error_message
    
    def get_conversation_context(self):
        """
        Get the current conversation context.
        
        Returns:
            list: The conversation history.
        """
        return self.history
    
    def reset(self):
        """
        Reset the agent's state.
        """
        self.history = []
