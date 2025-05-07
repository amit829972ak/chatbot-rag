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
            # When provided with history from the database
            for msg in history:
                # Convert 'user' and 'assistant' roles to Streamlit format
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
            # Validate API key availability
            if not self.api_key:
                return f"Please enter a valid API key for the selected {self.selected_model.capitalize()} model in the sidebar."
            
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
            
            # Add the user query to history
            self.add_to_history("user", query)
            
            # Determine response strategy based on inputs
            response = None
            
            # Case 1: If we have an image, analyze it and respond to the query in that context
            if image:
                try:
                    # Step 1: Analyze the image
                    image_analysis = analyze_image(image, self.api_key, self.model_version)
                    
                    # Step 2: Generate a response that incorporates both the query and image analysis
                    response = get_multimodal_response(
                        query=query, 
                        image_analysis=image_analysis, 
                        api_key=self.api_key,
                        model_version=self.model_version
                    )
                except Exception as e:
                    response = f"Error analyzing image: {str(e)}"
            
            # Case 2: If we have document content, use that as context
            elif document_content:
                try:
                    # Include the document in the response
                    system_prompt = """
                    You are a helpful assistant that specializes in analyzing documents and answering questions about them.
                    
                    For structured data like CSV, TSV, or XLSX content, format your response in a clear, tabular way when appropriate.
                    For PDF content, include page references if available.
                    Be concise but comprehensive in your analysis.
                    """
                    
                    # Truncate document if it's too long (some API limits)
                    max_len = 50000  # Reasonable limit for most APIs
                    truncated_content = document_content[:max_len]
                    if len(document_content) > max_len:
                        truncated_content += f"\n\n[Document truncated due to length. {len(document_content) - max_len} characters omitted.]"
                    
                    # Combine the query with document content instruction
                    enhanced_prompt = f"""
                    I'm going to ask a question about the following document content:
                    
                    {truncated_content}
                    
                    My question is: {query}
                    
                    Please provide a detailed answer based on the content of the document.
                    """
                    
                    # Get response with document context
                    response = get_ai_response(
                        prompt=enhanced_prompt, 
                        system_prompt=system_prompt,  
                        context=self.get_conversation_context(),
                        api_key=self.api_key,
                        model_version=self.model_version
                    )
                except Exception as e:
                    response = f"Error processing document: {str(e)}"
            
            # Case 3: Use RAG if we have a vector store
            elif vector_store:
                try:
                    # Get the query embedding
                    query_embedding = get_embedding(query, self.api_key, self.model_version)
                    
                    # Search the vector store
                    relevant_docs = search_vector_store(vector_store, query_embedding)
                    
                    if relevant_docs and len(relevant_docs) > 0:
                        # We have relevant context, use RAG
                        response = get_rag_response(
                            query=query, 
                            context=relevant_docs, 
                            api_key=self.api_key,
                            model_version=self.model_version
                        )
                    else:
                        # No relevant context found, fall back to standard response
                        response = get_ai_response(
                            prompt=query, 
                            context=self.get_conversation_context(),
                            api_key=self.api_key,
                            model_version=self.model_version
                        )
                except Exception as e:
                    # If RAG fails, fall back to standard response
                    response = get_ai_response(
                        prompt=query, 
                        context=self.get_conversation_context(),
                        api_key=self.api_key,
                        model_version=self.model_version
                    )
            
            # Case 4: Default to standard response
            else:
                response = get_ai_response(
                    prompt=query, 
                    context=self.get_conversation_context(),
                    api_key=self.api_key,
                    model_version=self.model_version
                )
            
            # Add the response to history
            self.add_to_history("assistant", response)
            
            return response
            
        except Exception as e:
            error_message = f"Error processing your query: {str(e)}"
            
            # Add error to history
            self.add_to_history("assistant", error_message)
            
            return error_message
    
    def get_conversation_context(self):
        """
        Get the current conversation context.
        
        Returns:
            list: The conversation history.
        """
        # Get the most recent messages (limited to avoid token limits)
        recent_history = self.history[-10:] if len(self.history) > 10 else self.history
        return recent_history
    
    def reset(self):
        """
        Reset the agent's state.
        """
        self.history = []
