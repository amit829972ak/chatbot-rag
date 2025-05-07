from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory

from utils.gemini_utils import DEFAULT_TEXT_MODEL, GEMINI_MODELS

def create_rag_chain(api_key=None, model_version=None):
    """
    Create a RAG chain using LangChain with Google Gemini.
    
    Args:
        api_key (str, optional): Google Gemini API key.
        
    Returns:
        LLMChain: A LangChain chain for RAG responses.
    """
    # Use the specified model version or fall back to default
    selected_model = model_version or DEFAULT_TEXT_MODEL
    
    # Make sure the selected model is valid
    if selected_model not in GEMINI_MODELS:
        selected_model = DEFAULT_TEXT_MODEL
    
    # Get the API name for the selected model
    model_name = GEMINI_MODELS[selected_model]["api_name"]
    
    # Create the language model
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.2
    )
    
    # Define the RAG prompt template
    template = """
    You are a helpful AI assistant with access to relevant information from a knowledge base.
    
    User Query: {query}
    
    Relevant Information from Knowledge Base:
    {context}
    
    Based on the relevant information provided and your general knowledge, please give a comprehensive and accurate response to the user's query.
    If the information is not sufficient to answer the query completely, acknowledge that and provide what you can.
    If the knowledge base doesn't contain relevant information, rely on your general knowledge but make it clear you're doing so.
    
    Your response:
    """
    
    # Create the prompt
    prompt = PromptTemplate(
        template=template,
        input_variables=["query", "context"]
    )
    
    # Create and return the LLMChain
    return LLMChain(llm=llm, prompt=prompt)

def get_rag_response(query, context, api_key=None, model_version=None):
    """
    Get a RAG-enhanced response using the provided context.
    
    Args:
        query (str): The user's query.
        context (list): List of relevant information from the vector store.
        api_key (str, optional): Google Gemini API key.
        
    Returns:
        str: The RAG-enhanced response.
    """
    try:
        # Convert context to string if it's not already
        if isinstance(context, list):
            # Handle both string lists and document lists
            context_str = ""
            for item in context:
                if hasattr(item, 'page_content'):
                    # LangChain document format
                    content = item.page_content
                    if hasattr(item, 'metadata') and item.metadata:
                        content += f" (Source: {item.metadata.get('source', 'Unknown')})"
                    context_str += content + "\n\n"
                else:
                    # Plain string
                    context_str += str(item) + "\n\n"
        else:
            context_str = str(context)
        
        # Create the RAG chain
        chain = create_rag_chain(api_key, model_version)
        
        # Run the chain
        response = chain.run(query=query, context=context_str)
        
        return response
    
    except Exception as e:
        return f"Error getting RAG response from Google Gemini: {str(e)}"

def get_multimodal_response(query, image_analysis, api_key=None, model_version=None):
    """
    Get a response that incorporates both text query and image analysis.
    
    Args:
        query (str): The user's text query.
        image_analysis (str): Analysis of the uploaded image.
        api_key (str, optional): Google Gemini API key.
        
    Returns:
        str: Response that considers both the text and image.
    """
    try:
        # Use the specified model version or fall back to default
        selected_model = model_version or DEFAULT_TEXT_MODEL
        
        # Make sure the selected model is valid
        if selected_model not in GEMINI_MODELS:
            selected_model = DEFAULT_TEXT_MODEL
        
        # Get the API name for the selected model
        model_name = GEMINI_MODELS[selected_model]["api_name"]
        
        # Create the language model
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.2
        )
        
        # Define the multimodal prompt template
        template = """
        You are a helpful AI assistant that can understand both text and images.
        
        User Query: {query}
        
        Image Analysis: {image_analysis}
        
        Based on the user's query and the image analysis provided, please give a comprehensive response.
        Make sure to address all aspects of the user's query in relation to the image when relevant.
        
        Your response:
        """
        
        # Create the prompt
        prompt = PromptTemplate(
            template=template,
            input_variables=["query", "image_analysis"]
        )
        
        # Create the chain
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Run the chain
        response = chain.run(query=query, image_analysis=image_analysis)
        
        return response
    
    except Exception as e:
        return f"Error getting multimodal response from Google Gemini: {str(e)}"
