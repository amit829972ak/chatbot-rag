import os
import io
import base64
import google.generativeai as genai
from PIL import Image

# Gemini model options 
GEMINI_MODELS = {
    "gemini-1.0-pro": {
        "name": "Gemini 1.0 Pro",
        "description": "Free tier model - balanced performance",
        "api_name": "gemini-1.0-pro"
    },
    "gemini-1.5-flash-latest": {
        "name": "Gemini 1.5 Flash",
        "description": "Faster response times, cost-effective",
        "api_name": "gemini-1.5-flash-latest"
    },
    "gemini-1.5-pro-latest": {
        "name": "Gemini 1.5 Pro",
        "description": "Latest model - most powerful reasoning",
        "api_name": "gemini-1.5-pro-latest"
    }
}

# Default text model
DEFAULT_TEXT_MODEL = "gemini-1.0-pro"  # Free tier model

# Default vision model 
DEFAULT_VISION_MODEL = "gemini-1.5-pro-latest"  # Best for multimodal

def get_gemini_client(api_key=None):
    """
    Configure the Gemini API with the provided API key.
    
    Args:
        api_key (str, optional): Google Gemini API key
        
    Returns:
        None: Configuration is set globally
        
    Raises:
        Exception: If API key is invalid or missing
    """
    if not api_key:
        raise Exception("No API key provided. Please provide a Google Gemini API key.")
    
    try:
        # Configure the Google Gemini client
        genai.configure(api_key=api_key)
        
        # Simple validation of API key format
        if len(api_key) < 20:
            raise Exception("API key appears to be too short. Google API keys are typically longer.")
        
        return True
    
    except Exception as e:
        error_msg = str(e).lower()
        if "invalid" in error_msg and "api" in error_msg:
            raise Exception("Invalid API key. Please check your Google Gemini API key and try again.")
        elif "rate" in error_msg or "quota" in error_msg or "limit" in error_msg:
            raise Exception("API rate limit reached. Please try again later or use a different API key.")
        else:
            raise Exception(f"Error configuring Google Gemini API: {str(e)}")

def get_ai_response(prompt, system_prompt=None, context=None, api_key=None, model_version=None):
    """
    Get a response from the Google Gemini API.
    
    Args:
        prompt (str): The user's prompt.
        system_prompt (str, optional): System instructions to guide the AI.
        context (list, optional): Previous conversation context.
        api_key (str, optional): Google Gemini API key.
        model_version (str, optional): Specific version of Gemini model to use.
        
    Returns:
        str: The AI's response.
        
    Raises:
        Exception: If there's an API error
    """
    try:
        # Configure the client
        get_gemini_client(api_key)
        
        # Use the specified model version or fall back to default
        selected_model = model_version or DEFAULT_TEXT_MODEL
        
        # Make sure the selected model is valid
        if selected_model not in GEMINI_MODELS:
            selected_model = DEFAULT_TEXT_MODEL
        
        # Get the API name for the selected model
        model_name = GEMINI_MODELS[selected_model]["api_name"]
        
        # Create generation config
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
        }
        
        # Create safety settings (default)
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_ONLY_HIGH",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_ONLY_HIGH",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_ONLY_HIGH",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH",
            },
        ]
        
        # Get the model
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        # Prepare chat history format if context is provided
        messages = []
        if context:
            for message in context:
                role = "user" if message["role"] == "user" else "model"
                # The Gemini chat API expects text content, not dict with 'content' key
                messages.append({"role": role, "parts": [message["content"]]})
        
        # Add system prompt if provided
        if system_prompt:
            # For Gemini, we add system prompt as a user message at the beginning
            if not messages:
                messages.append({"role": "user", "parts": [f"System: {system_prompt}"]})
            
        # Start chat if we have context
        if messages:
            chat = model.start_chat(history=messages)
            response = chat.send_message(prompt)
        else:
            # No context, just send a prompt directly
            response = model.generate_content(prompt)
        
        return response.text
    
    except Exception as e:
        error_msg = str(e)
        if "API key" in error_msg:
            raise Exception("Invalid Google Gemini API key. Please check your API key settings and try again.")
        elif "safety" in error_msg.lower():
            raise Exception(f"The request was blocked due to safety concerns: {error_msg}")
        elif "quota" in error_msg.lower() or "rate" in error_msg.lower():
            raise Exception(f"Google Gemini API quota or rate limit reached: {error_msg}")
        else:
            raise Exception(f"Error from Google Gemini API: {error_msg}")

def encode_image_to_base64(image):
    """
    Encode an image to base64 for API transmission.
    
    Args:
        image (PIL.Image): The image to encode.
        
    Returns:
        str: Base64-encoded image string.
    """
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()
    encoded_image = base64.b64encode(image_bytes).decode('utf-8')
    return encoded_image

def analyze_image_content(image, api_key=None, model_version=None):
    """
    Analyze an image using Google Gemini's vision capabilities.
    
    Args:
        image (PIL.Image): The image to analyze.
        api_key (str, optional): Google Gemini API key.
        model_version (str, optional): Specific version of Gemini model to use.
        
    Returns:
        str: Analysis of the image content.
    """
    try:
        # Configure the client
        get_gemini_client(api_key)
        
        # For image analysis, prefer the vision model
        selected_model = model_version or DEFAULT_VISION_MODEL
        
        # Make sure the selected model is valid for vision
        # Note: Not all Gemini models support vision, in case of non-vision model, use default vision
        if selected_model not in GEMINI_MODELS or selected_model == "gemini-1.0-pro":
            selected_model = DEFAULT_VISION_MODEL
        
        # Get the API name for the selected model
        model_name = GEMINI_MODELS[selected_model]["api_name"]
        
        # Get the model
        model = genai.GenerativeModel(model_name=model_name)
        
        # Convert PIL Image to bytes and then to a MIME format
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        img_bytes = img_bytes.getvalue()
        
        # Create the image content part
        image_part = {"mime_type": "image/jpeg", "data": img_bytes}
        
        # Create the prompt
        prompt = "Analyze this image in detail. Identify objects, people, text, scenes, and other elements. Describe what you see and provide any relevant context or insights."
        
        # Get the response
        response = model.generate_content(
            contents=[prompt, image_part]
        )
        
        return response.text
    
    except Exception as e:
        error_msg = str(e)
        if "API key" in error_msg:
            raise Exception("Invalid Google Gemini API key. Please check your API key settings and try again.")
        elif "safety" in error_msg.lower():
            raise Exception(f"The image analysis request was blocked due to safety concerns: {error_msg}")
        else:
            raise Exception(f"Error analyzing image with Google Gemini: {error_msg}")

def get_embedding(text, api_key=None, model_version=None):
    """
    Get an embedding vector for the given text using Google's embedding model.
    
    Args:
        text (str): The text to embed.
        api_key (str, optional): Google API key.
        model_version (str, optional): Not used for embeddings, but included for API consistency.
        
    Returns:
        list: The embedding vector.
    """
    try:
        # Configure the client
        get_gemini_client(api_key)
        
        # Use embedding model
        embedding_model = "embedding-001"  # Fixed model for embeddings
        
        # Get the embedding model
        model = genai.GenerativeModel(embedding_model)
        
        # Get embedding
        response = model.embed_content(
            content=text,
            task_type="retrieval_document"  # Best for RAG
        )
        
        # Return embedding as a list
        return response.embedding
    
    except Exception as e:
        error_msg = str(e)
        if "API key" in error_msg:
            raise Exception(f"Invalid Google API key for embedding: {error_msg}")
        elif "quota" in error_msg.lower() or "rate" in error_msg.lower():
            raise Exception(f"Google API quota or rate limit reached for embedding: {error_msg}")
        else:
            raise Exception(f"Error getting embedding from Google API: {error_msg}")
