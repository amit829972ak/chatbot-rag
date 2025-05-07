import os
import io
import base64
from PIL import Image
import anthropic
from anthropic import Anthropic

# Anthropic model options
ANTHROPIC_MODELS = {
    "claude-3-5-sonnet-20241022": {
        "name": "Claude 3.5 Sonnet",
        "description": "Latest and most capable model - best for multimodal tasks and documents",
        "api_name": "claude-3-5-sonnet-20241022"
    },
    "claude-3-opus-20240229": {
        "name": "Claude 3 Opus",
        "description": "Most powerful reasoning and comprehension",
        "api_name": "claude-3-opus-20240229"
    },
    "claude-3-sonnet-20240229": {
        "name": "Claude 3 Sonnet",
        "description": "Excellent balance of intelligence and speed",
        "api_name": "claude-3-sonnet-20240229"
    },
    "claude-3-haiku-20240307": {
        "name": "Claude 3 Haiku",
        "description": "Fastest and most compact model",
        "api_name": "claude-3-haiku-20240307"
    }
}

# Default model (excellent for CSV, TSV, XLSX, PDF, and images)
DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
# The newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024

def get_anthropic_client(api_key=None):
    """
    Get an Anthropic client with the provided API key.
    
    Args:
        api_key (str, optional): Anthropic API key
        
    Returns:
        Anthropic: Anthropic client
        
    Raises:
        Exception: If API key is invalid or missing
    """
    if not api_key:
        raise Exception("No API key provided. Please provide an Anthropic API key.")
    
    try:
        # Create the client
        client = Anthropic(api_key=api_key)
        
        # Simple validation of API key format
        if len(api_key) < 20:
            raise Exception("API key appears to be too short. Anthropic API keys are typically longer.")
        
        return client
    
    except Exception as e:
        error_msg = str(e).lower()
        if "invalid" in error_msg and "api" in error_msg:
            raise Exception("Invalid API key. Please check your Anthropic API key and try again.")
        elif "rate" in error_msg or "quota" in error_msg or "limit" in error_msg:
            raise Exception("API rate limit reached. Please try again later or use a different API key.")
        else:
            raise Exception(f"Error configuring Anthropic API: {str(e)}")

def get_ai_response(prompt, system_prompt=None, context=None, api_key=None, model_version=None):
    """
    Get a response from the Anthropic API.
    
    Args:
        prompt (str): The user's prompt.
        system_prompt (str, optional): System instructions to guide the AI.
        context (list, optional): Previous conversation context.
        api_key (str, optional): Anthropic API key.
        model_version (str, optional): The specific model version to use.
        
    Returns:
        str: The AI's response.
        
    Raises:
        Exception: If there's an API error
    """
    try:
        # Get client
        client = get_anthropic_client(api_key)
        
        # Use the specified model version or fall back to default
        selected_model = model_version or DEFAULT_MODEL
        
        # Make sure the selected model is valid
        if selected_model not in ANTHROPIC_MODELS:
            selected_model = DEFAULT_MODEL
        
        # Get the API name for the selected model
        model_name = ANTHROPIC_MODELS[selected_model]["api_name"]
        
        # Prepare messages
        messages = []
        
        # Add context if available
        if context:
            for message in context:
                role = "user" if message["role"] == "user" else "assistant"
                messages.append({"role": role, "content": message["content"]})
        
        # Add system prompt if provided
        if system_prompt:
            # For Anthropic, system instructions are a separate parameter
            system_instruction = system_prompt
        else:
            system_instruction = "You are Claude, an AI assistant by Anthropic. You are helpful, harmless, and honest."
        
        # Add the current prompt
        messages.append({"role": "user", "content": prompt})
        
        # Create the message and get the response
        response = client.messages.create(
            model=model_name,
            system=system_instruction,
            messages=messages,
            max_tokens=2000,
        )
        
        return response.content[0].text
    
    except Exception as e:
        error_msg = str(e)
        if "API key" in error_msg:
            raise Exception("Invalid Anthropic API key. Please check your API key settings and try again.")
        elif "quota" in error_msg.lower() or "rate" in error_msg.lower():
            raise Exception(f"Anthropic API quota or rate limit reached: {error_msg}")
        else:
            raise Exception(f"Error from Anthropic API: {error_msg}")

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
    Analyze an image using Anthropic's vision capabilities.
    
    Args:
        image (PIL.Image): The image to analyze.
        api_key (str, optional): Anthropic API key.
        model_version (str, optional): The specific model version to use.
        
    Returns:
        str: Analysis of the image content.
    """
    try:
        # Get client
        client = get_anthropic_client(api_key)
        
        # Use the specified model version or fall back to default
        selected_model = model_version or DEFAULT_MODEL
        
        # Make sure the selected model is valid
        if selected_model not in ANTHROPIC_MODELS:
            selected_model = DEFAULT_MODEL
        
        # Get the API name for the selected model
        model_name = ANTHROPIC_MODELS[selected_model]["api_name"]
        
        # Convert image to bytes
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        image_bytes = buffered.getvalue()
        
        # Create the message with the image
        response = client.messages.create(
            model=model_name,
            system="You are an expert at analyzing images in detail. Identify objects, people, text, scenes, and other elements. Describe what you see and provide any relevant context or insights.",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.b64encode(image_bytes).decode(),
                            },
                        },
                        {
                            "type": "text",
                            "text": "Analyze this image in detail. Identify objects, people, text, scenes, and other elements. Describe what you see and provide any relevant context or insights."
                        }
                    ]
                }
            ],
            max_tokens=2000,
        )
        
        return response.content[0].text
    
    except Exception as e:
        error_msg = str(e)
        if "API key" in error_msg:
            raise Exception("Invalid Anthropic API key. Please check your API key settings and try again.")
        elif "quota" in error_msg.lower() or "rate" in error_msg.lower():
            raise Exception(f"Anthropic API quota or rate limit reached: {error_msg}")
        else:
            raise Exception(f"Error analyzing image: {error_msg}")

def get_embedding(text, api_key=None, model_version=None):
    """
    Get an embedding vector for the given text.
    
    Note: As of this implementation, Anthropic doesn't have a dedicated embeddings API
    through their standard client. This function uses a simple fallback.
    
    Args:
        text (str): The text to embed.
        api_key (str, optional): Anthropic API key.
        model_version (str, optional): The specific model version to use.
        
    Returns:
        list: A simulated embedding vector (using a different service or model).
    """
    try:
        # For Anthropic, we'll use the OpenAI embedding API as a fallback
        from utils.openai_utils import get_embedding as openai_get_embedding
        
        try:
            # Try to use OpenAI for embeddings
            return openai_get_embedding(text, api_key=None)
        except:
            # If OpenAI fails, use Google's embedding as another fallback
            from utils.gemini_utils import get_embedding as gemini_get_embedding
            return gemini_get_embedding(text, api_key=None)
            
    except Exception as e:
        # If all else fails, return a simple hash-based embedding (not for production use)
        import hashlib
        import numpy as np
        
        # Create a deterministic but simple embedding from the text hash
        hash_object = hashlib.sha256(text.encode())
        hash_digest = hash_object.digest()
        
        # Convert the hash to a normalized embedding of length 768
        embedding = np.zeros(768)
        for i, byte in enumerate(hash_digest):
            if i < 768:
                embedding[i] = (byte / 255.0) * 2 - 1
        
        return embedding.tolist()
