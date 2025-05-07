import os
import io
import base64
from openai import OpenAI

# OpenAI model options
OPENAI_MODELS = {
    "gpt-4o": {
        "name": "GPT-4o",
        "description": "Latest multimodal model with vision",
        "api_name": "gpt-4o"
    },
    "gpt-4-turbo": {
        "name": "GPT-4 Turbo",
        "description": "High performance, most capable model",
        "api_name": "gpt-4-turbo"
    },
    "gpt-3.5-turbo": {
        "name": "GPT-3.5 Turbo",
        "description": "Free tier model - fast & cost-effective",
        "api_name": "gpt-3.5-turbo"
    }
}

# Default model
DEFAULT_MODEL = "gpt-4o"
# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user

def get_openai_client(api_key=None):
    """
    Get or create an OpenAI client with the provided API key.
    If no key is provided, try to use the environment variable.
    
    Args:
        api_key (str, optional): OpenAI API key
        
    Returns:
        OpenAI: OpenAI client
    """
    if not api_key:
        raise Exception("No API key provided. Please provide an OpenAI API key.")
    
    try:
        # Create the client - simplified to fix the 'proxies' error
        client = OpenAI(api_key=api_key)
        
        # Simple validation of API key format
        if not api_key.startswith("sk-") or len(api_key) < 20:
            raise Exception("API key appears to be invalid. OpenAI API keys start with 'sk-' and are longer.")
        
        return client
    
    except Exception as e:
        error_msg = str(e).lower()
        if "invalid" in error_msg and "api" in error_msg:
            raise Exception("Invalid API key. Please check your OpenAI API key and try again.")
        elif "rate" in error_msg or "quota" in error_msg or "limit" in error_msg:
            raise Exception("API rate limit reached. Please try again later or use a different API key.")
        else:
            raise Exception(f"Error configuring OpenAI API: {str(e)}")

def get_ai_response(prompt, system_prompt=None, context=None, api_key=None, model_version=None):
    """
    Get a response from the OpenAI API.
    
    Args:
        prompt (str): The user's prompt.
        system_prompt (str, optional): System instructions to guide the AI.
        context (list, optional): Previous conversation context.
        api_key (str, optional): OpenAI API key.
        model_version (str, optional): The specific model version to use.
        
    Returns:
        str: The AI's response.
        
    Raises:
        Exception: If there's an API error (rate limit, authentication, etc.)
    """
    try:
        # Get client
        client = get_openai_client(api_key)
        
        # Use the specified model version or fall back to default
        selected_model = model_version or DEFAULT_MODEL
        
        # Make sure the selected model is valid
        if selected_model not in OPENAI_MODELS:
            selected_model = DEFAULT_MODEL
        
        # Get the API name for the selected model
        model_name = OPENAI_MODELS[selected_model]["api_name"]
        
        # Prepare messages
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append({"role": "system", "content": "You are a helpful assistant."})
        
        # Add context if available
        if context:
            for message in context:
                role = message["role"]
                content = message["content"]
                # Make sure the role is valid for OpenAI
                if role not in ["user", "assistant", "system"]:
                    role = "user" if role == "human" else "assistant"
                messages.append({"role": role, "content": content})
        
        # Add the current prompt
        messages.append({"role": "user", "content": prompt})
        
        # Get the chat response
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        error_msg = str(e)
        if "API key" in error_msg:
            raise Exception("Invalid OpenAI API key. Please check your API key settings and try again.")
        elif "429" in error_msg:
            raise Exception("OpenAI API rate limit exceeded or quota reached. Please try again later.")
        else:
            raise Exception(f"Error from OpenAI API: {error_msg}")

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
    Analyze an image using OpenAI's vision capabilities.
    
    Args:
        image (PIL.Image): The image to analyze.
        api_key (str, optional): OpenAI API key.
        model_version (str, optional): The specific model version to use.
        
    Returns:
        str: Analysis of the image content.
    """
    try:
        # Get client
        client = get_openai_client(api_key)
        
        # For image analysis, we need a vision-capable model
        # Ensure we're using GPT-4o which has vision capabilities
        model_name = "gpt-4o"
        
        # Convert image to base64
        base64_image = encode_image_to_base64(image)
        
        # Create the messages with the image
        messages = [
            {
                "role": "system",
                "content": "You are an expert at analyzing images in detail. Identify objects, people, text, scenes, and other elements. Describe what you see and provide any relevant context or insights."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this image in detail. Identify objects, people, text, scenes, and other elements. Describe what you see and provide any relevant context or insights."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        
        # Get the response
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=1000,
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        error_msg = str(e)
        if "API key" in error_msg:
            raise Exception("Invalid OpenAI API key. Please check your API key settings and try again.")
        elif "429" in error_msg:
            raise Exception("OpenAI API rate limit exceeded or quota reached. Please try again later.")
        else:
            raise Exception(f"Error analyzing image: {error_msg}")

def get_embedding(text, api_key=None, model_version=None):
    """
    Get an embedding vector for the given text.
    
    Args:
        text (str): The text to embed.
        api_key (str, optional): OpenAI API key.
        model_version (str, optional): Not used for embeddings, but included for API consistency.
        
    Returns:
        list: The embedding vector.
    """
    try:
        # Get client
        client = get_openai_client(api_key)
        
        # Use embedding model (ada is most cost effective)
        model = "text-embedding-3-small"
        
        # Get the embedding
        response = client.embeddings.create(
            model=model,
            input=text,
            encoding_format="float"
        )
        
        # Return embedding as a list
        return response.data[0].embedding
    
    except Exception as e:
        error_msg = str(e)
        if "API key" in error_msg:
            raise Exception(f"Invalid OpenAI API key for embedding: {error_msg}")
        elif "429" in error_msg:
            raise Exception("OpenAI API rate limit exceeded or quota reached for embeddings. Please try again later.")
        else:
            raise Exception(f"Error getting embedding from OpenAI API: {error_msg}")
