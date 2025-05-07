import io
from PIL import Image, ImageOps, ExifTags

def process_image(image):
    """
    Process and optimize an image for analysis.
    
    Args:
        image (PIL.Image): The image to process.
        
    Returns:
        PIL.Image: The processed image.
    """
    try:
        # Handle EXIF orientation (if present)
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            
            exif = dict(image._getexif().items())
            
            if exif[orientation] == 3:
                image = image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                image = image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                image = image.rotate(90, expand=True)
        except (AttributeError, KeyError, IndexError):
            # No EXIF data or no orientation tag, continue without rotation
            pass
        
        # Convert to RGB if image is in RGBA mode (has transparency)
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])  # Use alpha channel as mask
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize large images to reduce processing time and API costs
        max_dimension = 1280  # Max width or height
        width, height = image.size
        
        if width > max_dimension or height > max_dimension:
            # Calculate new dimensions while preserving aspect ratio
            if width > height:
                new_width = max_dimension
                new_height = int(height * (max_dimension / width))
            else:
                new_height = max_dimension
                new_width = int(width * (max_dimension / height))
            
            # Resize with high quality
            image = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Optimize quality vs. size for API transmission
        # Lower quality = smaller file size but potentially less detail
        quality = 85
        
        # Create a new in-memory buffer to compress the image
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        
        # Read the compressed image back into a PIL Image
        buffer.seek(0)
        processed_image = Image.open(buffer)
        
        return processed_image
    
    except Exception as e:
        # If processing fails, return the original image
        print(f"Image processing error: {str(e)}")
        return image

def convert_image_to_bytes(image):
    """
    Convert a PIL Image to bytes.
    
    Args:
        image (PIL.Image): The image to convert.
        
    Returns:
        bytes: The image as bytes.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()

def bytes_to_image(image_bytes):
    """
    Convert bytes to a PIL Image.
    
    Args:
        image_bytes (bytes): The image bytes.
        
    Returns:
        PIL.Image: The reconstructed image.
    """
    return Image.open(io.BytesIO(image_bytes))
