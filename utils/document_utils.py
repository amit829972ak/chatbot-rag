import io
import re
import pandas as pd
from typing import Tuple, Optional
from PIL import Image
import docx
import PyPDF2

def process_document(file, file_type: str) -> Tuple[str, Optional[pd.DataFrame]]:
    """
    Process various document types and extract their content.
    
    Args:
        file: The uploaded file object
        file_type (str): The file extension/type
        
    Returns:
        Tuple[str, Optional[pd.DataFrame]]: Tuple containing extracted text content and dataframe (if applicable)
    """
    file_type = file_type.lower()
    
    # CSV processing
    if file_type == '.csv':
        try:
            df = pd.read_csv(file)
            return convert_df_to_csv_string(df) + "\n\n" + convert_df_to_json_string(df), df
        except Exception as e:
            return f"Error processing CSV file: {str(e)}", None
    
    # TSV processing
    elif file_type == '.tsv':
        try:
            df = pd.read_csv(file, sep='\t')
            return convert_df_to_csv_string(df) + "\n\n" + convert_df_to_json_string(df), df
        except Exception as e:
            return f"Error processing TSV file: {str(e)}", None
    
    # Excel processing
    elif file_type == '.xlsx' or file_type == '.xls':
        try:
            # Read all sheets
            sheet_names = pd.ExcelFile(file).sheet_names
            dfs = {}
            content = ""
            
            for sheet in sheet_names:
                df = pd.read_excel(file, sheet_name=sheet)
                dfs[sheet] = df
                content += f"Sheet: {sheet}\n"
                content += convert_df_to_csv_string(df) + "\n\n"
            
            # Return the first sheet's dataframe for visualization purposes
            first_sheet_df = dfs[sheet_names[0]] if sheet_names else None
            return content, first_sheet_df
        except Exception as e:
            return f"Error processing Excel file: {str(e)}", None
    
    # PDF processing
    elif file_type == '.pdf':
        try:
            reader = PyPDF2.PdfReader(file)
            content = ""
            
            # Extract text from each page with page numbers
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    content += f"--- Page {i+1} ---\n{page_text}\n\n"
            
            return content, None
        except Exception as e:
            return f"Error processing PDF file: {str(e)}", None
    
    # Plain text processing
    elif file_type == '.txt':
        try:
            content = file.read().decode('utf-8')
            return content, None
        except UnicodeDecodeError:
            try:
                # Try another common encoding
                content = file.read().decode('latin-1')
                return content, None
            except Exception as e:
                return f"Error processing text file: {str(e)}", None
        except Exception as e:
            return f"Error processing text file: {str(e)}", None
    
    # DOCX processing
    elif file_type == '.docx':
        try:
            doc = docx.Document(file)
            content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return content, None
        except Exception as e:
            return f"Error processing Word document: {str(e)}", None
    
    # Unsupported file type
    else:
        return f"Unsupported file type: {file_type}", None

def get_file_extension(filename: str) -> str:
    """
    Get the lowercase file extension including the dot.
    
    Args:
        filename (str): The filename
        
    Returns:
        str: Lowercase file extension with dot
    """
    match = re.search(r'(\.[^.]+)$', filename.lower())
    return match.group(1) if match else ""

def convert_df_to_csv_string(df: pd.DataFrame) -> str:
    """
    Convert a DataFrame to a CSV string.
    
    Args:
        df (pd.DataFrame): DataFrame to convert
        
    Returns:
        str: CSV string representation
    """
    return df.to_csv(index=False)

def convert_df_to_json_string(df: pd.DataFrame) -> str:
    """
    Convert a DataFrame to a JSON string.
    
    Args:
        df (pd.DataFrame): DataFrame to convert
        
    Returns:
        str: JSON string representation
    """
    return df.to_json(orient='records', indent=2)

def get_document_summary(document_content: str) -> str:
    """
    Create a brief summary of the document content.
    
    Args:
        document_content (str): The document content
        
    Returns:
        str: A brief summary
    """
    # Simple truncation-based summary
    max_summary_length = 300
    if len(document_content) <= max_summary_length:
        return document_content
    
    # Truncate and add ellipsis
    summary = document_content[:max_summary_length].strip()
    return summary + "..."
