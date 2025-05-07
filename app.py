"""
Gradio app containing the UI and application logic for PowerPoint generation with PDF RAG support.
"""
import datetime
import logging
import os
import pathlib
import random
import tempfile
import shutil
from typing import List, Dict, Union, Any, Tuple, Optional
import fitz
import json5
import ollama
import gradio as gr
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever

from global_config import GlobalConfig
from helpers import llm_helper, pptx_helper, text_helper
from sentence_transformers import SentenceTransformer, util
import uuid
from PIL import Image
import io

# Add these new imports
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Google Drive configuration
SERVICE_ACCOUNT_FILE = os.path.join(os.path.dirname(__file__), "ggupload-456902-a47a222a2bea.json")
GDRIVE_FOLDER_ID = "1Z9W94KrIgASDficdth06WqoaqF0nlSus"  # Your Google Drive folder ID (optional)
ENABLE_GDRIVE_UPLOAD = False  # Default setting for Google Drive uploads

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set protobuf environment variable to avoid error messages for ChromaDB
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Define persistent directory for ChromaDB
PERSIST_DIRECTORY = os.path.join("data", "vectors")
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

load_dotenv()

# Global state variables
chat_history = []
is_refinement = False
download_file_path = None
pptx_template = "Basic"  # Default template
llm_provider_to_use = "mistral:v0.2"  # Default LLM
vector_db = None  # For storing document embeddings
pdf_images = []  # For storing extracted images
image_embeddings = []  # For storing image embeddings
image_embedding_model = None  # For storing the image embedding model
used_pdf_images = set()  # Set of image paths already used
used_pexels_images = set()  # Set of Pexels URLs already used


def load_strings() -> dict:
    """
    Load various strings to be displayed in the app.
    :return: The dictionary of strings.
    """
    with open(GlobalConfig.APP_STRINGS_FILE, 'r', encoding='utf-8') as in_file:
        return json5.loads(in_file.read())


def get_prompt_template(is_refinement_prompt: bool, with_context: bool = False) -> str:
    """
    Return a prompt template with optional PDF context integration.

    :param is_refinement_prompt: Whether this is the initial or refinement prompt.
    :param with_context: Whether to include PDF context in the prompt.
    :return: The prompt template as f-string.
    """
    if is_refinement_prompt:
        with open(GlobalConfig.REFINEMENT_PROMPT_TEMPLATE, 'r', encoding='utf-8') as in_file:
            template = in_file.read()
    else:
        with open(GlobalConfig.INITIAL_PROMPT_TEMPLATE, 'r', encoding='utf-8') as in_file:
            template = in_file.read()
    
    # If we have PDF context, add it to the template in a structured way
    if with_context:
        context_template = """
        Use the following information from the PDF document as a knowledge source:
        
        {context}
        
        Extract key points, data, and insights from this content, but create a well-structured 
        presentation that extends beyond just these facts. Use this information to enhance 
        your response while still addressing the core request.
        """
        
        # Insert the context instructions at an appropriate point in the template
        if "create a PowerPoint presentation" in template.lower():
            # Insert before the main instructions
            parts = template.split("create a PowerPoint presentation", 1)
            template = parts[0] + context_template + "\n\ncreate a PowerPoint presentation" + parts[1]
        else:
            # Fallback: append to the beginning
            template = context_template + "\n\n" + template
    
    return template


def are_all_inputs_valid(user_prompt: str, selected_provider: str, selected_model: str) -> tuple[bool, str]:
    """
    Validate user input and LLM selection.

    :param user_prompt: The prompt.
    :param selected_provider: The LLM provider.
    :param selected_model: Name of the model.
    :return: Tuple of (is_valid, error_message)
    """
    if not text_helper.is_valid_prompt(user_prompt):
        return False, ('Not enough information provided! Please be a little more descriptive and '
                      'type a few words with a few characters :)')

    if not selected_provider or not selected_model:
        return False, 'No valid LLM provider and/or model name found!'

    return True, ""


def handle_error(error_msg: str, should_log: bool = True) -> str:
    """
    Log an error message.

    :param error_msg: The error message to be displayed.
    :param should_log: If `True`, log the message.
    :return: The error message.
    """
    if should_log:
        logger.error(error_msg)
    return error_msg


def extract_and_embed_images(pdf_path):
    """
    Extract images from PDF and create embeddings
    
    :param pdf_path: Path to the PDF file
    :return: List of extracted images with metadata
    """
    global pdf_images, image_embeddings, image_embedding_model
    
    # Initialize the CLIP model for image embeddings if not already loaded
    if image_embedding_model is None:
        try:
            # Load a multimodal model that can handle images and text
            image_embedding_model = SentenceTransformer('clip-ViT-B-32')
            logger.info("Loaded CLIP embedding model for images")
        except Exception as e:
            logger.error(f"Failed to load image embedding model: {e}")
            return []
    
    # Create directory for extracted images
    img_dir = os.path.join(tempfile.gettempdir(), f"pdf_images_{uuid.uuid4().hex}")
    os.makedirs(img_dir, exist_ok=True)
    logger.info(f"Created temp directory for images: {img_dir}")
    
    try:
        # Try to import PyMuPDF
        # try:
            # import fitz  # PyMuPDF
        # except ImportError:
        #     logger.error("PyMuPDF not installed. Please run: pip install pymupdf")
        #     return []
            
        # # Try to import PIL
        # try:
        #     from PIL import Image
        # except ImportError:
        #     logger.error("Pillow not installed. Please run: pip install pillow")
        #     return []
        
        # Open the PDF
        doc = fitz.open(pdf_path)
        image_count = 0
        extracted_images = []
        
        # Iterate through pages and extract images
        for page_num, page in enumerate(doc):
            image_list = page.get_images(full=True)
            
            # Process each image on the page
            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Generate a unique filename
                img_filename = f"page{page_num+1}_img{img_idx+1}.{image_ext}"
                img_path = os.path.join(img_dir, img_filename)
                
                # Save the image
                with open(img_path, "wb") as img_file:
                    img_file.write(image_bytes)
                
                # Create image object for embedding
                try:
                    # Open with PIL for processing
                    pil_img = Image.open(io.BytesIO(image_bytes))
                    
                    # Skip very small images (likely icons, etc.)
                    if pil_img.width < 100 or pil_img.height < 100:
                        continue
                    
                    # Create embedding for the image
                    img_embedding = image_embedding_model.encode(pil_img)
                    
                    # Store image with metadata
                    extracted_images.append({
                        "path": img_path,
                        "page": page_num + 1,
                        "image_bytes": image_bytes,
                        "embedding": img_embedding,
                        "width": pil_img.width,
                        "height": pil_img.height,
                        "aspect_ratio": pil_img.width / pil_img.height
                    })
                    
                    image_count += 1
                except Exception as e:
                    logger.warning(f"Failed to process image {img_path}: {e}")
        
        # Store images globally
        pdf_images = extracted_images
        logger.info(f"Extracted {image_count} images from PDF")
        return extracted_images
        
    except Exception as e:
        logger.error(f"Error extracting images from PDF: {e}")
        return []
    

def search_pdf_images(query, top_k=3):
    """
    Search for relevant images in the extracted PDF images based on text query
    
    :param query: Text query to search for
    :param top_k: Number of top results to return
    :return: List of image paths and metadata for most relevant images
    """
    global pdf_images, image_embedding_model
    
    if not pdf_images or not image_embedding_model:
        logger.warning("No PDF images available or embedding model not loaded")
        return None
    
    try:
        # Create text embedding for the query
        text_embedding = image_embedding_model.encode(query)
        
        # Calculate similarity scores
        similarities = []
        for idx, img_data in enumerate(pdf_images):
            similarity = util.cos_sim(text_embedding, img_data["embedding"])[0][0].item()
            similarities.append((idx, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        results = []
        for idx, score in similarities[:top_k]:
            img_data = pdf_images[idx].copy()
            img_data["score"] = score
            # Don't include embedding in results to keep response small
            del img_data["embedding"]
            results.append(img_data)
            
        logger.info(f"Found {len(results)} relevant images for query: {query}")
        return results
        
    except Exception as e:
        logger.error(f"Error searching PDF images: {e}")
        return None


def get_image_for_slide(keywords, use_pdf_images=True):
    """
    Get the best image for a slide based on keywords.
    First tries PDF images if available, then falls back to Pexels.
    Avoids reusing images that have already been selected.
    
    :param keywords: Keywords to search for images
    :param use_pdf_images: Whether to prioritize PDF images
    :return: Image data dictionary or None
    """
    global pdf_images, used_pdf_images, used_pexels_images
    
    if not keywords or isinstance(keywords, str) and not keywords.strip():
        logger.warning("Empty keywords for image search")
        return None
    
    logger.info(f"Searching image for keywords: {keywords}")
        
    # First try PDF images if requested and available
    if use_pdf_images and pdf_images:
        # Get more results than needed to allow for filtering out used ones
        pdf_results = search_pdf_images(keywords, top_k=5)  # Increase from 1 to 5
        
        if pdf_results and len(pdf_results) > 0:
            # Try to find an unused image first
            unused_results = [img for img in pdf_results if img["path"] not in used_pdf_images]
            
            if unused_results:
                # Pick the highest-ranking unused image
                selected_img = unused_results[0]
                logger.info(f"Using unused PDF image for keywords: {keywords}")
                used_pdf_images.add(selected_img["path"])  # Mark as used
                return selected_img
            elif pdf_results:
                # All similar images used, pick the best one but log it
                logger.info(f"All similar PDF images already used, reusing for: {keywords}")
                selected_img = pdf_results[0]
                return selected_img
    
    # Fall back to Pexels API
    try:
        from helpers import image_search as ims
        
        # Try up to 3 times to get an unused image
        for attempt in range(3):
            photo_url, page_url = ims.get_photo_url_from_api_response(
                ims.search_pexels(query=keywords, size='medium')
            )
            
            # If we got a result and it's not been used before
            if photo_url and photo_url not in used_pexels_images:
                logger.info(f"Using new Pexels image for keywords: {keywords}")
                image_bytes = ims.get_image_from_url(photo_url)
                if not image_bytes:
                    logger.warning(f"Failed to download image from {photo_url}")
                    continue
                
                used_pexels_images.add(photo_url)  # Mark as used
                return {
                    "source": "pexels",
                    "url": photo_url,
                    "page_url": page_url,
                    "image_bytes": image_bytes
                }
        
        # If we still couldn't get an unused image, try one more time and accept a used one
        photo_url, page_url = ims.get_photo_url_from_api_response(
            ims.search_pexels(query=keywords, size='medium')
        )
        
        if photo_url:
            logger.info(f"Reusing Pexels image for keywords: {keywords}")
            image_bytes = ims.get_image_from_url(photo_url)
            if image_bytes:
                return {
                    "source": "pexels",
                    "url": photo_url,
                    "page_url": page_url,
                    "image_bytes": image_bytes
                }
    except Exception as e:
        logger.error(f"Error fetching Pexels image: {e}")
        
    logger.warning(f"No image found for keywords: {keywords}")
    return None


def create_vector_db(file_upload) -> Optional[Chroma]:
    """
    Create a vector database from an uploaded PDF file with timestamped filename.
    Also extracts and processes images from the PDF.
    """
    if not file_upload:
        return None
        
    logger.info(f"Creating vector DB from file upload")
    temp_dir = tempfile.mkdtemp()

    try:
        # Generate a timestamp for the filename
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"uploaded_{timestamp_str}.pdf"
        
        path = os.path.join(temp_dir, pdf_filename)
        with open(path, "wb") as f:
            f.write(file_upload)
            logger.info(f"File saved to temporary path: {path}")
        
        # Extract and embed images from PDF
        logger.info("Extracting images from PDF...")
        extract_and_embed_images(path)
        
        # Use PyMuPDFLoader instead of PyPDFLoader
        loader = PyMuPDFLoader(path)
        data = loader.load()
        logger.info(f"Loaded {len(data)} pages from PDF")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
        chunks = text_splitter.split_documents(data)
        logger.info(f"Document split into {len(chunks)} chunks")

        if len(chunks) == 0:
            logger.warning("No chunks extracted from PDF")
            return None
            
        # Create embeddings with persistent storage
        # Choose an embedding model that works well with text
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY,
            collection_name=f"pdf_{timestamp_str}"
        )

        logger.info("Vector DB created with persistent storage")
        return vector_db
        
    except Exception as e:
        logger.error(f"Error creating vector DB: {e}")
        return None
    finally:
        # Don't delete temp_dir as we need the images
        logger.info(f"Keeping temporary directory {temp_dir} for image access")


def get_pdf_context(question: str, vector_db: Chroma, model: str) -> str:
    """
    Retrieve and format relevant context from PDF based on the question.
    
    :param question: The user's question
    :param vector_db: The vector database containing the PDF content
    :param model: The LLM model to use
    :return: Relevant context from the PDF
    """
    if not vector_db:
        return ""
    
    try:
        progress = gr.Progress()
        progress(0.2, desc="Analyzing PDF content...")
        
        # Set up the LLM for query generation
        llm = ChatOllama(model=model, temperature=0.2)
        
        # Improved query prompt for better retrieval
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""Generate 3 search queries to find information in a document that would help create 
            a comprehensive presentation about: {question}
            
            The queries should cover different aspects of the topic and help retrieve essential facts, 
            figures, and concepts needed for a presentation. Focus on retrieving factual information
            rather than opinions or general knowledge.
            
            Format: Return only the 3 search queries, one per line.
            """,
        )
        
        # Set up retriever with multi-query for better coverage
        retriever = MultiQueryRetriever.from_llm(
            vector_db.as_retriever(search_kwargs={"k": 4}), 
            llm,
            prompt=QUERY_PROMPT
        )
        
        progress(0.4, desc="Finding relevant information...")
        
        # Get context documents
        documents = retriever.get_relevant_documents(question)
        if not documents:
            return ""
        
        # Format the retrieved content for better integration with the prompt
        progress(0.7, desc="Organizing document insights...")
        
        formatted_contexts = []
        for i, doc in enumerate(documents):
            # Extract page number if available
            page_info = f"Page {doc.metadata.get('page', i+1)}" if hasattr(doc, 'metadata') else f"Section {i+1}"
            
            # Clean and format the content
            content = doc.page_content.strip()
            if len(content) > 1000:
                content = content[:997] + "..."
                
            formatted_contexts.append(f"--- {page_info} ---\n{content}")
        
        # Join all contexts with clear separation
        context = "\n\n".join(formatted_contexts)
        progress(1.0, desc="Document analysis complete")
        
        logger.info(f"Retrieved {len(documents)} relevant document sections")
        return context
        
    except Exception as e:
        logger.error(f"Error retrieving PDF context: {e}")
        return ""


def generate_slide_deck(json_str: str) -> Union[pathlib.Path, None]:
    """
    Create a slide deck and return the file path with a timestamp.
    """
    global download_file_path
    
    # Generate a timestamp for the filename
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        parsed_data = json5.loads(json_str)
    except ValueError:
        handle_error('Encountered error while parsing JSON...will fix it and retry')
        try:
            parsed_data = json5.loads(text_helper.fix_malformed_json(json_str))
        except ValueError:
            handle_error(
                'Encountered an error again while fixing JSON...'
                'the slide deck cannot be created, unfortunately ☹'
                '\nPlease try again later.'
            )
            return None
    except RecursionError:
        handle_error(
            'Encountered a recursion error while parsing JSON...'
            'the slide deck cannot be created, unfortunately ☹'
            '\nPlease try again later.'
        )
        return None
    except Exception as ex:
        handle_error(
            f'Encountered an error while parsing JSON: {str(ex)}...'
            'the slide deck cannot be created, unfortunately ☹'
            '\nPlease try again later.'
        )
        return None

    if download_file_path:
        path = pathlib.Path(download_file_path)
    else:
        # Add timestamp to file name
        file_name = f"presentation_{timestamp_str}.pptx"
        temp_dir = tempfile.mkdtemp()
        path = pathlib.Path(os.path.join(temp_dir, file_name))
        download_file_path = str(path)

    try:
        logger.debug(f'Creating PPTX file: {download_file_path}...')
        pptx_helper.generate_powerpoint_presentation(
            parsed_data,
            slides_template=pptx_template,
            output_file_path=path,
            pdf_image_getter=get_image_for_slide
        )
        
        # Verify the file was created and has content
        if os.path.exists(path) and os.path.getsize(path) > 0:
            logger.info(f"Successfully created presentation at {path}")
        else:
            logger.error(f"Failed to create presentation at {path}")
            return None
            
    except Exception as ex:
        error_msg = APP_TEXT['content_generation_error']
        logger.error(f'Caught a generic exception: {str(ex)}')
        download_file_path = None  # Reset on error
        return None

    return path


def upload_to_gdrive(file_path: str) -> Optional[Tuple[str, str]]:
    """
    Upload a file to Google Drive using service account credentials and make it accessible.
    
    :param file_path: Path to the file to upload
    :return: Tuple of (file_id, view_link) or None on error
    """
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        logger.warning("Google Drive credentials missing")
        return None
    
    try:
        # Set up credentials for service account
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE,
            scopes=["https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
        )
        
        # Build the Drive API client
        drive_service = build('drive', 'v3', credentials=creds)
        
        # File metadata - without folder ID to upload to root
        file_metadata = {
            'name': os.path.basename(file_path),
        }
        
        # Determine mime type based on file extension
        file_extension = os.path.splitext(file_path)[1].lower()
        mime_type = "application/pdf"  # Default
        if file_extension == ".pptx":
            mime_type = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        
        # Upload file
        media = MediaFileUpload(
            file_path,
            mimetype=mime_type,
            resumable=True
        )
        
        file = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id,webViewLink'
        ).execute()
        
        file_id = file.get('id')
        
        # Make the file accessible to anyone with the link as an editor
        drive_service.permissions().create(
            fileId=file_id,
            body={
                'type': 'anyone',
                'role': 'writer'  # Changed from 'reader' to 'writer'
            }
        ).execute()
        
        # Get the updated file with webViewLink
        updated_file = drive_service.files().get(
            fileId=file_id, 
            fields='webViewLink'
        ).execute()
        
        view_link = updated_file.get('webViewLink', '')
        
        logger.info(f"File uploaded to Google Drive with ID: {file_id}")
        logger.info(f"File can be viewed and edited at: {view_link}")
        
        return file_id, view_link
    
    except Exception as e:
        logger.error(f"Error uploading to Google Drive: {e}")
        return None


def get_user_messages(history) -> List[str]:
    """
    Get a list of user messages submitted until now.
    :param history: The chat history in Gradio format
    :return: The list of user messages.
    """
    return [msg[0] for msg in history if msg[0] is not None]


def get_last_response(history) -> str:
    """
    Get the last response generated by AI.
    :param history: The chat history in Gradio format
    :return: The response text.
    """
    for msg in reversed(history):
        if msg[1] is not None:
            return msg[1]
    return ""


def is_it_refinement(history) -> bool:
    """
    Whether it is the initial prompt or a refinement.
    :param history: The chat history in Gradio format
    :return: True if it is a refinement; False otherwise.
    """
    global is_refinement
    
    if is_refinement:
        return True
    
    if len(history) >= 1:
        is_refinement = True
        return True
    
    return False


def process_message(message: str, history: List, use_pdf_context: bool = False) -> tuple[str, List]:
    """
    Process user message and generate a response with optional PDF context.
    
    :param message: The user's message
    :param history: The chat history
    :param use_pdf_context: Whether to use PDF context in the response
    :return: The empty message (to clear input) and updated history
    """
    global is_refinement, download_file_path, vector_db, used_pdf_images, used_pexels_images
    
    # Reset image tracking when starting a new presentation
    used_pdf_images = set()
    used_pexels_images = set()
    
    if not message.strip():
        return "", history
    
    provider, llm_name = llm_helper.get_provider_model(
        llm_provider_to_use,
        use_ollama=True
    )
    
    valid, error_msg = are_all_inputs_valid(message, provider, llm_name)
    if not valid:
        history.append((message, error_msg))
        return "", history
    
    logger.info(
        f'User input: {message} | #characters: {len(message)} | LLM: {llm_name} | Use PDF: {use_pdf_context}'
    )
    
    # Add user message to history without showing response yet
    history.append((message, None))
    
    # Retrieve context from PDF if available and requested
    pdf_context = ""
    if use_pdf_context and vector_db:
        progress = gr.Progress()
        progress(0.1, desc="Analyzing PDF for relevant information...")
        pdf_context = get_pdf_context(message, vector_db, llm_name)
        
        if pdf_context:
            logger.info(f"PDF context retrieved ({len(pdf_context)} characters)")
        else:
            logger.warning("No relevant PDF context found")
    
    # Prepare the prompt with context if available
    prompt_template = ChatPromptTemplate.from_template(
        get_prompt_template(
            is_refinement_prompt=is_it_refinement(history),
            with_context=bool(pdf_context)
        )
    )
    
    # Format the template appropriately based on refinement status and context
    if is_it_refinement(history):
        user_messages = get_user_messages(history)
        list_of_msgs = [
            f'{idx + 1}. {msg}' for idx, msg in enumerate(user_messages)
        ]
        
        # Format with context if available
        if pdf_context:
            formatted_template = prompt_template.format(
                **{
                    'instructions': '\n'.join(list_of_msgs),
                    'previous_content': get_last_response(history),
                    'context': pdf_context
                }
            )
        else:
            formatted_template = prompt_template.format(
                **{
                    'instructions': '\n'.join(list_of_msgs),
                    'previous_content': get_last_response(history)
                }
            )
    else:
        if pdf_context:
            formatted_template = prompt_template.format(
                **{'question': message, 'context': pdf_context}
            )
        else:
            formatted_template = prompt_template.format(
                **{'question': message}
            )
    
    # Call the LLM
    response = ""
    progress = gr.Progress()
    progress(0, desc="Preparing to call LLM...")
    
    try:
        llm = llm_helper.get_langchain_llm(
            provider=provider,
            model=llm_name,
            max_new_tokens=4096
        )
        
        if not llm:
            error_msg = (
                'Failed to create an LLM instance! Make sure that you have provided '
                'the correct model name.'
            )
            handle_error(error_msg, False)
            history[-1] = (message, error_msg)
            return "", history
        
        # Stream the response but don't show it in the UI yet
        progress(0.1, desc="Generating content...")
        
        # Get the full response without streaming to the UI
        for chunk in llm.stream(formatted_template):
            if isinstance(chunk, str):
                response += chunk
            else:
                response += chunk.content  # AIMessageChunk
            
            # Update progress but don't update the UI
            progress(
                min(len(response) / 4096, 0.95),
                desc="Processing content... Please wait..."
            )
            
    except ollama.ResponseError:
        error_msg = (
            f'The model `{llm_name}` is unavailable with Ollama on your system. '
            f'Make sure that you have provided the correct LLM name or pull it using '
            f'`ollama pull {llm_name}`. View LLMs available locally by running `ollama list`.'
        )
        handle_error(error_msg)
        history[-1] = (message, error_msg)
        download_file_path = None  # Reset the download path
        return "", history
    except Exception as ex:
        error_msg = (
            f'An unexpected error occurred while generating the content: {str(ex)}\n\n'
            'Please try again later, possibly with different inputs.'
        )
        handle_error(error_msg)
        history[-1] = (message, error_msg)
        return "", history
    
    # Clean up the JSON response
    response = text_helper.get_clean_json(response)
    logger.info(f'Cleaned JSON length: {len(response)}')
    
    # Generate the slide deck
    progress(0.95, desc="Creating presentation... Please wait...")
    slide_path = generate_slide_deck(response)
    progress(1.0, desc="Done!")
    
    # Only update the UI with the final response
    history[-1] = (message, response)
    
    return "", history


def check_pdf_similarity(pdf_bytes):
    """
    Check if a similar PDF already exists in the database by comparing content.
    
    :param pdf_bytes: The PDF file content as bytes
    :return: Tuple of (similarity_found, collection_name, similarity_score)
    """
    try:
        # Extract a sample of text from the PDF
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_file.write(pdf_bytes)
        temp_file.close()
        
        logger.info(f"Starting similarity check for uploaded PDF")
        
        # Use PyMuPDFLoader to extract text
        loader = PyMuPDFLoader(temp_file.name)
        data = loader.load()
        
        # Clean up the temporary file
        os.unlink(temp_file.name)
        
        # If no text was extracted, return not found
        if not data or len(data) == 0:
            logger.warning("No text content extracted from the PDF")
            return False, None, 0
        
        # Take a more substantial sample - up to 5 pages instead of 3
        sample_text = " ".join([page.page_content for page in data[:min(5, len(data))]])
        
        # If the sample is too short, return not found
        if len(sample_text) < 100:
            logger.warning(f"PDF text sample too short ({len(sample_text)} chars)")
            return False, None, 0
        
        # Get a list of all collections by scanning directory
        collections = []
        if os.path.exists(PERSIST_DIRECTORY):
            # Look for collection directories which contain a chroma.sqlite3 file
            for item in os.listdir(PERSIST_DIRECTORY):
                item_path = os.path.join(PERSIST_DIRECTORY, item)
                if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "chroma.sqlite3")):
                    if item.startswith("pdf_"):
                        collections.append(item)
        
        logger.info(f"Found {len(collections)} collections to check for similarity")
        
        # Set up the embedding function - must match what's used in create_vector_db
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        # Check each collection for similar content
        highest_similarity = 0
        most_similar_collection = None
        
        for collection_name in collections:
            if not collection_name.startswith("pdf_"):
                continue
                
            try:
                # Load the collection
                db = Chroma(
                    collection_name=collection_name,
                    persist_directory=PERSIST_DIRECTORY,
                    embedding_function=embeddings
                )
                
                # Search for similar documents with more samples
                results = db.similarity_search_with_score(sample_text, k=3)  # Get more results
                
                if results and len(results) > 0:
                    # Average the top 3 scores (or fewer if less available)
                    avg_similarity = 0
                    for doc, score in results:
                        similarity = 1.0 - score  # Convert distance to similarity
                        avg_similarity += similarity
                    
                    avg_similarity /= len(results)
                    
                    logger.info(f"Collection {collection_name} similarity: {avg_similarity:.4f}")
                    
                    if avg_similarity > highest_similarity:
                        highest_similarity = avg_similarity
                        most_similar_collection = collection_name
            except Exception as e:
                logger.warning(f"Error checking collection {collection_name}: {e}")
                continue
        
        # Lower the threshold from 0.85 to 0.80 for better detection
        threshold = 0.80
        if highest_similarity > threshold:  
            logger.info(f"Similar PDF found! Collection: {most_similar_collection}, Similarity: {highest_similarity:.4f}")
            return True, most_similar_collection, highest_similarity
        
        logger.info(f"No similar PDF found above threshold ({threshold}). Highest similarity: {highest_similarity:.4f}")
        return False, None, highest_similarity
            
    except Exception as e:
        logger.error(f"Error checking PDF similarity: {e}")
        return False, None, 0


# Add this import at the top
import hashlib

# Add this function to check for exact duplicates
def get_pdf_hash(pdf_bytes):
    """
    Generate a hash for the PDF content to identify exact duplicates
    
    :param pdf_bytes: Raw PDF data
    :return: SHA-256 hash of the PDF content
    """
    return hashlib.sha256(pdf_bytes).hexdigest()

# Add a global dictionary to store PDF hashes
pdf_hashes = {}  # Maps hash -> collection_name
    
def template_changed(template_name: str) -> None:
    """
    Handle template selection change.
    
    :param template_name: The selected template name
    """
    global pptx_template
    pptx_template = template_name
    logger.info(f"Template changed to: {template_name}")


def llm_changed(model_name: str) -> None:
    """
    Handle LLM selection change.
    
    :param model_name: The selected LLM name
    """
    global llm_provider_to_use
    llm_provider_to_use = model_name
    logger.info(f"LLM changed to: {model_name}")


def upload_pdf(file_obj):
    """
    Handle PDF upload and vector DB creation.
    
    :param file_obj: Uploaded file object or bytes
    :return: Status message about success or failure with similarity info
    """
    global vector_db, pdf_hashes
    
    if file_obj is None:
        return "No file uploaded"
    
    try:
        # Calculate hash but with additional error handling
        try:
            pdf_hash = get_pdf_hash(file_obj)
            logger.info(f"PDF hash: {pdf_hash}")
            
            # Check if this exact PDF has been processed before
            if pdf_hash in pdf_hashes:
                collection_name = pdf_hashes[pdf_hash]
                logger.info(f"Exact duplicate PDF detected! Using collection: {collection_name}")
                
                try:
                    # Load the existing vector DB with better error handling
                    vector_db = Chroma(
                        collection_name=collection_name,
                        persist_directory=PERSIST_DIRECTORY,
                        embedding_function=OllamaEmbeddings(model="nomic-embed-text")
                    )
                    return f"📄 Exact duplicate PDF detected! Using existing embeddings from collection: {collection_name}"
                except Exception as e:
                    logger.warning(f"Failed to load existing collection {collection_name}: {e}")
                    # Fall through to reprocess
            
            # Try content similarity check with better error handling
            try:
                similar_found, collection_name, similarity = check_pdf_similarity(file_obj)
                
                if similar_found and collection_name:
                    try:
                        # Load the existing vector DB
                        vector_db = Chroma(
                            collection_name=collection_name,
                            persist_directory=PERSIST_DIRECTORY,
                            embedding_function=OllamaEmbeddings(model="nomic-embed-text")
                        )
                        
                        # Store the hash for future reference
                        pdf_hashes[pdf_hash] = collection_name
                        
                        # Format the similarity as a percentage
                        similarity_pct = f"{similarity * 100:.1f}%"
                        
                        logger.info(f"Using existing PDF embeddings for similar document: {collection_name}")
                        return f"📄 Similar PDF detected ({similarity_pct} match)! Using existing embeddings from collection: {collection_name}"
                    except Exception as e:
                        logger.warning(f"Failed to load similar collection {collection_name}: {e}")
                        # Fall through to reprocess
            except Exception as e:
                logger.warning(f"Error in similarity check: {e}")
                # Fall through to reprocess
        except Exception as e:
            logger.warning(f"Error in hash calculation: {e}")
            # Fall through to reprocess
        
        # Process the new PDF
        logger.info("Processing PDF as new document")
        vector_db = create_vector_db(file_obj)
        
        if vector_db:
            try:
                # Get collection name safely
                collection_name = getattr(vector_db._collection, 'name', None)
                if collection_name:
                    # Store the hash for the newly created collection
                    pdf_hashes[pdf_hash] = collection_name
            except Exception as e:
                logger.warning(f"Could not store PDF hash: {e}")
                
            return "✅ Successfully processed new PDF"
        else:
            return "❌ Error processing PDF. Please try another file."
            
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        return f"❌ Error processing PDF: {str(e)}"


def download_presentation():
    """
    Return the presentation file for download with a timestamp in name.
    
    :return: Tuple of (file_path, file_name) for download
    """
    global download_file_path
    
    if download_file_path and os.path.exists(download_file_path):
        # Use a timestamp for the downloaded file if not already in the filename
        if "_202" not in download_file_path:
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"Presentation_{timestamp_str}.pptx"
        else:
            file_name = os.path.basename(download_file_path)
            
        return download_file_path, file_name
    return None, None


def upload_presentation_to_gdrive() -> str:
    """
    Upload the current presentation to Google Drive.
    
    :return: Status message about the upload with clickable link
    """
    global download_file_path
    
    if not download_file_path or not os.path.exists(download_file_path):
        return "No presentation file available for upload."
    
    try:
        result = upload_to_gdrive(download_file_path)
        if result:
            file_id, view_link = result
            # Return HTML with a clickable link
            return f"<div>Successfully uploaded to Google Drive!</div><div><a href='{view_link}' target='_blank' style='color: #4285F4; text-decoration: underline;'>Click here to open and edit</a></div>"
        else:
            return "Failed to upload to Google Drive. Check logs for details."
    except Exception as e:
        logger.error(f"Error in upload to Google Drive: {e}")
        return f"Error: {str(e)}"


def create_initial_message() -> List:
    """
    Create an initial greeting message for the chatbot.
    
    :return: Initial chat history with AI greeting
    """
    greeting = random.choice(APP_TEXT['ai_greetings'])
    return [(None, greeting)]


def is_download_ready(history: List) -> bool:
    """
    Check if the download is ready based on chat history and file existence.
    
    :param history: The chat history
    :return: True if download is ready, False otherwise
    """
    global download_file_path
    
    # Only show download when a file exists, has content, and there's a complete chat response
    return (history and len(history) > 0 and 
            history[-1][1] is not None and 
            download_file_path and 
            os.path.exists(download_file_path) and
            os.path.getsize(download_file_path) > 0)


def set_gdrive_upload(value: bool) -> None:
    """
    Toggle Google Drive upload setting.
    
    :param value: Whether to enable Google Drive uploads
    """
    global ENABLE_GDRIVE_UPLOAD
    ENABLE_GDRIVE_UPLOAD = value
    logger.info(f"Google Drive upload {'enabled' if value else 'disabled'}")


def get_vector_db_info():
    """
    Get formatted information about the current vector database.
    
    :return: Tuple of (content, visibility_update)
    """
    if not vector_db:
        return "", gr.update(visible=False)
        
    try:
        # Safely get collection info
        collection_name = "Unknown"
        count = "Unknown"
        
        if hasattr(vector_db, '_collection'):
            if hasattr(vector_db._collection, 'name'):
                collection_name = vector_db._collection.name
            if hasattr(vector_db._collection, 'count') and callable(vector_db._collection.count):
                try:
                    count = vector_db._collection.count()
                except:
                    pass
        
        info = "### PDF Collection Info\n"
        info += "- **Status**: Active\n"
        info += f"- **Collection**: {collection_name}\n"
        info += f"- **Document Count**: {count} chunks\n"
        
        return info, gr.update(visible=True)
    except Exception as e:
        logger.error(f"Error getting vector DB info: {e}")
        return "### PDF Collection Info\n- **Status**: Active (Error retrieving details)", gr.update(visible=True)


def create_ui() -> gr.Blocks:
    """
    Create the Gradio UI with improved layout and download handling.
    
    :return: Gradio Blocks interface
    """
    custom_css = """
    #download_file {
        width: 100%;
        margin-top: 8px;
    }
    """
    # Set up template selection
    templates = list(GlobalConfig.PPTX_TEMPLATE_FILES.keys())
    captions = [GlobalConfig.PPTX_TEMPLATE_FILES[x]['caption'] for x in templates]
    
    with gr.Blocks(title="Presentation Generator", theme=gr.themes.Soft(), css=custom_css) as demo:
        gr.Markdown(f"# {APP_TEXT['app_name']}")
        gr.Markdown(f"*{APP_TEXT['caption']}*")
        
        with gr.Row():
            # Left panel - Settings and controls
            with gr.Column(scale=1):
                # Settings section with a border
                with gr.Group(visible=True) as settings_group:
                    gr.Markdown("### Settings")
                    
                    # Template selection
                    template_dropdown = gr.Dropdown(
                        choices=templates,
                        value=pptx_template,
                        label="Select a presentation template:",
                        interactive=True
                    )
                    
                    # LLM selection
                    llm_input = gr.Textbox(
                        value=llm_provider_to_use,
                        label="Enter Ollama model name:",
                        info="Examples: mistral:v0.2, llama3:latest",
                        interactive=True
                    )

                    # Google Drive upload setting
                    gdrive_upload = gr.Checkbox(
                        label="Upload files to Google Drive",
                        value=ENABLE_GDRIVE_UPLOAD,
                        info="When enabled, PDF and presentations will be uploaded to Google Drive"
                    )
                
                # PDF upload section
                with gr.Group():
                    gr.Markdown("### PDF Reference (Optional)")
                    gr.Markdown("Upload a PDF to use as a knowledge source for your presentation")
                    
                    # PDF upload
                    with gr.Row():
                        pdf_file = gr.File(
                            label="Select PDF file",
                            file_types=[".pdf"],
                            type="binary"
                        )
                        pdf_process_btn = gr.Button("Process PDF", variant="primary")
                    
                    pdf_result = gr.Textbox(
                        label="PDF Status", 
                        interactive=False,
                        value="No PDF processed yet"
                    )
                    
                    pdf_info = gr.Markdown(
                        value="",
                        visible=False,
                        label="PDF Collection Info"
                    )
                    
                    use_pdf = gr.Checkbox(
                        label="Use PDF content for presentation generation",
                        value=False,
                        info="When enabled, information from the PDF will be used to enhance your presentation"
                    )
                
                # Usage info in collapsible section
                with gr.Accordion("Usage Instructions", open=False):
                    gr.Markdown(GlobalConfig.CHAT_USAGE_INSTRUCTIONS)
                
                # Result area (only visible after successful generation)
                with gr.Group(visible=False) as download_box:
                    gr.Markdown("### Your Presentation")
                    gr.Markdown("Your presentation is ready!")
                    
                    with gr.Row():
                        download_btn = gr.Button("⬇️ Download Presentation", variant="primary")
                        gdrive_upload_btn = gr.Button("📤 Upload to Google Drive", variant="secondary")
                    
                    # Status message for Google Drive upload
                    gdrive_status = gr.HTML(
                        value="",
                        label="Upload Status", 
                        visible=False
                    )
                    
                    # Download file component
                    download_file = gr.File(
                        label=None, 
                        visible=False, 
                        elem_id="download_file"
                    )
                
            # Right panel - Chat interface
            with gr.Column(scale=1):
                # Chat interface
                chatbot = gr.Chatbot(
                    value=create_initial_message(),
                    height=600,
                    show_copy_button=True
                )
                
                # Input area
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder=APP_TEXT['chat_placeholder'],
                        show_label=False
                    )
                    submit = gr.Button("Send", variant="primary")
                
                # Action buttons
                with gr.Row():
                    clear = gr.Button("Clear Chat")
        
        # Handle PDF upload and processing
        pdf_process_btn.click(
            upload_pdf,
            inputs=[pdf_file],
            outputs=[pdf_result]
        ).then(
            lambda: get_vector_db_info(),
            outputs=[pdf_info, pdf_info] 
        )
        
        # Event handlers for chat
        submit_event = submit.click(
            lambda msg, history, use_pdf: process_message(msg, history, use_pdf),
            inputs=[msg, chatbot, use_pdf],
            outputs=[msg, chatbot]
        )
        
        submit_event.then(
            lambda h: gr.update(visible=is_download_ready(h)),
            inputs=[chatbot],
            outputs=[download_box]
        )
        
        msg.submit(
            lambda msg, history, use_pdf: process_message(msg, history, use_pdf),
            inputs=[msg, chatbot, use_pdf],
            outputs=[msg, chatbot]
        ).then(
            lambda h: gr.update(visible=is_download_ready(h)),
            inputs=[chatbot],
            outputs=[download_box]
        )
        
        # Handle template selection
        template_dropdown.change(
            template_changed,
            inputs=[template_dropdown]
        )
        
        # Handle LLM selection
        llm_input.change(
            llm_changed,
            inputs=[llm_input]
        )

        # Handle Google Drive upload setting
        gdrive_upload.change(
            lambda val: set_gdrive_upload(val),
            inputs=[gdrive_upload]
        )
        
        # Setup download button
        download_btn.click(
            fn=download_presentation,
            outputs=[download_file, gr.Textbox(visible=False)]
        ).then(
            lambda: gr.update(visible=True),
            None,
            [download_file]
        )
        
        # Setup Google Drive upload button
        gdrive_upload_btn.click(
            fn=lambda: upload_presentation_to_gdrive(),
            outputs=[gdrive_status]
        ).then(
            lambda: gr.update(visible=True),
            None,
            [gdrive_status]
        )
        
        # Handle clear button
        clear.click(
            lambda: ([], ""),
            outputs=[chatbot, msg]
        ).then(
            lambda: gr.update(visible=False),
            None,
            [download_box]
        ).then(
            create_initial_message,
            outputs=[chatbot]
        )
    
    return demo


# Load app strings
APP_TEXT = load_strings()

# Create and launch the interface
if __name__ == "__main__":
    # demo = create_ui()
    # demo.launch(share=False, server_name="0.0.0.0", show_error=True)

    demo = create_ui()
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        debug=True,     
        show_error=True
    )