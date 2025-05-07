"""
FastAPI backend for PowerPoint generation with PDF RAG support.
"""
import datetime
import logging
import os
import pathlib
import random
import tempfile
import hashlib
import shutil
from typing import List, Dict, Union, Any, Tuple, Optional
import fitz
import json5
import ollama
import uuid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends, Request, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2AuthorizationCodeBearer
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from authlib.integrations.starlette_client import OAuth, OAuthError
from starlette.config import Config
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import RedirectResponse

from global_config import GlobalConfig
from helpers import llm_helper, pptx_helper, text_helper
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import io
import json
import re

# Google Drive integration
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

import hashlib
import shutil
from langchain.schema import Document as LangchainDocument
from helpers.doc_processor import DocProcessor
from sentence_transformers import SentenceTransformer


import pymysql
from db_handler import DatabaseHandler

# Initialize database handler
db = DatabaseHandler()

# Google Drive configuration
SERVICE_ACCOUNT_FILE = os.path.join(os.path.dirname(__file__), "ggupload-456902-a47a222a2bea.json")
GDRIVE_FOLDER_ID = "1Z9W94KrIgASDficdth06WqoaqF0nlSus"  # Your Google Drive folder ID (optional)
ENABLE_GDRIVE_UPLOAD = False  # Default setting for Google Drive uploads

# Load Google OAuth config
with open('client_secret_22849360846.json', 'r') as f:
    google_config = json.load(f)['web']

# Configure OAuth
config = Config(environ={
    'GOOGLE_CLIENT_ID': google_config['client_id'],
    'GOOGLE_CLIENT_SECRET': google_config['client_secret']
})

oauth = OAuth(config)
oauth.register(
    name='google',
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

# User dictionary to store authenticated users
authenticated_users = {}

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
# Note: In production, you would use a database or Redis for state management
chat_history = {}  # Now using a dict with session_id as key
is_refinement = {}  # Track refinement by session
download_file_path = {}  # Track download path by session
pptx_template = "Basic"  # Default template
llm_provider_to_use = "mistral:v0.2"  # Default LLM
vector_db = None  # For storing document embeddings
pdf_images = []  # For storing extracted images
image_embeddings = []  # For storing image embeddings
image_embedding_model = None  # For storing the image embedding model
used_pdf_images = set()  # Set of image paths already used
used_pexels_images = set()  # Set of Pexels URLs already used
pdf_hashes = {}  # Maps hash -> collection_name
openai_api_key = None  # OpenAI API key

# Initialize FastAPI app
app = FastAPI(
    title="Presentation Generator API", 
    description="API for generating presentations using LLMs and PDFs",
    version="1.0.0"
)

# Add CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add session middleware (add right after CORS middleware)
app.add_middleware(SessionMiddleware, secret_key="your-secret-key-change-this")

# Load app strings
def load_strings() -> dict:
    """
    Load various strings to be displayed in the app.
    :return: The dictionary of strings.
    """
    with open(GlobalConfig.APP_STRINGS_FILE, 'r', encoding='utf-8') as in_file:
        return json5.loads(in_file.read())

APP_TEXT = load_strings()

# Define Pydantic models for request/response
class SessionRequest(BaseModel):
    session_id: str = None

class MessageRequest(BaseModel):
    session_id: str
    message: str
    use_pdf_context: bool = False
    slide_count: int = 10  # Default value is 10 slides
    is_new_presentation: bool = False  # Add this flag to indicate a new presentation

class TemplateRequest(BaseModel):
    template_name: str

class LLMRequest(BaseModel):
    model_name: str
    api_key: Optional[str] = None
    provider: Optional[str] = None

class GDriveSettingRequest(BaseModel):
    enable: bool

class PresentationResponse(BaseModel):
    success: bool
    message: str
    download_url: Optional[str] = None
    history: Optional[List] = None

class PDFResponse(BaseModel):
    success: bool
    message: str
    collection_info: Optional[Dict[str, Any]] = None

class VectorDBInfoResponse(BaseModel):
    success: bool
    collection_name: Optional[str] = None
    count: Optional[int] = None
    visible: bool = False

class GDriveUploadResponse(BaseModel):
    success: bool
    message: str
    view_link: Optional[str] = None

# Session management
def get_session_id():
    """Generate a unique session ID"""
    return str(uuid.uuid4())

# Auth dependency for protected routes
async def get_current_user(request: Request):
    user = request.session.get('user')
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    return user

# Add authentication routes
@app.route('/auth/login')
async def login(request: Request):
    redirect_uri = request.url_for('auth_callback')
    return await oauth.google.authorize_redirect(request, redirect_uri)

@app.route('/auth/callback')
async def auth_callback(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
        user_info = token.get('userinfo')
        if user_info:
            # Store user info in session
            request.session['user'] = dict(user_info)
            user_id = user_info['sub']  # Google's unique identifier
            email = user_info['email']
            name = user_info.get('name', '')
            picture = user_info.get('picture', '')
            
            # Store user in database instead of memory
            db.save_user(
                user_id=user_id,
                email=email,
                name=name,
                picture=picture
            )
            
            # Redirect to frontend with auth success
            return RedirectResponse(url=f"http://localhost:5173?auth=success&userId={user_id}")
        return RedirectResponse(url="http://localhost:5173?auth=error")
    except OAuthError as e:
        logger.error(f"OAuth error: {e}")
        return RedirectResponse(url="http://localhost:5173?auth=error")

@app.get("/api/user")
async def get_user(request: Request):
    """Get authenticated user information"""
    user = request.session.get('user')
    if user:
        return {"authenticated": True, "user": user}
    return {"authenticated": False}

@app.get("/api/logout")
async def logout(request: Request):
    """Log out the current user"""
    request.session.pop('user', None)
    return {"success": True, "message": "Logged out successfully"}

# Core functionality
def get_prompt_template(is_refinement_prompt: bool, with_context: bool = False, slide_count: int = 10) -> str:
    """
    Return a prompt template with optional PDF context integration and slide count.

    :param is_refinement_prompt: Whether this is the initial or refinement prompt.
    :param with_context: Whether to include PDF context in the prompt.
    :param slide_count: Number of slides to generate
    :return: The prompt template as f-string.
    """
    if is_refinement_prompt:
        with open(GlobalConfig.REFINEMENT_PROMPT_TEMPLATE, 'r', encoding='utf-8') as in_file:
            template = in_file.read()
    else:
        with open(GlobalConfig.INITIAL_PROMPT_TEMPLATE, 'r', encoding='utf-8') as in_file:
            template = in_file.read()
    
    # More robust slide count replacement - handle all variations
    # Replace any instruction about creating 10-12 slides
    template = re.sub(
        r'create\s+\d+\s+TO\s+\d+\s+SLIDES\s+in\s+total',
        f'create EXACTLY {slide_count} SLIDES in total',
        template,
        flags=re.IGNORECASE
    )
    
    template = re.sub(
        r'Unless explicitly (specified|instructed)[\w\s,]+create\s+\d+\s+TO\s+\d+\s+SLIDES\s+in\s+total',
        f'Create EXACTLY {slide_count} SLIDES in total',
        template,
        flags=re.IGNORECASE
    )
    
    # Add an explicit instruction at the beginning for emphasis
    slide_count_instruction = f"\n\nIMPORTANT: Generate EXACTLY {slide_count} slides, no more and no fewer.\n\n"
    
    # Insert this instruction at a strategic location
    if "Your output, i.e., the content" in template:
        template = template.replace(
            "Your output, i.e., the content",
            f"{slide_count_instruction}Your output, i.e., the content"
        )
    else:
        # If the marker text isn't found, insert near the beginning
        template = slide_count_instruction + template
    
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

def get_pdf_hash(pdf_bytes):
    """
    Generate a hash for the PDF content to identify exact duplicates
    
    :param pdf_bytes: Raw PDF data
    :return: SHA-256 hash of the PDF content
    """
    return hashlib.sha256(pdf_bytes).hexdigest()

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

def get_pdf_context(question: str, vector_db, model: str) -> str:
    """
    Retrieve relevant context from the PDF based on the user's question.
    
    :param question: User's question/prompt
    :param vector_db: Vector database to search
    :param model: LLM model name
    :return: Extracted context or empty string if none found
    """
    try:
        logger.info(f"Getting PDF context for question: {question[:50]}...")
        
        context_query = question
        
        # Search for relevant documents
        retriever = vector_db.as_retriever(search_kwargs={"k": 5})
        docs = retriever.get_relevant_documents(context_query)
        
        if not docs:
            logger.warning("No relevant documents found in PDF")
            return ""
            
        # Combine the content from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Limit the context length
        max_context_length = 4000  # Characters
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
            
        logger.info(f"Retrieved {len(docs)} documents, {len(context)} characters of context")
        return context
        
    except Exception as e:
        logger.error(f"Error retrieving PDF context: {e}")
        return ""
    
def search_pdf_images(query, top_k=1):
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

def generate_slide_deck(json_str: str, session_id: str, template: str = "Basic") -> Union[pathlib.Path, None]:
    """Create a slide deck and return the file path with a timestamp."""
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
                'the slide deck cannot be created, unfortunately'
                '\nPlease try again later.'
            )
            return None
    except RecursionError:
        handle_error(
            'Encountered a recursion error while parsing JSON...'
            'the slide deck cannot be created, unfortunately'
            '\nPlease try again later.'
        )
        return None
    except Exception as ex:
        handle_error(
            f'Encountered an error while parsing JSON: {str(ex)}...'
            'the slide deck cannot be created, unfortunately'
            '\nPlease try again later.'
        )
        return None

    if session_id in download_file_path and download_file_path[session_id]:
        path = pathlib.Path(download_file_path[session_id])
    else:
        # Add timestamp to file name
        file_name = f"presentation_{timestamp_str}.pptx"
        temp_dir = tempfile.mkdtemp()
        path = pathlib.Path(os.path.join(temp_dir, file_name))
        download_file_path[session_id] = str(path)

    try:
        logger.debug(f'Creating PPTX file: {download_file_path[session_id]}...')
        pptx_helper.generate_powerpoint_presentation(
            parsed_data,
            slides_template=template,
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
        download_file_path[session_id] = None  # Reset on error
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
    :param history: The chat history
    :return: The list of user messages.
    """
    return [msg[0] for msg in history if msg[0] is not None]

def get_last_response(history) -> str:
    """
    Get the last response generated by AI.
    :param history: The chat history
    :return: The response text.
    """
    for msg in reversed(history):
        if msg[1] is not None:
            return msg[1]
    return ""

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Welcome to the Presentation Generator API"}

@app.get("/health")
async def health_check():
    """API health check endpoint"""
    return {"status": "ok"}

# Add a dependency that optionally gets the user
async def get_current_user_optional(request: Request):
    """Get the current user if authenticated, otherwise return None"""
    return request.session.get('user')

@app.post("/api/session")
async def create_session(req: SessionRequest = None, user: dict = Depends(get_current_user_optional)):
    """Create a new session"""
    if req and req.session_id:
        session_id = req.session_id
    else:
        session_id = get_session_id()
    
    # Get user ID if authenticated
    user_id = user.get('sub') if user else None
    
    # Create session in the database
    db.create_session(session_id, user_id)
    
    # Add an initial greeting
    greeting = random.choice(APP_TEXT['ai_greetings'])
    db.save_message(session_id, False, greeting)
    
    return {"session_id": session_id, "greeting": greeting}

@app.post("/api/generate")
async def generate_presentation(req: MessageRequest, request: Request, user: dict = Depends(get_current_user)):
    """Process user message and generate a presentation"""
    global is_refinement, download_file_path, vector_db, used_pdf_images, used_pexels_images, openai_api_key
    
    session_id = req.session_id
    message = req.message
    use_pdf_context = req.use_pdf_context
    # slide_count = req.slide_count
    slide_count = max(1, req.slide_count - 2) if req.slide_count > 3 else req.slide_count
    
    # Update session activity
    db.update_session_activity(session_id)
    
    # Get existing session messages from database
    session_messages = db.get_session_messages(session_id)
    
    user_template = db.get_user_template(session_id) if hasattr(db, 'get_user_template') else None
    current_template = user_template or pptx_template

    # Reset image tracking when starting a new presentation
    if req.is_new_presentation:
        download_file_path[session_id] = None
        used_pdf_images = set()
        used_pexels_images = set()
    
    if not message.strip():
        return JSONResponse(status_code=400, content={"error": "Message cannot be empty"})
    
    # Check if user has a preferred OpenAI model set in session
    if "current_openai_model" in request.session and "openai_api_keys" in request.session:
        current_model = request.session.get("current_openai_model")
        if current_model in request.session["openai_api_keys"]:
            # Use OpenAI as provider with the selected model
            provider = "openai"
            llm_name = current_model
            api_key = request.session["openai_api_keys"][current_model]
            logger.info(f"Using user's selected OpenAI model: {llm_name}")
        else:
            # Fallback to global setting
            provider, llm_name = llm_helper.get_provider_model(
                llm_provider_to_use,
                use_ollama=True
            )
            api_key = None
    else:
        # Use global default (Ollama model)
        provider, llm_name = llm_helper.get_provider_model(
            llm_provider_to_use,
            use_ollama=True
        )
        api_key = None
    
    valid, error_msg = are_all_inputs_valid(message, provider, llm_name)
    if not valid:
        return JSONResponse(status_code=400, content={"error": error_msg})
    
    logger.info(
        f'User input: {message} | #characters: {len(message)} | Provider: {provider} | Model: {llm_name} | Use PDF: {use_pdf_context} | Slides: {slide_count}'
    )
    
    # Save user message to database
    db.save_message(session_id, True, message)
    
    # Retrieve context from PDF if available and requested
    pdf_context = ""
    if use_pdf_context and vector_db:
        pdf_context = get_pdf_context(message, vector_db, llm_name)
        
        if pdf_context:
            logger.info(f"PDF context retrieved ({len(pdf_context)} characters)")
        else:
            logger.warning("No relevant PDF context found")
    
    # Prepare the prompt for slide generation
    prompt_template = ChatPromptTemplate.from_template(
        get_prompt_template(
            is_refinement_prompt=False,
            with_context=bool(pdf_context),
            slide_count=slide_count
        )
    )
    
    # Format the template with the message and context
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
    
    try:
        # Use the provider and model determined above
        llm = llm_helper.get_langchain_llm(
            provider=provider,
            model=llm_name,
            max_new_tokens=16384,
            api_key=api_key
        )
        
        if not llm:
            error_msg = (
                'Failed to create an LLM instance! Make sure that you have provided '
                'the correct model name.'
            )
            handle_error(error_msg, False)
            db.save_message(session_id, False, error_msg)
            return JSONResponse(status_code=500, content={"error": error_msg})
        
        # Get the full response
        for chunk in llm.stream(formatted_template):
            if isinstance(chunk, str):
                response += chunk
            else:
                response += chunk.content  # AIMessageChunk
            
    except ollama.ResponseError:
        error_msg = (
            f'The model `{llm_name}` is unavailable with Ollama on your system. '
            f'Make sure that you have provided the correct LLM name or pull it using '
            f'`ollama pull {llm_name}`. View LLMs available locally by running `ollama list`.'
        )
        handle_error(error_msg)
        db.save_message(session_id, False, error_msg)
        download_file_path[session_id] = None
        return JSONResponse(status_code=500, content={"error": error_msg})
    except Exception as ex:
        error_msg = (
            f'An unexpected error occurred while generating the content: {str(ex)}\n\n'
            'Please try again later, possibly with different inputs.'
        )
        handle_error(error_msg)
        db.save_message(session_id, False, error_msg)
        return JSONResponse(status_code=500, content={"error": error_msg})
    
    # Clean up the JSON response
    response = text_helper.get_clean_json(response)
    logger.info(f'Cleaned JSON length: {len(response)}')
    
    # Validate slide count and add note if necessary
    try:
        parsed_data = json5.loads(response)
        actual_slide_count = len(parsed_data.get('slides', []))
        logger.info(f'Generated {actual_slide_count} slides (requested {slide_count})')
        
        # If we have significantly fewer or more slides than requested, log a warning
        if abs(actual_slide_count - slide_count) > 2:
            logger.warning(f'Slide count mismatch: requested {slide_count}, got {actual_slide_count}')
            
            # Optionally add a note to the response
            if 'slides' in parsed_data and len(parsed_data['slides']) > 0:
                note = f"Note: You requested {slide_count} slides but received {actual_slide_count}."
                # Add note to the first slide
                if 'bullet_points' in parsed_data['slides'][0]:
                    if isinstance(parsed_data['slides'][0]['bullet_points'], list):
                        parsed_data['slides'][0]['bullet_points'].insert(0, note)
                        response = json.dumps(parsed_data)  # Update response with the note
    except Exception as e:
        logger.error(f'Error validating slide count: {e}')
    
    # Save AI response to database
    db.save_message(session_id, False, response)
    
    # Generate the slide deck
    slide_path = generate_slide_deck(response, session_id, current_template)
    
    if slide_path:
        # Save the presentation to database
        presentation_id = db.save_presentation(
            session_id=session_id,
            file_path=str(slide_path),
            template=pptx_template,
            slide_count=actual_slide_count if 'actual_slide_count' in locals() else slide_count
        )
        
        return PresentationResponse(
            success=True,
            message="Presentation generated successfully",
            download_url=f"/api/download?session_id={session_id}",
            history=db.get_session_messages(session_id)
        )
    else:
        return JSONResponse(status_code=500, content={
            "success": False, 
            "error": "Failed to generate presentation", 
            "history": db.get_session_messages(session_id)
        })

@app.get("/api/history")
async def get_history(session_id: str):
    """Get chat history for a session"""
    messages = db.get_session_messages(session_id)
    if not messages:
        return JSONResponse(status_code=404, content={"error": "Session not found or no messages"})
    
    return {"history": messages}

@app.post("/api/clear-history")
async def clear_history(req: SessionRequest):
    """Clear chat history for a session"""
    session_id = req.session_id
    
    # Clear messages in database
    if not db.clear_session_messages(session_id):
        return JSONResponse(status_code=404, content={"error": "Session not found"})
    
    # Reset other session state
    download_file_path[session_id] = None
    used_pdf_images = set()
    used_pexels_images = set()
    
    # Add a new greeting
    greeting = random.choice(APP_TEXT['ai_greetings'])
    db.save_message(session_id, False, greeting)
    
    return {"success": True, "message": "History cleared", "greeting": greeting}



def create_vector_db_from_doc(file_content) -> Optional[Chroma]:
    """
    Create a vector database from an uploaded DOC/DOCX file with timestamped filename.
    Also extracts and processes images from the document.
    """
    global pdf_images, image_embedding_model  # Reuse the same global variable for both PDFs and DOCs
    
    if not file_content:
        logger.warning("No file content provided")
        return None
        
    logger.info(f"Creating vector DB from DOC/DOCX file")
    
    try:
        # Make sure we have the embedding model initialized
        if image_embedding_model is None:
            try:
                # Load a multimodal model that can handle images and text
                image_embedding_model = SentenceTransformer('clip-ViT-B-32')
                logger.info("Loaded CLIP embedding model for images")
            except Exception as e:
                logger.error(f"Failed to load image embedding model: {e}")
                # Continue without image processing capability
                logger.info("Continuing without image processing capability")
        
        # Generate a timestamp for the filename
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        collection_name = f"doc_{timestamp_str}"
        
        # Process the document with better error handling
        try:
            doc_processor = DocProcessor(image_embedding_model=image_embedding_model)
            logger.info("DocProcessor created successfully")
            
            # Process the file with detailed logging
            extracted_text, extracted_images = doc_processor.process_doc_file(file_content)
            logger.info(f"Document processing complete. Text length: {len(extracted_text)}, Images: {len(extracted_images)}")
            
            if not extracted_text:
                logger.warning("No text extracted from DOC file")
                return None
        except Exception as e:
            logger.error(f"Error in document processing: {e}", exc_info=True)
            raise
        
        # Create documents from extracted text
        try:
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
            texts = text_splitter.split_text(extracted_text)
            
            # Create document objects
            documents = [
                LangchainDocument(
                    page_content=chunk,
                    metadata={"source": f"doc_chunk_{i}", "chunk_id": i}
                )
                for i, chunk in enumerate(texts)
            ]
            
            logger.info(f"Document split into {len(documents)} chunks")

            if len(documents) == 0:
                logger.warning("No chunks created from DOC file")
                return None
        except Exception as e:
            logger.error(f"Error splitting document into chunks: {e}", exc_info=True)
            raise
        
        # Create vector embeddings
        try:
            # Check if Ollama is running first to provide a clearer error message
            try:
                # Make a simple request to check if Ollama is running
                import requests
                response = requests.get("http://localhost:11434/api/version", timeout=2)
                if response.status_code == 200:
                    logger.info("Ollama server is running and reachable")
                else:
                    logger.warning(f"Ollama server returned status code {response.status_code}")
            except requests.exceptions.ConnectionError:
                logger.error("Cannot connect to Ollama server at http://localhost:11434")
                logger.error("Please make sure Ollama is installed and running:")
                logger.error("1. Download from https://ollama.com/")
                logger.error("2. Start the Ollama application")
                logger.error("3. Pull the required model: 'ollama pull nomic-embed-text'")
                raise ConnectionError("Cannot connect to Ollama server - please ensure it's running")
            except Exception as e:
                logger.warning(f"Error checking Ollama status: {e}")
            
            # Create the embeddings
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            
            # Create the vector database
            logger.info(f"Creating vector database with collection name: {collection_name}")
            vector_db = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=PERSIST_DIRECTORY,
                collection_name=collection_name
            )
        except Exception as e:
            logger.error(f"Error creating vector database: {e}", exc_info=True)
            raise
        
        # Store extracted images in the same global variable used for PDFs
        if extracted_images:
            pdf_images = extracted_images
            logger.info(f"Stored {len(extracted_images)} images from DOC file")
        
        logger.info("Vector DB created for DOC file with persistent storage")
        return vector_db
        
    except ConnectionError as e:
        logger.error(f"Connection error: {e}")
        return None
    except ImportError as e:
        logger.error(f"Missing dependency for DOC processing: {e}")
        logger.error("Please install required packages: pip install python-docx pillow requests")
        return None
    except Exception as e:
        logger.error(f"Error creating vector DB from DOC file: {e}", exc_info=True)
        return None
        
@app.post("/api/process-document")
async def process_document(file: UploadFile = File(...), session_id: str = Form(...), user: dict = Depends(get_current_user_optional)):
    """Process PDF, DOC, or DOCX file and create vector database"""
    global vector_db, pdf_images, pdf_hashes
    
    user_id = user.get('sub') if user else None
    
    # Check file extension
    filename = file.filename.lower()
    if not (filename.endswith('.pdf') or filename.endswith('.doc') or filename.endswith('.docx')):
        return JSONResponse(status_code=400, content={
            "success": False,
            "message": "Only PDF, DOC, or DOCX files are accepted"
        })
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Calculate hash
        doc_hash = hashlib.sha256(file_content).hexdigest()
        logger.info(f"Document hash: {doc_hash}")
        
        # Check if this exact document has been processed before in the database
        existing_file = db.get_file_by_hash(doc_hash)
        existing_file = False
        if existing_file:
            collection_name = existing_file['collection_name']
            logger.info(f"Exact duplicate document detected! Using collection: {collection_name}")
            
            try:
                # Load the existing vector DB
                vector_db = Chroma(
                    collection_name=collection_name,
                    persist_directory=PERSIST_DIRECTORY,
                    embedding_function=OllamaEmbeddings(model="nomic-embed-text")
                )
                
                # Store in memory hash map too for quick lookup
                pdf_hashes[doc_hash] = collection_name
                
                return PDFResponse(
                    success=True,
                    message=f"📄 Exact duplicate document detected! Using existing embeddings.",
                    collection_info={
                        "name": collection_name,
                        "is_duplicate": True
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to load existing collection {collection_name}: {e}")
                # Fall through to reprocess
        
        # Process the document based on file type
        if filename.endswith('.pdf'):
            # Existing PDF processing
            vector_db = create_vector_db(file_content)
            file_type = "application/pdf"
        else:
            # DOC/DOCX processing
            vector_db = create_vector_db_from_doc(file_content)
            file_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            
        if vector_db:
            try:
                # Get collection name safely
                collection_name = getattr(vector_db._collection, 'name', None)
                if collection_name:
                    # Store the hash in memory and save to database
                    pdf_hashes[doc_hash] = collection_name
                    
                    # Save file upload record
                    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    file_path = f"uploaded_{timestamp_str}.{filename.split('.')[-1]}"
                    
                    db.save_file_upload(
                        session_id=session_id,
                        user_id=user_id,
                        file_name=file.filename,
                        file_path=file_path,
                        file_type=file_type,
                        file_hash=doc_hash,
                        collection_name=collection_name
                    )
                    
                    # Extract vector DB info
                    count = "Unknown"
                    if hasattr(vector_db._collection, 'count') and callable(vector_db._collection.count):
                        try:
                            count = vector_db._collection.count()
                        except:
                            pass
                            
                    return PDFResponse(
                        success=True,
                        message=f"✅ Successfully processed {file.filename}",
                        collection_info={
                            "name": collection_name,
                            "count": count,
                            "is_new": True
                        }
                    )
            except Exception as e:
                logger.warning(f"Could not store document hash: {e}")
                
            return PDFResponse(
                success=True,
                message=f"✅ Successfully processed {file.filename}"
            )
        else:
            return JSONResponse(status_code=500, content={
                "success": False,
                "message": f"❌ Error processing {file.filename}. Please try another file."
            })
            
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        return JSONResponse(status_code=500, content={
            "success": False,
            "message": f"❌ Error processing document: {str(e)}"
        })


@app.post("/api/process-pdf")
async def process_pdf(file: UploadFile = File(...), session_id: str = Form(...), user: dict = Depends(get_current_user_optional)):
    """Process PDF file and create vector database"""
    global vector_db, pdf_hashes
    
    user_id = user.get('sub') if user else None
    
    if not file.filename.endswith('.pdf'):
        return JSONResponse(status_code=400, content={
            "success": False,
            "message": "Only PDF files are accepted"
        })
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Calculate hash
        pdf_hash = get_pdf_hash(file_content)
        logger.info(f"PDF hash: {pdf_hash}")
        
        # Check if this exact PDF has been processed before in the database
        existing_file = db.get_file_by_hash(pdf_hash)
        if existing_file:
            collection_name = existing_file['collection_name']
            logger.info(f"Exact duplicate PDF detected! Using collection: {collection_name}")
            
            try:
                # Load the existing vector DB
                vector_db = Chroma(
                    collection_name=collection_name,
                    persist_directory=PERSIST_DIRECTORY,
                    embedding_function=OllamaEmbeddings(model="nomic-embed-text")
                )
                
                # Store in memory hash map too for quick lookup
                pdf_hashes[pdf_hash] = collection_name
                
                return PDFResponse(
                    success=True,
                    message=f"📄 Exact duplicate PDF detected! Using existing embeddings.",
                    collection_info={
                        "name": collection_name,
                        "is_duplicate": True
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to load existing collection {collection_name}: {e}")
                # Fall through to reprocess
        
        # Try content similarity check
        similar_found, collection_name, similarity = check_pdf_similarity(file_content)
        
        if similar_found and collection_name:
            try:
                # Load the existing vector DB
                vector_db = Chroma(
                    collection_name=collection_name,
                    persist_directory=PERSIST_DIRECTORY,
                    embedding_function=OllamaEmbeddings(model="nomic-embed-text")
                )
                
                # Store in memory and database
                pdf_hashes[pdf_hash] = collection_name
                
                # Format the similarity as a percentage
                similarity_pct = f"{similarity * 100:.1f}%"
                
                # Save file upload record
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = f"uploaded_{timestamp_str}.pdf"
                
                db.save_file_upload(
                    session_id=session_id,
                    user_id=user_id,
                    file_name=file.filename,
                    file_path=file_path,
                    file_type="application/pdf",
                    file_hash=pdf_hash,
                    collection_name=collection_name
                )
                
                logger.info(f"Using existing PDF embeddings for similar document: {collection_name}")
                return PDFResponse(
                    success=True,
                    message=f"📄 Similar PDF detected ({similarity_pct} match)! Using existing embeddings.",
                    collection_info={
                        "name": collection_name,
                        "similarity": similarity,
                        "similarity_percent": similarity_pct,
                        "is_similar": True
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to load similar collection {collection_name}: {e}")
        
        # Process the new PDF
        logger.info("Processing PDF as new document")
        vector_db = create_vector_db(file_content)
        
        if vector_db:
            try:
                # Get collection name safely
                collection_name = getattr(vector_db._collection, 'name', None)
                if collection_name:
                    # Store the hash in memory and save to database
                    pdf_hashes[pdf_hash] = collection_name
                    
                    # Save file upload record
                    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    file_path = f"uploaded_{timestamp_str}.pdf"
                    
                    db.save_file_upload(
                        session_id=session_id,
                        user_id=user_id,
                        file_name=file.filename,
                        file_path=file_path,
                        file_type="application/pdf",
                        file_hash=pdf_hash,
                        collection_name=collection_name
                    )
                    
                    # Extract vector DB info
                    count = "Unknown"
                    if hasattr(vector_db._collection, 'count') and callable(vector_db._collection.count):
                        try:
                            count = vector_db._collection.count()
                        except:
                            pass
                            
                    return PDFResponse(
                        success=True,
                        message="✅ Successfully processed new PDF",
                        collection_info={
                            "name": collection_name,
                            "count": count,
                            "is_new": True
                        }
                    )
            except Exception as e:
                logger.warning(f"Could not store PDF hash: {e}")
                
            return PDFResponse(
                success=True,
                message="✅ Successfully processed new PDF"
            )
        else:
            return JSONResponse(status_code=500, content={
                "success": False,
                "message": "❌ Error processing PDF. Please try another file."
            })
            
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        return JSONResponse(status_code=500, content={
            "success": False,
            "message": f"❌ Error processing PDF: {str(e)}"
        })
    
@app.get("/api/vector-db-info")
async def get_vector_db_info_api():
    """Get information about the current vector database"""
    if not vector_db:
        return VectorDBInfoResponse(success=False, visible=False)
        
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
        
        return VectorDBInfoResponse(
            success=True,
            collection_name=collection_name,
            count=count,
            visible=True
        )
    except Exception as e:
        logger.error(f"Error getting vector DB info: {e}")
        return VectorDBInfoResponse(success=False, error=str(e), visible=False)

@app.get("/api/download")
async def download_presentation(session_id: str):
    """Download the generated presentation"""
    if session_id not in download_file_path or not download_file_path[session_id]:
        raise HTTPException(status_code=404, detail="Presentation not found")
    
    file_path = download_file_path[session_id]
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Presentation file not found")
    
    # Use a timestamp for the downloaded file name if not already in the filename
    if "_202" not in file_path:
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"Presentation_{timestamp_str}.pptx"
    else:
        file_name = os.path.basename(file_path)
    
    return FileResponse(
        path=file_path,
        filename=file_name,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation"
    )

@app.post("/api/upload-to-gdrive")
async def upload_to_gdrive_api(req: SessionRequest):
    """Upload the presentation to Google Drive"""
    session_id = req.session_id
    
    if session_id not in download_file_path or not download_file_path[session_id]:
        return JSONResponse(status_code=404, content={
            "success": False,
            "message": "No presentation file available for upload."
        })
    
    file_path = download_file_path[session_id]
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={
            "success": False,
            "message": "Presentation file not found."
        })
    
    try:
        result = upload_to_gdrive(file_path)
        if result:
            file_id, view_link = result
            
            # Get the most recent presentation for this session
            presentation = db.get_latest_presentation(session_id)
            if presentation:
                # Update the presentation with the Google Drive link
                db.update_presentation_gdrive(presentation['id'], view_link)
            
            return GDriveUploadResponse(
                success=True,
                message="Successfully uploaded to Google Drive!",
                view_link=view_link
            )
        else:
            return JSONResponse(status_code=500, content={
                "success": False,
                "message": "Failed to upload to Google Drive. Check logs for details."
            })
    except Exception as e:
        logger.error(f"Error in upload to Google Drive: {e}")
        return JSONResponse(status_code=500, content={
            "success": False,
            "message": f"Error: {str(e)}"
        })
    
@app.get("/api/templates")
async def get_templates():
    """Get available presentation templates"""
    templates = list(GlobalConfig.PPTX_TEMPLATE_FILES.keys())
    template_info = {
        name: {
            "caption": GlobalConfig.PPTX_TEMPLATE_FILES[name]['caption'],
            "description": GlobalConfig.PPTX_TEMPLATE_FILES[name].get('description', '')
        } for name in templates
    }
    
    return {
        "templates": templates,
        "current_template": pptx_template,
        "template_info": template_info
    }

# 3. Update the set_template function to store per-user or per-session preferences
@app.post("/api/set-template")
async def set_template(req: TemplateRequest, request: Request, user: dict = Depends(get_current_user_optional)):
    """Set the presentation template"""
    global pptx_template
    
    template_name = req.template_name
    templates = list(GlobalConfig.PPTX_TEMPLATE_FILES.keys())
    
    if template_name not in templates:
        return JSONResponse(status_code=400, content={
            "success": False,
            "message": f"Template '{template_name}' not found"
        })
    
    # Update the global default (for backward compatibility)
    pptx_template = template_name
    
    # Store user's template preference in session
    if user:
        # If you have a database function to store user preferences:
        if hasattr(db, 'save_user_template'):
            db.save_user_template(user.get('sub'), template_name)
        # Otherwise store in session
        request.session['template'] = template_name
    
    logger.info(f"Template changed to: {template_name} for user: {user.get('sub') if user else 'anonymous'}")
    
    return {
        "success": True,
        "message": f"Template changed to: {template_name}"
    }

@app.post("/api/set-llm")
async def set_llm(req: LLMRequest, request: Request):
    """Set the LLM model"""
    global llm_provider_to_use
    
    model_name = req.model_name
    
    # Handle OpenAI models
    if req.provider == "openai" and req.api_key:
        # Store OpenAI API key in session instead of global variable
        if "openai_api_keys" not in request.session:
            request.session["openai_api_keys"] = {}
        
        # Store the key with the model as identifier
        request.session["openai_api_keys"][model_name] = req.api_key
        request.session["current_openai_model"] = model_name
        
        # Format as [op]model_name as expected by the llm_helper
        llm_provider_to_use = f"[op]{model_name}"
        logger.info(f"LLM changed to OpenAI: {model_name} (key stored in session)")
    else:
        # Use Ollama or other models as before
        llm_provider_to_use = model_name
        logger.info(f"LLM changed to: {model_name}")
    
    return {
        "success": True,
        "message": f"LLM changed to: {model_name}"
    }

@app.post("/api/set-gdrive-upload")
async def set_gdrive_upload_api(req: GDriveSettingRequest):
    """Set Google Drive upload setting"""
    global ENABLE_GDRIVE_UPLOAD
    
    ENABLE_GDRIVE_UPLOAD = req.enable
    logger.info(f"Google Drive upload {'enabled' if req.enable else 'disabled'}")
    
    return {
        "success": True,
        "enabled": ENABLE_GDRIVE_UPLOAD
    }

@app.get("/api/usage-instructions")
async def get_usage_instructions():
    """Get usage instructions"""
    return {"instructions": GlobalConfig.CHAT_USAGE_INSTRUCTIONS}


@app.get("/api/user-profile")
async def get_user_profile(user: dict = Depends(get_current_user)):
    """Get user profile information"""
    if not user:
        return JSONResponse(status_code=401, content={
            "success": False,
            "message": "Unauthorized"
        })
    
    user_id = user.get('sub')
    
    try:
        user_data = db.get_user(user_id)
        if user_data:
            return {
                "success": True,
                "profile": {
                    "id": user_data.get('id'),
                    "email": user_data.get('email'),
                    "name": user_data.get('name'),
                    "picture_url": user_data.get('picture'),
                    "created_at": user_data.get('created_at').isoformat() if user_data.get('created_at') else None,
                    "last_login": user_data.get('last_login').isoformat() if user_data.get('last_login') else None
                }
            }
    except Exception as e:
        logger.error(f"Error retrieving user profile: {e}")
    
    # Fallback to session data
    return {
        "success": True,
        "profile": {
            "id": user.get('sub'),
            "email": user.get('email'),
            "name": user.get('name'),
            "picture_url": user.get('picture'),
            "created_at": datetime.datetime.now().isoformat(),
            "last_login": datetime.datetime.now().isoformat()
        }
    }

@app.get("/api/user-uploads")
async def get_user_uploads(user: dict = Depends(get_current_user)):
    """Get user's PDF uploads"""
    if not user:
        return JSONResponse(status_code=401, content={
            "success": False,
            "message": "Unauthorized"
        })
    
    user_id = user.get('sub')
    
    try:
        uploads = []
        with db.get_connection().cursor() as cursor:
            cursor.execute(
                """
                SELECT id, file_name, collection_name, file_hash, created_at
                FROM file_uploads
                WHERE user_id = %s
                ORDER BY created_at DESC
                """,
                (user_id,)
            )
            
            for row in cursor.fetchall():
                uploads.append({
                    "id": row['id'],
                    "file_name": row['file_name'],
                    "collection_name": row['collection_name'],
                    "file_hash": row['file_hash'],
                    "created_at": row['created_at'].isoformat()
                })
                
        return {
            "success": True,
            "uploads": uploads
        }
    except Exception as e:
        logger.error(f"Error retrieving user uploads: {e}")
        return {
            "success": False,
            "message": str(e),
            "uploads": []
        }

@app.get("/api/user-presentations")
async def get_user_presentations(user: dict = Depends(get_current_user)):
    """Get user's generated presentations"""
    if not user:
        return JSONResponse(status_code=401, content={
            "success": False,
            "message": "Unauthorized"
        })
    
    user_id = user.get('sub')
    
    try:
        presentations = []
        with db.get_connection().cursor() as cursor:
            # More robust query that doesn't fail if column is missing
            cursor.execute("""
                SELECT p.id, p.session_id, p.file_path, p.template, p.slide_count, 
                       p.created_at, p.gdrive_link,
                       IFNULL(p.download_count, 0) as download_count
                FROM presentations p
                JOIN sessions s ON p.session_id = s.session_id
                WHERE s.user_id = %s
                ORDER BY p.created_at DESC
                """,
                (user_id,)
            )
            
            for row in cursor.fetchall():
                presentations.append({
                    "id": row['id'],
                    "session_id": row['session_id'],
                    "file_path": row['file_path'],
                    "template": row['template'],
                    "slide_count": row['slide_count'],
                    "created_at": row['created_at'].isoformat(),
                    "gdrive_link": row['gdrive_link'],
                    "download_count": row['download_count']
                })
                
        return {
            "success": True,
            "presentations": presentations
        }
    except Exception as e:
        logger.error(f"Error retrieving user presentations: {e}")
        return {
            "success": False,
            "message": str(e),
            "presentations": []
        }

@app.get("/api/user-sessions")
async def get_user_sessions(user: dict = Depends(get_current_user)):
    """Get user's chat sessions"""
    if not user:
        return JSONResponse(status_code=401, content={
            "success": False,
            "message": "Unauthorized"
        })
    
    user_id = user.get('sub')
    
    try:
        sessions = []
        with db.get_connection().cursor() as cursor:
            # Get sessions with message counts
            cursor.execute(
                """
                SELECT s.session_id, s.created_at, s.last_activity,
                       COUNT(cm.id) as message_count
                FROM sessions s
                LEFT JOIN chat_messages cm ON s.session_id = cm.session_id
                WHERE s.user_id = %s
                GROUP BY s.session_id
                ORDER BY s.last_activity DESC
                """,
                (user_id,)
            )
            
            for row in cursor.fetchall():
                sessions.append({
                    "id": row['session_id'],
                    "created_at": row['created_at'].isoformat(),
                    "last_active": row['last_activity'].isoformat(),
                    "message_count": row['message_count']
                })
                
        return {
            "success": True,
            "sessions": sessions
        }
    except Exception as e:
        logger.error(f"Error retrieving user sessions: {e}")
        return {
            "success": False,
            "message": str(e),
            "sessions": []
        }

@app.on_event("startup")
async def startup_db_client():
    try:
        # Ensure database tables are created
        db.init_db()
        
        # Add missing columns if they don't exist
        try:
            with db.get_connection().cursor() as cursor:
                # Check if column exists
                cursor.execute("""
                    SELECT COUNT(*) as count
                    FROM information_schema.columns 
                    WHERE table_name = 'presentations' 
                    AND column_name = 'download_count'
                    AND table_schema = DATABASE()
                """)
                result = cursor.fetchone()
                
                # Add the column if it doesn't exist
                if result and result['count'] == 0:
                    cursor.execute("""
                        ALTER TABLE presentations 
                        ADD COLUMN download_count INT DEFAULT 0
                    """)
                    logger.info("Added missing download_count column to presentations table")
        except Exception as e:
            logger.error(f"Error checking/adding columns: {e}")
            
        logger.info("Database initialized successfully at startup")
    except Exception as e:
        logger.error(f"Failed to initialize database at startup: {e}")

# Close database connection on shutdown
@app.on_event("shutdown")
async def shutdown_db_client():
    try:
        db.close_connection()
        logger.info("Database connection closed")
    except Exception as e:
        logger.error(f"Error closing database connection: {e}")
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)