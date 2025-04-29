"""
Gradio app containing the UI and application logic for PowerPoint generation
with PDF RAG support. Modified for user Google Drive upload via Flask backend.
"""
import datetime
import logging
import os
import pathlib
import random
import tempfile
import shutil
from typing import List, Dict, Union, Any, Tuple, Optional
import fitz # PyMuPDF
import json5
import ollama
import gradio as gr
# Bỏ dotenv load ở đây, Flask sẽ load
# from dotenv import load_dotenv
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
import hashlib
import requests # Thêm thư viện requests để gọi Flask backend
import json

# --- Google Drive Imports (Chỉ cần cho hàm upload mới) ---
# Bỏ google.oauth2.service_account
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
# ---

# Logging setup (giữ nguyên)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
PERSIST_DIRECTORY = os.path.join("data", "vectors")
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

# Load dotenv() được thực hiện bởi Flask wrapper

# Global state variables (quản lý bởi Gradio instance)
# chat_history = [] # Không dùng global, dùng state của Gradio
is_refinement = False
# download_file_path = None # Dùng gr.State thay thế
pptx_template = "Basic"
llm_provider_to_use = "mistral:v0.2"
vector_db = None
pdf_images = []
image_embeddings = []
image_embedding_model = None
used_pdf_images = set()
used_pexels_images = set()
pdf_hashes = {} # Maps hash -> collection_name

# --- Các hàm xử lý cốt lõi (load_strings, get_prompt_template, are_all_inputs_valid, handle_error, ...) ---
# --- Giữ nguyên các hàm này ---
def load_strings() -> dict:
    with open(GlobalConfig.APP_STRINGS_FILE, 'r', encoding='utf-8') as in_file:
        return json5.loads(in_file.read())

APP_TEXT = load_strings() # Load strings khi module được import

def get_prompt_template(is_refinement_prompt: bool, with_context: bool = False) -> str:
    # ... (giữ nguyên code)
    if is_refinement_prompt:
        template_path = GlobalConfig.REFINEMENT_PROMPT_TEMPLATE
    else:
        template_path = GlobalConfig.INITIAL_PROMPT_TEMPLATE

    try:
        with open(template_path, 'r', encoding='utf-8') as in_file:
            template = in_file.read()
    except FileNotFoundError:
        logger.error(f"Prompt template file not found: {template_path}")
        # Fallback to a very basic template
        if is_refinement_prompt:
            template = "Refine the previous content based on these instructions:\n{instructions}\n\nPrevious Content:\n{previous_content}"
            if with_context:
                 template += "\n\nUse this context:\n{context}"
        else:
            template = "Create a presentation about: {question}"
            if with_context:
                template = "Use this context:\n{context}\n\nCreate a presentation about: {question}"
        logger.warning("Using basic fallback prompt template.")


    if with_context:
        context_template = """
        Use the following information from the PDF document as a knowledge source:

        {context}

        Extract key points, data, and insights from this content, but create a well-structured
        presentation that extends beyond just these facts. Use this information to enhance
        your response while still addressing the core request.
        """
        # Insert context instructions (logic giữ nguyên)
        # ... (logic to insert context_template into template)
        if "create a PowerPoint presentation" in template.lower():
             parts = template.split("create a PowerPoint presentation", 1)
             template = parts[0] + context_template + "\n\ncreate a PowerPoint presentation" + parts[1]
        elif "refine the previous content" in template.lower() and is_refinement_prompt:
             # Insert after instructions but before previous content
             parts = template.split("Previous Content:", 1)
             template = parts[0] + context_template + "\n\nPrevious Content:" + parts[1]
        else:
             template = context_template + "\n\n" + template

    return template

def are_all_inputs_valid(user_prompt: str, selected_provider: str, selected_model: str) -> tuple[bool, str]:
    # ... (giữ nguyên code)
    if not text_helper.is_valid_prompt(user_prompt):
        return False, ('Not enough information provided! Please be a little more descriptive and '
                       'type a few words with a few characters :)')
    if not selected_provider or not selected_model:
        return False, 'No valid LLM provider and/or model name found!'
    return True, ""


def handle_error(error_msg: str, should_log: bool = True) -> str:
     # ... (giữ nguyên code)
    if should_log:
        logger.error(error_msg)
    return error_msg

# --- Image Handling (extract_and_embed_images, search_pdf_images, get_image_for_slide) ---
# --- Giữ nguyên các hàm này ---
def extract_and_embed_images(pdf_path):
    global pdf_images, image_embedding_model
    if image_embedding_model is None:
        try:
            image_embedding_model = SentenceTransformer('clip-ViT-B-32')
            logger.info("Loaded CLIP embedding model for images")
        except Exception as e:
            logger.error(f"Failed to load image embedding model: {e}")
            return []

    img_dir = os.path.join(tempfile.gettempdir(), f"pdf_images_{uuid.uuid4().hex}")
    os.makedirs(img_dir, exist_ok=True)
    logger.info(f"Created temp directory for images: {img_dir}")
    extracted_images = []
    image_count = 0
    try:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            image_list = page.get_images(full=True)
            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    img_filename = f"page{page_num+1}_img{img_idx+1}.{image_ext}"
                    img_path = os.path.join(img_dir, img_filename)

                    with open(img_path, "wb") as img_file:
                        img_file.write(image_bytes)

                    pil_img = Image.open(io.BytesIO(image_bytes))
                    if pil_img.width < 100 or pil_img.height < 100:
                        os.remove(img_path) # Clean up small images
                        continue

                    img_embedding = image_embedding_model.encode(pil_img)
                    extracted_images.append({
                        "path": img_path, "page": page_num + 1, "image_bytes": image_bytes,
                        "embedding": img_embedding, "width": pil_img.width, "height": pil_img.height,
                         "aspect_ratio": pil_img.width / pil_img.height if pil_img.height else 1
                    })
                    image_count += 1
                except Exception as e:
                    logger.warning(f"Failed to process image xref {xref} on page {page_num+1}: {e}")
        doc.close() # Close the document
    except Exception as e:
        logger.error(f"Error extracting images from PDF: {e}")
        # Clean up directory if extraction fails badly
        if os.path.exists(img_dir):
             try:
                 shutil.rmtree(img_dir)
             except Exception as rm_err:
                 logger.error(f"Failed to clean up temp image directory {img_dir}: {rm_err}")
        return []

    pdf_images = extracted_images
    logger.info(f"Extracted {image_count} usable images from PDF")
    # Keep the temp dir, maybe add a cleanup mechanism later if needed
    return extracted_images

def search_pdf_images(query, top_k=3):
    global pdf_images, image_embedding_model
    if not pdf_images or not image_embedding_model:
        logger.warning("No PDF images available or embedding model not loaded")
        return None
    try:
        text_embedding = image_embedding_model.encode(query)
        similarities = []
        for idx, img_data in enumerate(pdf_images):
            # Ensure embedding exists
            if "embedding" in img_data and img_data["embedding"] is not None:
                 similarity = util.cos_sim(text_embedding, img_data["embedding"])[0][0].item()
                 similarities.append((idx, similarity))
            else:
                 logger.warning(f"Missing embedding for image index {idx}")

        similarities.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in similarities[:top_k]:
            img_data = pdf_images[idx].copy() # Return a copy
            img_data["score"] = score
            if "embedding" in img_data:
                del img_data["embedding"] # Don't return embedding
            results.append(img_data)
        logger.info(f"Found {len(results)} relevant PDF images for query: {query}")
        return results
    except Exception as e:
        logger.error(f"Error searching PDF images: {e}")
        return None


def get_image_for_slide(keywords, use_pdf_images=True):
    global pdf_images, used_pdf_images, used_pexels_images
    if not keywords or isinstance(keywords, str) and not keywords.strip():
        logger.warning("Empty keywords for image search")
        return None

    logger.info(f"Searching image for keywords: {keywords}")

    # Try PDF images first
    if use_pdf_images and pdf_images:
        pdf_results = search_pdf_images(keywords, top_k=5) # Get more results
        if pdf_results:
             unused_results = [img for img in pdf_results if img["path"] not in used_pdf_images]
             if unused_results:
                 selected_img = unused_results[0] # Pick best unused
                 logger.info(f"Using unused PDF image: {selected_img['path']}")
                 used_pdf_images.add(selected_img["path"])
                 # Return a dictionary format consistent with Pexels result for simplicity
                 return {
                     "source": "pdf",
                     "path": selected_img["path"],
                     "page": selected_img.get("page"),
                     "image_bytes": selected_img.get("image_bytes"), # Include bytes if needed by pptx_helper
                     "score": selected_img.get("score")
                 }
             elif pdf_results: # All top results already used, reuse the best one
                 selected_img = pdf_results[0]
                 logger.info(f"Reusing PDF image (all top results used): {selected_img['path']}")
                 # Don't add to used_pdf_images again
                 return {
                     "source": "pdf",
                     "path": selected_img["path"],
                     "page": selected_img.get("page"),
                     "image_bytes": selected_img.get("image_bytes"),
                     "score": selected_img.get("score")
                 }

    # Fallback to Pexels
    try:
        from helpers import image_search as ims
        pexels_api_key = os.getenv("PEXELS_API_KEY")
        if not pexels_api_key:
             logger.warning("PEXELS_API_KEY not found in environment. Cannot search Pexels.")
             return None

        # Try to get an unused Pexels image
        for attempt in range(3):
            api_response = ims.search_pexels(query=keywords, size='medium', api_key=pexels_api_key)
            photo_url, page_url = ims.get_photo_url_from_api_response(api_response)
            if photo_url and photo_url not in used_pexels_images:
                logger.info(f"Using new Pexels image: {photo_url}")
                image_bytes = ims.get_image_from_url(photo_url)
                if image_bytes:
                    used_pexels_images.add(photo_url)
                    return {
                        "source": "pexels", "url": photo_url, "page_url": page_url,
                        "image_bytes": image_bytes
                    }
                else:
                     logger.warning(f"Failed to download image from Pexels URL: {photo_url}")
                     continue # Try again
        # If loop finishes without unused image, try one last time and reuse if necessary
        api_response = ims.search_pexels(query=keywords, size='medium', api_key=pexels_api_key)
        photo_url, page_url = ims.get_photo_url_from_api_response(api_response)
        if photo_url:
            logger.info(f"Reusing Pexels image (all attempts failed to find unused): {photo_url}")
            image_bytes = ims.get_image_from_url(photo_url)
            if image_bytes:
                # Don't add to used_pexels_images again if reusing
                return {
                    "source": "pexels", "url": photo_url, "page_url": page_url,
                    "image_bytes": image_bytes
                }
    except ImportError:
         logger.warning("helpers.image_search module not found. Cannot search Pexels.")
    except Exception as e:
        logger.error(f"Error fetching Pexels image: {e}")

    logger.warning(f"No image found for keywords: {keywords}")
    return None

# --- PDF Processing (create_vector_db, get_pdf_context, check_pdf_similarity, get_pdf_hash) ---
# --- Giữ nguyên các hàm này ---
def get_pdf_hash(pdf_bytes):
     return hashlib.sha256(pdf_bytes).hexdigest()

def create_vector_db(file_upload) -> Optional[Chroma]:
    global pdf_hashes # Access global hash dict
    if not file_upload: return None

    logger.info(f"Creating vector DB from file upload")
    temp_dir = tempfile.mkdtemp()
    vector_db_instance = None # Use local variable

    try:
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Include a random element to further decrease collision chance if needed
        random_suffix = uuid.uuid4().hex[:6]
        pdf_filename = f"uploaded_{timestamp_str}_{random_suffix}.pdf"
        collection_name = f"pdf_{timestamp_str}_{random_suffix}" # Use same unique id for collection

        path = os.path.join(temp_dir, pdf_filename)
        with open(path, "wb") as f:
            f.write(file_upload)
            logger.info(f"File saved to temporary path: {path}")

        # Extract images FIRST, so they are available even if text processing fails
        logger.info("Extracting images from PDF...")
        extract_and_embed_images(path) # Stores globally in pdf_images

        # Load text data
        loader = PyMuPDFLoader(path)
        data = loader.load()
        logger.info(f"Loaded {len(data)} pages from PDF")

        if not data:
            logger.warning("No text data loaded from PDF.")
            # Keep images, but return no vector_db
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
        chunks = text_splitter.split_documents(data)
        logger.info(f"Document split into {len(chunks)} chunks")

        if not chunks:
            logger.warning("No text chunks generated after splitting.")
            return None

        # Use consistent embedding model for text
        text_embeddings = OllamaEmbeddings(model="nomic-embed-text") # Make sure this model is available

        vector_db_instance = Chroma.from_documents(
            documents=chunks,
            embedding=text_embeddings,
            persist_directory=PERSIST_DIRECTORY,
            collection_name=collection_name # Use the unique name
        )
        logger.info(f"Vector DB created/loaded with collection: {collection_name}")

        # Store hash mapping AFTER successful creation
        try:
            pdf_hash = get_pdf_hash(file_upload)
            pdf_hashes[pdf_hash] = collection_name
            logger.info(f"Associated PDF hash {pdf_hash} with collection {collection_name}")
        except Exception as hash_err:
            logger.error(f"Failed to calculate or store PDF hash: {hash_err}")

        return vector_db_instance

    except fitz.fitz.FileDataError:
         logger.error(f"Invalid or corrupted PDF file provided.")
         handle_error("The uploaded file appears to be invalid or corrupted. Please try a different PDF.", False)
         return None
    except Exception as e:
        logger.error(f"Error creating vector DB: {e}", exc_info=True)
        # Clean up temp file on error
        if 'path' in locals() and os.path.exists(path):
             try:
                 os.remove(path)
             except Exception as rm_err:
                 logger.error(f"Failed to remove temporary PDF file {path}: {rm_err}")
        return None
    finally:
        # Clean up the main temp directory *only if* vector DB creation failed entirely
        # If successful, keep the temp dir as images might be stored there by path
        if vector_db_instance is None and os.path.exists(temp_dir):
             try:
                 # shutil.rmtree(temp_dir) # Be cautious with this if images live here
                 logger.info(f"Keeping temporary directory {temp_dir} for potential image access, even on DB failure.")
             except Exception as rm_err:
                 logger.error(f"Failed to clean up main temp directory {temp_dir}: {rm_err}")
        elif vector_db_instance is not None:
             logger.info(f"Keeping temporary directory {temp_dir} containing source PDF and potentially images.")


def get_pdf_context(question: str, vector_db_local: Chroma, model: str) -> str:
    # Renamed vector_db to vector_db_local to avoid conflict with global
    if not vector_db_local: return ""

    try:
        progress = gr.Progress(track_tqdm=True) # Enable tqdm tracking if available
        progress(0.1, desc="Analyzing PDF content...")

        llm = ChatOllama(model=model, temperature=0.1) # Lower temperature for focused queries

        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""Generate 3 diverse and specific search queries based on the core topic "{question}".
            These queries should aim to retrieve key facts, figures, definitions, and distinct concepts from a document relevant to this topic, suitable for building a presentation outline.
            Focus on factual information extraction. Return only the 3 queries, one per line.""",
        )

        retriever = MultiQueryRetriever.from_llm(
            vector_db_local.as_retriever(search_kwargs={"k": 5}), # Retrieve more chunks initially
            llm,
            prompt=QUERY_PROMPT
        )

        progress(0.4, desc="Searching relevant document sections...")
        documents = retriever.invoke(question) # Use invoke for newer Langchain versions

        if not documents:
            logger.warning("MultiQueryRetriever returned no documents.")
            return ""

        progress(0.7, desc="Formatting context...")
        # Deduplicate and format context
        unique_contents = {}
        for doc in documents:
             content_key = doc.page_content[:200] # Use start of content as rough key
             if content_key not in unique_contents:
                 page_info = f"Source Page {doc.metadata.get('page', 'N/A')}" if hasattr(doc, 'metadata') else "Source Section"
                 content = doc.page_content.strip()
                 # Optional: Truncate long chunks if needed, but maybe let LLM handle it
                 # if len(content) > 1500: content = content[:1497] + "..."
                 unique_contents[content_key] = f"--- {page_info} ---\n{content}"


        context = "\n\n".join(unique_contents.values())
        progress(1.0, desc="Document analysis complete")
        logger.info(f"Retrieved {len(unique_contents)} unique relevant document sections for context.")
        return context

    except Exception as e:
        logger.error(f"Error retrieving PDF context: {e}", exc_info=True)
        return "" # Return empty string on error


def check_pdf_similarity(pdf_bytes):
    global pdf_hashes # Use global hash dict
    if not os.path.exists(PERSIST_DIRECTORY):
         logger.info("Persistence directory doesn't exist, skipping similarity check.")
         return False, None, 0

    try:
        current_pdf_hash = get_pdf_hash(pdf_bytes)
        if current_pdf_hash in pdf_hashes:
             collection_name = pdf_hashes[current_pdf_hash]
             logger.info(f"Exact PDF hash match found! Using collection: {collection_name}")
             return True, collection_name, 1.0 # Exact match is 100% similar

        # If hash not found, proceed with content similarity (more expensive)
        logger.info("No exact hash match. Proceeding with content similarity check...")
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_file.write(pdf_bytes)
        temp_file.close() # Close file handle before loader uses it

        sample_text = ""
        try:
            loader = PyMuPDFLoader(temp_file.name)
            # Load only first few pages for sampling to speed up
            data = loader.load() # Consider loader.load(max_pages=5) if available/needed
            if data:
                sample_text = " ".join([page.page_content for page in data[:min(5, len(data))]])
            else:
                 logger.warning("No text content extracted from PDF for similarity check.")
                 return False, None, 0
        except Exception as load_err:
            logger.error(f"Error loading PDF for similarity check: {load_err}")
            return False, None, 0
        finally:
            os.unlink(temp_file.name) # Clean up temp file

        if len(sample_text) < 200: # Increase threshold for meaningful comparison
            logger.warning(f"PDF text sample too short ({len(sample_text)} chars) for reliable similarity check.")
            return False, None, 0

        # List existing collections (more robustly)
        try:
            client = Chroma(persist_directory=PERSIST_DIRECTORY) # Connect to the client instance
            collections = client.list_collections()
            pdf_collections = [c.name for c in collections if c.name.startswith("pdf_")]
            logger.info(f"Found {len(pdf_collections)} existing PDF collections to check.")
        except Exception as list_err:
             logger.error(f"Could not list Chroma collections: {list_err}")
             return False, None, 0 # Cannot perform check if collections can't be listed

        if not pdf_collections:
             logger.info("No existing PDF collections found for similarity check.")
             return False, None, 0

        text_embeddings = OllamaEmbeddings(model="nomic-embed-text")
        highest_similarity = 0
        most_similar_collection = None

        progress = gr.Progress(desc="Checking PDF similarity...", track_tqdm=True)
        for i, collection_name in enumerate(pdf_collections):
            progress(i / len(pdf_collections))
            try:
                db = Chroma(
                    collection_name=collection_name,
                    persist_directory=PERSIST_DIRECTORY,
                    embedding_function=text_embeddings
                )
                # Use query embeddings for similarity check for consistency
                results = db.similarity_search_with_score(sample_text, k=3)
                if results:
                    # Average similarity of top results (score is distance, convert to similarity)
                    avg_similarity = sum([1.0 - score for _, score in results]) / len(results)
                    logger.debug(f"Collection {collection_name} avg similarity: {avg_similarity:.4f}")
                    if avg_similarity > highest_similarity:
                        highest_similarity = avg_similarity
                        most_similar_collection = collection_name
            except Exception as e:
                logger.warning(f"Error checking similarity against collection {collection_name}: {e}")
                continue
        progress(1.0)

        threshold = 0.85 # Keep threshold relatively high for content similarity
        if highest_similarity > threshold:
            logger.info(f"Similar PDF found! Collection: {most_similar_collection}, Similarity: {highest_similarity:.4f}")
            # Store hash mapping for this newly found similarity
            pdf_hashes[current_pdf_hash] = most_similar_collection
            return True, most_similar_collection, highest_similarity

        logger.info(f"No sufficiently similar PDF found (Threshold: {threshold}). Highest similarity: {highest_similarity:.4f}")
        return False, None, highest_similarity

    except Exception as e:
        logger.error(f"Error checking PDF similarity: {e}", exc_info=True)
        return False, None, 0

# --- Presentation Generation & Download ---
def generate_slide_deck(json_str: str, current_download_path: Optional[str]) -> Union[pathlib.Path, None]:
    """Create/Update slide deck and return path. Uses current_download_path if refining."""
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = None

    try:
        parsed_data = json5.loads(json_str)
    except Exception as json_err:
        logger.error(f"Initial JSON parsing failed: {json_err}")
        handle_error('Encountered error while parsing JSON... attempting to fix')
        try:
            fixed_json = text_helper.fix_malformed_json(json_str)
            if fixed_json:
                parsed_data = json5.loads(fixed_json)
                logger.info("Successfully parsed JSON after fixing.")
            else:
                 raise ValueError("JSON fixing failed.")
        except Exception as fix_err:
            logger.error(f"JSON fixing also failed: {fix_err}")
            handle_error(
                'Could not parse the generated content even after attempting fixes. '
                'The slide deck cannot be created. Please try modifying your prompt or try again later.'
            )
            return None # Critical failure

    # Determine output path
    if current_download_path and os.path.exists(current_download_path) and is_refinement:
        # Overwrite existing file if refining
        output_path = pathlib.Path(current_download_path)
        logger.info(f"Refining existing presentation file: {output_path}")
    else:
        # Create new file
        file_name = f"presentation_{timestamp_str}.pptx"
        temp_dir = tempfile.mkdtemp(prefix="pptxgen_") # Unique prefix
        output_path = pathlib.Path(os.path.join(temp_dir, file_name))
        logger.info(f"Creating new presentation file: {output_path}")


    try:
        logger.info(f'Generating PPTX content...')
        pptx_helper.generate_powerpoint_presentation(
            slides_data=parsed_data,
            slides_template=pptx_template, # Use global template selection
            output_file_path=output_path,
            image_query_func=get_image_for_slide # Pass the image getter function
        )

        if output_path.exists() and output_path.stat().st_size > 100: # Check size > 100 bytes
            logger.info(f"Successfully created/updated presentation at {output_path}")
            return output_path # Return the path (string or Path object)
        else:
            logger.error(f"Presentation file generation failed or file is empty: {output_path}")
            return None

    except Exception as ex:
        logger.error(f'Error during PowerPoint generation: {ex}', exc_info=True)
        handle_error(APP_TEXT['content_generation_error'])
        return None


def download_presentation(current_download_path):
    """Return the presentation file path for download."""
    if current_download_path and os.path.exists(current_download_path):
        return current_download_path # Return the actual path Gradio File component needs
    return None # Return None if path is invalid


# --- NEW Google Drive Upload Function (Uses User Credentials) ---
def upload_to_gdrive(file_path: str, user_credentials) -> Optional[Tuple[str, str]]:
    """
    Upload a file to the logged-in user's Google Drive using their credentials.
    (This function is called by the Flask backend, not directly by Gradio UI)

    :param file_path: Path to the file to upload
    :param user_credentials: The OAuth2 credentials object for the logged-in user.
    :return: Tuple of (file_id, view_link) or None on error
    """
    if not user_credentials:
        logger.error("Google Drive upload failed: Missing user credentials.")
        return None
    if not file_path or not os.path.exists(file_path):
        logger.error(f"Google Drive upload failed: File not found at {file_path}")
        return None

    try:
        drive_service = build('drive', 'v3', credentials=user_credentials)
        logger.info(f"Built Drive service using user credentials for file: {file_path}")

        file_metadata = {'name': os.path.basename(file_path)}
        mime_type = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        media = MediaFileUpload(file_path, mimetype=mime_type, resumable=True)

        logger.info(f"Uploading {file_path} to user's Google Drive...")
        file = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id,webViewLink'
        ).execute()

        file_id = file.get('id')
        view_link = file.get('webViewLink')
        logger.info(f"File uploaded successfully to user's Google Drive. File ID: {file_id}, Link: {view_link}")
        return file_id, view_link

    except Exception as e:
        logger.error(f"Error uploading to user's Google Drive: {e}", exc_info=True)
        # Handle specific errors like invalid credentials if possible
        # For example: if 'invalid_grant' in str(e): logger.error("Credentials may be expired or revoked.")
        return None

# --- Gradio UI Interaction Logic ---
def get_user_messages(history) -> List[str]:
    # ... (giữ nguyên code)
    return [msg[0] for msg in history if msg and msg[0] is not None]

def get_last_response(history) -> str:
    # ... (giữ nguyên code)
    if not history: return ""
    for i in range(len(history) - 1, -1, -1):
         msg_pair = history[i]
         if msg_pair and len(msg_pair) > 1 and msg_pair[1] is not None:
              # Check if it's just an error message or actual content
              if isinstance(msg_pair[1], str) and "{" in msg_pair[1] and "}" in msg_pair[1]:
                   return msg_pair[1] # Assume it's JSON content
    return "" # Return empty if no valid previous response found


def determine_refinement(history) -> bool:
    """More robust check if it's a refinement based on valid responses."""
    # A refinement happens if there's at least one user message AND
    # at least one valid-looking previous AI response (contains JSON structure).
    has_user_message = any(msg and msg[0] is not None for msg in history)
    has_valid_response = bool(get_last_response(history)) # Uses the improved getter
    return has_user_message and has_valid_response


# --- Main Message Processing Function ---
def process_message(message: str, history: List, use_pdf_context: bool, current_llm: str, current_path: Optional[str]) -> tuple[str, List, Optional[str]]:
    """
    Process user message, generate response/presentation, and update state.

    :return: (Empty message, updated history, updated file path or None)
    """
    global is_refinement, vector_db, used_pdf_images, used_pexels_images # Access globals needed

    # Reset image usage tracking for a new presentation request
    # Determine if this is the start of a new request vs refinement
    is_new_request = not determine_refinement(history)
    if is_new_request:
        logger.info("New presentation request detected, resetting image usage.")
        used_pdf_images = set()
        used_pexels_images = set()
        # Reset refinement flag for clarity, although determine_refinement handles logic
        is_refinement = False
        # Reset download path for new request
        current_path = None
    else:
         is_refinement = True # Mark as refinement if not new
         logger.info("Refinement request detected.")


    if not message.strip():
        return "", history, current_path # No input, return current state

    # Validate inputs (using the globally set LLM now passed in)
    provider, llm_name = llm_helper.get_provider_model(current_llm, use_ollama=True)
    valid, error_msg = are_all_inputs_valid(message, provider, llm_name)
    if not valid:
        history.append((message, error_msg))
        return "", history, current_path # Return current path on validation error

    logger.info(f'Input: "{message}" | LLM: {llm_name} | PDF: {use_pdf_context} | Refine: {is_refinement}')

    # Add user message to history temporarily
    history.append((message, None))
    # Use yield to update UI immediately with user message
    # yield "", history, current_path # Yield intermediate state

    pdf_context = ""
    if use_pdf_context and vector_db:
        with gr.Progress(track_tqdm=True) as progress: # Use context manager
            progress(0, desc="Analyzing PDF...") # Start progress bar
            pdf_context = get_pdf_context(message, vector_db, llm_name) # Pass local vector_db
            if pdf_context:
                logger.info(f"PDF context retrieved ({len(pdf_context)} chars)")
            else:
                logger.warning("No relevant PDF context found or vector_db missing.")
    elif use_pdf_context and not vector_db:
         logger.warning("PDF context requested, but no vector database is loaded.")
         # Optionally add a message to the user in history?
         # history[-1] = (message, "Warning: PDF context was requested, but no PDF has been processed.")
         # yield "", history, current_path


    # Determine refinement status *before* formatting prompt
    is_refinement_prompt = determine_refinement(history[:-1]) # Check history *before* adding current message tuple

    # Prepare prompt
    prompt_template_str = get_prompt_template(
        is_refinement_prompt=is_refinement_prompt,
        with_context=bool(pdf_context)
    )
    prompt_template = ChatPromptTemplate.from_template(prompt_template_str)

    # Format prompt based on state
    template_args = {}
    if is_refinement_prompt:
        user_messages = get_user_messages(history[:-1]) # Exclude current msg placeholder
        # Include the current message as the latest instruction
        all_instructions = [f'{idx + 1}. {msg}' for idx, msg in enumerate(user_messages)]
        all_instructions.append(f'{len(all_instructions) + 1}. {message}') # Add current message
        template_args['instructions'] = '\n'.join(all_instructions)
        template_args['previous_content'] = get_last_response(history[:-1]) # Get last valid JSON
        if not template_args['previous_content']:
             logger.error("Refinement requested, but no previous valid content found!")
             history[-1] = (message, "Error: Cannot refine, no previous presentation content available.")
             return "", history, current_path
    else:
        template_args['question'] = message

    if pdf_context:
        template_args['context'] = pdf_context

    # Final formatted prompt string for logging/debugging
    try:
         # Use format_prompt for ChatPromptTemplate if available and returns PromptValue
         # prompt_value = prompt_template.format_prompt(**template_args)
         # formatted_template = prompt_value.to_string()
         # Or simply use format if it works directly for the model
         formatted_template = prompt_template.invoke(template_args).to_string() # For Langchain core > 0.1
    except Exception as format_err:
         logger.error(f"Error formatting prompt template: {format_err}")
         # Fallback to basic f-string formatting if invoke/format_prompt fails
         try:
              formatted_template = prompt_template_str.format(**template_args)
         except KeyError as key_err:
              logger.error(f"Missing key in template arguments: {key_err}")
              history[-1] = (message, f"Error: Internal issue formatting prompt (missing key: {key_err}).")
              return "", history, current_path


    # Call LLM
    response_content = ""
    llm_error = None
    try:
        llm = llm_helper.get_langchain_llm(
            provider=provider, model=llm_name, max_new_tokens=8192 # Increase tokens for potentially larger JSON
        )
        if not llm:
            raise ValueError("Failed to initialize LLM instance.")

        logger.info("Streaming response from LLM...")
        # Stream response directly to history (Gradio handles streaming display)
        history[-1] = (message, "") # Initialize empty response string
        # Use Gradio streaming: https://www.gradio.app/guides/streaming-outputs
        stream = llm.stream(formatted_template)
        for chunk in stream:
            chunk_content = chunk if isinstance(chunk, str) else chunk.content
            response_content += chunk_content
            history[-1] = (message, response_content) # Update last history item
            # No yield here, Gradio handles UI update from history change implicitly for Chatbot
            # yield "", history, current_path # Don't yield inside the stream loop usually

    except ollama.ResponseError as ore:
        llm_error = (
            f'Ollama Error: The model `{llm_name}` seems unavailable or encountered an issue. '
            f'Details: {ore}. Make sure Ollama is running and the model is pulled (`ollama pull {llm_name}`).'
        )
    except Exception as ex:
        llm_error = (
            f'An unexpected error occurred while communicating with the LLM: {ex}'
        )

    if llm_error:
        logger.error(llm_error)
        history[-1] = (message, llm_error) # Show error in chat
        return "", history, current_path # Return current path on LLM error

    # --- Post-processing and Slide Generation ---
    logger.info(f"LLM response received (length: {len(response_content)}).")
    cleaned_json = text_helper.get_clean_json(response_content)
    if not cleaned_json:
        logger.error("Failed to extract clean JSON from LLM response.")
        history[-1] = (message, "Error: The LLM response did not contain valid presentation data structure.")
        return "", history, current_path

    logger.info("Generating slide deck from cleaned JSON...")
    # Use current_path if refining, otherwise it's None for new generation
    new_path = generate_slide_deck(cleaned_json, current_path if is_refinement else None)

    if new_path:
        logger.info(f"Slide deck generated/updated successfully: {new_path}")
        # Update history with the final *cleaned* JSON for clarity, or keep raw response?
        # Let's keep the cleaned JSON for potential future refinements
        history[-1] = (message, cleaned_json)
        # Return the NEW path (whether it's a new file or the existing one)
        return "", history, str(new_path)
    else:
        logger.error("Slide deck generation failed.")
        history[-1] = (message, "Error: Failed to generate the PowerPoint file from the LLM response.")
        # Keep the old path if generation failed during refinement? Or reset?
        # Reset path if generation fails to avoid downloading broken file.
        return "", history, None


# --- PDF Upload and Status Functions ---
def upload_pdf(file_obj):
    """Handles PDF upload, similarity check, and vector DB creation."""
    global vector_db, pdf_hashes, pdf_images # Allow modification of globals

    if file_obj is None:
        vector_db = None # Reset DB if no file is uploaded
        pdf_images = [] # Clear images
        return "No file provided.", gr.update(value="", visible=False), gr.update(value=False, interactive=False) # Message, PDF info, Use PDF checkbox

    pdf_bytes = file_obj # Gradio 'binary' type gives bytes
    status_message = "Processing PDF..."
    collection_info = ""
    collection_visible = False
    enable_use_pdf = False

    try:
        # 1. Check Similarity
        similar_found, collection_name, similarity = check_pdf_similarity(pdf_bytes)

        if similar_found and collection_name:
             similarity_pct = f"{similarity * 100:.1f}%"
             status_message = f"✅ Similar PDF ({similarity_pct}) found! Using existing data: {collection_name}"
             logger.info(f"Loading existing Chroma collection: {collection_name}")
             try:
                 # Ensure embedding function matches what was used before
                 text_embeddings = OllamaEmbeddings(model="nomic-embed-text")
                 vector_db = Chroma(
                     collection_name=collection_name,
                     persist_directory=PERSIST_DIRECTORY,
                     embedding_function=text_embeddings
                 )
                 # Try to load associated images if stored separately (complex)
                 # For now, we might need to re-extract images if using existing DB
                 # Let's re-extract images even if DB exists for consistency here
                 logger.info("Re-extracting images for the loaded similar PDF...")
                 # Need to save bytes to temp file to extract
                 temp_pdf_for_images = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                 temp_pdf_for_images.write(pdf_bytes)
                 temp_pdf_for_images.close()
                 extract_and_embed_images(temp_pdf_for_images.name)
                 os.remove(temp_pdf_for_images.name)

                 info_md, info_visible = get_vector_db_info() # Update info display
                 collection_info = info_md
                 collection_visible = info_visible.get('visible', False)
                 enable_use_pdf = True

             except Exception as load_err:
                 logger.error(f"Failed to load existing collection '{collection_name}': {load_err}")
                 status_message = f"⚠️ Found similar PDF, but failed to load its data. Reprocessing as new."
                 vector_db = None # Reset DB
                 pdf_images = [] # Clear images
                 # Fall through to create new DB

        # 2. Process as New if Not Similar or Loading Failed
        if not vector_db:
             status_message = "⏳ Processing new PDF... (This may take a moment)"
             # Yield intermediate status? Might be complex with multiple outputs
             # yield status_message, gr.update(value="", visible=False), gr.update(value=False, interactive=False)

             vector_db = create_vector_db(pdf_bytes) # This now also extracts images globally
             if vector_db:
                 status_message = "✅ Successfully processed new PDF."
                 info_md, info_visible = get_vector_db_info()
                 collection_info = info_md
                 collection_visible = info_visible.get('visible', False)
                 enable_use_pdf = True
             else:
                 status_message = "❌ Error processing PDF. Check logs. Please try another file."
                 vector_db = None # Ensure DB is None on failure
                 pdf_images = [] # Clear images
                 enable_use_pdf = False

    except Exception as e:
        logger.error(f"Critical error during PDF upload processing: {e}", exc_info=True)
        status_message = f"❌ Critical Error: {e}"
        vector_db = None
        pdf_images = []
        enable_use_pdf = False

    # Return final state for all output components
    return status_message, gr.update(value=collection_info, visible=collection_visible), gr.update(value=enable_use_pdf, interactive=enable_use_pdf)


def get_vector_db_info():
    """Get formatted info about the active vector DB."""
    global vector_db # Access the global DB state
    if not vector_db:
        return "", gr.update(visible=False)

    try:
        collection_name = "Unknown"
        count = "N/A"
        if hasattr(vector_db, '_collection'):
            collection = vector_db._collection
            if hasattr(collection, 'name'):
                collection_name = collection.name
            if hasattr(collection, 'count') and callable(collection.count):
                try:
                    count = collection.count()
                except Exception as count_err:
                    logger.warning(f"Could not get count for collection {collection_name}: {count_err}")
                    count = "Error"

        # Get embedding function details if possible
        embed_func_info = "Unknown"
        if hasattr(vector_db, '_embedding_function'):
             embed_func = vector_db._embedding_function
             if hasattr(embed_func, 'model'):
                  embed_func_info = f"Ollama ({getattr(embed_func, 'model', 'N/A')})"
             else:
                  embed_func_info = type(embed_func).__name__


        info = f"**PDF Collection:** `{collection_name}`\n"
        info += f"- **Chunks:** {count}\n"
        info += f"- **Embeddings:** {embed_func_info}\n"
        info += f"- **Images Extracted:** {len(pdf_images)}"

        return info, gr.update(visible=True)
    except Exception as e:
        logger.error(f"Error getting vector DB info: {e}")
        return "Error retrieving DB details.", gr.update(visible=True)


# --- UI Event Handlers ---
def template_changed(template_name: str) -> None:
    global pptx_template
    pptx_template = template_name
    logger.info(f"Template changed to: {template_name}")

def llm_changed(model_name: str) -> None:
    global llm_provider_to_use
    # Add basic validation for model name format if desired
    if model_name and ":" in model_name:
         llm_provider_to_use = model_name
         logger.info(f"LLM changed to: {model_name}")
    else:
         # Handle invalid input - maybe revert or show error?
         logger.warning(f"Invalid LLM format entered: {model_name}. Keeping previous: {llm_provider_to_use}")
         # Optionally, update the Gradio component back to the old value
         # return gr.update(value=llm_provider_to_use) # This requires the function to return an update


def create_initial_message() -> List:
    greeting = random.choice(APP_TEXT.get('ai_greetings', ["Hello! How can I help you create a presentation?"]))
    return [(None, greeting)]


def is_download_ready(current_path: Optional[str]) -> bool:
    """Check if download is ready based on valid file path."""
    return bool(current_path and os.path.exists(current_path) and os.path.getsize(current_path) > 100)

# --- NEW Gradio Function to Trigger Flask Backend Upload ---
def trigger_user_drive_upload(current_pptx_path: Optional[str], flask_app_url: str):
    """
    Called by Gradio button. Sends request to Flask backend to upload the file
    using the user's credentials stored in the Flask session.
    """
    if not current_pptx_path or not os.path.exists(current_pptx_path):
        logger.error("Trigger User Drive Upload: Failed, presentation path invalid or missing.")
        return "<div style='color: red;'>Error: Presentation file not found. Cannot upload.</div>", gr.update(visible=True)

    upload_endpoint = f"{flask_app_url.rstrip('/')}/upload_presentation_to_user_drive"
    payload = json.dumps({"file_path": current_pptx_path})
    headers = {'Content-Type': 'application/json'}
    status_message = "📤 Sending request to upload service..."
    show_status = True

    try:
        logger.info(f"Sending POST request to {upload_endpoint} for file {current_pptx_path}")
        # IMPORTANT: Assumes Flask and Gradio run on same host/port or CORS is set up
        # The browser needs to automatically send the Flask session cookie.
        response = requests.post(upload_endpoint, headers=headers, data=payload, timeout=180) # Longer timeout for upload

        logger.info(f"Received response from backend: {response.status_code}")
        response_data = response.json() # Assume backend always returns JSON

        if response.ok and response_data.get("success"):
            link = response_data.get("view_link")
            status_message = f"✅ Successfully uploaded! <a href='{link}' target='_blank' style='color: #4285F4;'>Click here to open in Google Drive</a>."
        elif response.status_code == 401: # Unauthorized
             status_message = "<div style='color: orange;'>⚠️ Authentication failed or expired. Please log out and log back in via the main page.</div>"
        else: # Other errors from backend
            error_msg = response_data.get("error", f"Server responded with status {response.status_code}")
            status_message = f"<div style='color: red;'>❌ Upload Failed: {error_msg}</div>"
            logger.error(f"Backend upload failed: {error_msg}")

    except requests.exceptions.ConnectionError:
        logger.error(f"Connection Error: Could not connect to Flask backend at {upload_endpoint}")
        status_message = "<div style='color: red;'>❌ Connection Error: Cannot reach the upload service. Is the backend running?</div>"
    except requests.exceptions.Timeout:
         logger.error(f"Timeout error connecting to {upload_endpoint}")
         status_message = "<div style='color: red;'>❌ Timeout: The upload request took too long to complete.</div>"
    except requests.exceptions.RequestException as e:
        logger.error(f"Error during request to Flask backend: {e}")
        status_message = f"<div style='color: red;'>❌ Request Error: {e}</div>"
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON response from backend. Status: {response.status_code}, Text: {response.text[:200]}")
        status_message = "<div style='color: red;'>❌ Error: Received invalid response from upload service.</div>"
    except Exception as e:
        logger.error(f"Unexpected error in trigger_user_drive_upload: {e}", exc_info=True)
        status_message = f"<div style='color: red;'>❌ An unexpected error occurred: {e}</div>"

    return status_message, gr.update(visible=show_status)


# --- Function to Create the Gradio UI ---
def create_ui(flask_app_url="http://127.0.0.1:5000") -> gr.Blocks:
    """
    Create the Gradio UI. Includes components for chat, PDF upload,
    settings, and download/upload actions.
    """
    custom_css = """
    #download_file { width: 100%; margin-top: 8px; }
    #gdrive_status { margin-top: 5px; padding: 5px; border-radius: 5px; }
    #pdf_info_box { background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-top: 5px;}
    """
    templates = list(GlobalConfig.PPTX_TEMPLATE_FILES.keys())

    with gr.Blocks(title="AI Presentation Generator", theme=gr.themes.Soft(), css=custom_css) as demo:
        gr.Markdown(f"# {APP_TEXT.get('app_name', 'AI Presentation Generator')}")
        gr.Markdown(f"*{APP_TEXT.get('caption', 'Generate presentations with AI, optionally using PDF context.')}*")

        # Store the path to the latest generated PPTX file
        current_pptx_path_state = gr.State(value=None)

        with gr.Row():
            # --- Left Panel ---
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### Settings")
                    template_dropdown = gr.Dropdown(
                        choices=templates, value=pptx_template, label="Presentation Template", interactive=True
                    )
                    llm_input = gr.Textbox(
                        value=llm_provider_to_use, label="Ollama Model", info="e.g., mistral:latest, llama3:8b", interactive=True
                    )

                with gr.Group():
                    gr.Markdown("### PDF Context (Optional)")
                    pdf_file = gr.File(label="Upload PDF", file_types=[".pdf"], type="binary")
                    pdf_process_btn = gr.Button("Process PDF", variant="secondary", size="sm")
                    pdf_result = gr.Textbox(label="PDF Status", value="No PDF processed.", interactive=False, lines=1)
                    with gr.Box(visible=False, elem_id="pdf_info_box") as pdf_info_box:
                         pdf_info = gr.Markdown(value="")
                    use_pdf = gr.Checkbox(label="Use PDF Content", value=False, interactive=False,
                                          info="Enable this *after* processing a PDF")

                with gr.Accordion("Usage Instructions", open=False):
                    gr.Markdown(GlobalConfig.CHAT_USAGE_INSTRUCTIONS)

                # --- Download/Upload Box ---
                with gr.Group(visible=False) as download_box:
                    gr.Markdown("### Presentation Ready")
                    with gr.Row():
                         # Use gr.DownloadButton for direct download
                         download_btn = gr.DownloadButton("⬇️ Download PPTX", variant="primary")
                         # Button to trigger upload via Flask backend
                         gdrive_upload_btn = gr.Button(" GDrive", icon="https://www.google.com/favicon.ico", variant="secondary") # Icon example

                    gdrive_status = gr.HTML(value="", label="Upload Status", visible=False, elem_id="gdrive_status")

            # --- Right Panel (Chat) ---
            with gr.Column(scale=2): # Make chat wider
                chatbot = gr.Chatbot(
                    value=create_initial_message, # Use function for initial value
                    label="Chat",
                    height=650,
                    show_copy_button=True,
                    bubble_full_width=False # Improve layout
                )
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder=APP_TEXT.get('chat_placeholder', "Enter your presentation topic or refinement instructions..."),
                        show_label=False, scale=4 # Give more space to text input
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)
                with gr.Row():
                     clear_btn = gr.Button("Clear Chat & Reset")


        # --- Define Event Handlers ---

        # Settings Change Handlers
        template_dropdown.change(template_changed, inputs=[template_dropdown], outputs=None)
        # Use submit instead of change for textbox to avoid firing on every keystroke
        llm_input.submit(llm_changed, inputs=[llm_input], outputs=None) # Or use .blur

        # PDF Processing Handler
        pdf_process_btn.click(
            upload_pdf,
            inputs=[pdf_file],
            outputs=[pdf_result, pdf_info_box, use_pdf] # Update status, info box visibility, and checkbox
        )

        # Chat Submission Handlers (handle both Enter and Button click)
        submit_action = [msg, chatbot, use_pdf, llm_input, current_pptx_path_state]
        submit_result = [msg, chatbot, current_pptx_path_state]

        submit_btn.click(process_message, inputs=submit_action, outputs=submit_result, concurrency_limit=1 # Limit concurrency
                         ).then(
                             # After processing, check if download is ready and update visibility
                             lambda path: gr.update(visible=is_download_ready(path)),
                             inputs=[current_pptx_path_state], outputs=[download_box]
                         )

        msg.submit(process_message, inputs=submit_action, outputs=submit_result, concurrency_limit=1
                   ).then(
                       lambda path: gr.update(visible=is_download_ready(path)),
                       inputs=[current_pptx_path_state], outputs=[download_box]
                   )

        # Download Button Handler (uses DownloadButton's built-in functionality)
        download_btn.click(
             download_presentation,
             inputs=[current_pptx_path_state],
             outputs=[download_btn] # Target the button itself for download action
        )

        # Google Drive Upload Button Handler
        gdrive_upload_btn.click(
            lambda path: trigger_user_drive_upload(path, flask_app_url), # Pass Flask URL
            inputs=[current_pptx_path_state],
            outputs=[gdrive_status, gdrive_status] # Update status HTML and visibility
        )

        # Clear Button Handler
        def clear_chat_and_reset():
            global vector_db, pdf_images, used_pdf_images, used_pexels_images, is_refinement
            vector_db = None
            pdf_images = []
            used_pdf_images = set()
            used_pexels_images = set()
            is_refinement = False
            logger.info("Chat cleared and state reset.")
            return (
                create_initial_message(), # New chat history
                "", # Clear message box
                None, # Clear pptx path state
                gr.update(visible=False), # Hide download box
                "No PDF processed.", # Reset PDF status
                gr.update(value="", visible=False), # Hide PDF info box
                gr.update(value=False, interactive=False) # Reset 'Use PDF' checkbox
            )
        clear_btn.click(
             clear_chat_and_reset,
             inputs=None,
             outputs=[chatbot, msg, current_pptx_path_state, download_box, pdf_result, pdf_info_box, use_pdf]
        )

    return demo

# --- Main execution handled by Flask wrapper ---
# if __name__ == "__main__":
#     # This part is removed as Flask will create and run the UI
#     pass