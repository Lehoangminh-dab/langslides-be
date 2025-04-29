# -*- coding: utf-8 -*-
import os
import logging
from flask import Flask, url_for, session, redirect, request, jsonify, render_template_string
from authlib.integrations.flask_client import OAuth
from dotenv import load_dotenv
import gradio as gr
from werkzeug.middleware.proxy_fix import ProxyFix # Important for deployment behind proxy

# --- Import Gradio UI creation function ---
# Make sure gradio_app.py is in the same directory or Python path
import gradio_app
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request as GoogleAuthRequest # Alias to avoid naming conflict

# --- Load Environment Variables ---
load_dotenv()

# --- Basic Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
flask_logger = logging.getLogger("flask_app")

# --- Flask App Initialization ---
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY') # MUST set in .env
# Apply ProxyFix if running behind a proxy (like Nginx) for correct URL generation (https)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)


# --- Authlib OAuth Configuration ---
oauth = OAuth(app)
GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID')
GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET')

if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
    flask_logger.error("FATAL: GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET environment variables not set!")
    # exit(1) # Or handle more gracefully

google = oauth.register(
    name='google',
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    access_token_url='https://accounts.google.com/o/oauth2/token',
    access_token_params=None,
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    authorize_params=None,
    api_base_url='https://www.googleapis.com/oauth2/v1/',
    userinfo_endpoint='https://openidconnect.googleapis.com/v1/userinfo',
    # IMPORTANT: Request 'offline' access to get a refresh token
    # Request drive.file scope for uploading
    client_kwargs={
        'scope': 'openid email profile https://www.googleapis.com/auth/drive.file',
        'prompt': 'consent' # Force consent screen for refresh token on first login
        },
    jwks_uri="https://www.googleapis.com/oauth2/v3/certs",
)

# --- Helper Function for Credentials (Essential for using google-api-python-client) ---
def get_valid_user_credentials_from_session(session_data):
    """
    Reconstructs Google Credentials object from session token.
    Handles token refresh if necessary.
    Returns Credentials object or None.
    """
    token_dict = session_data.get('user_token')
    if not token_dict:
        flask_logger.warning("No user_token found in session.")
        return None

    try:
        # Reconstruct credentials using info stored by Authlib
        creds = Credentials(
            token=token_dict.get('access_token'),
            refresh_token=token_dict.get('refresh_token'),
            token_uri=google.access_token_url, # From Authlib config
            client_id=GOOGLE_CLIENT_ID,
            client_secret=GOOGLE_CLIENT_SECRET,
            scopes=token_dict.get('scope', '').split() # Scopes might be slightly different format
        )

        # Check expiration and refresh if needed
        # Add a small buffer (e.g., 60 seconds) to expiry check
        if creds.expired and creds.refresh_token:
             flask_logger.info("Credentials expired, attempting refresh...")
             try:
                 creds.refresh(GoogleAuthRequest()) # Use the aliased request
                 # IMPORTANT: Update the session with the new token info!
                 session_data['user_token']['access_token'] = creds.token
                 # Authlib might store expiry differently, adjust as needed
                 if creds.expiry:
                      session_data['user_token']['expires_at'] = creds.expiry.timestamp()
                 flask_logger.info("Token refreshed successfully.")
             except Exception as refresh_error:
                 flask_logger.error(f"Failed to refresh token: {refresh_error}", exc_info=True)
                 # Clear potentially invalid token? Or let user re-login?
                 session.pop('user_token', None) # Clear invalid token
                 session.pop('user', None)
                 return None # Refresh failed

        elif not creds.valid:
             # Token might be invalid for other reasons (e.g., revoked)
             flask_logger.warning("Credentials are not valid (but not expired/no refresh token).")
             # Force re-login might be needed
             session.pop('user_token', None)
             session.pop('user', None)
             return None

        return creds

    except Exception as e:
        flask_logger.error(f"Error reconstructing or validating credentials: {e}", exc_info=True)
        return None


# --- Flask Routes ---
@app.route('/')
def index():
    """Main page: Shows login or the Gradio app."""
    user = session.get('user')
    if not user:
        # Simple login page
        return render_template_string("""
            <!DOCTYPE html>
            <html>
            <head><title>Login</title></head>
            <body>
                <h1>Welcome to AI Presentation Generator</h1>
                <p>Please log in with your Google Account to continue.</p>
                <a href="/login" style="padding: 10px 20px; background-color: #4285F4; color: white; text-decoration: none; border-radius: 5px;">
                    Login with Google
                </a>
            </body>
            </html>
        """)
    else:
        # User is logged in, show welcome and Gradio app will be mounted at /gradio
        user_email = user.get('email', 'User')
        user_name = user.get('given_name', user_email) # Use given name if available
        # Note: The Gradio app is usually accessed via its mount path, e.g., /gradio
        # This basic template just confirms login before redirecting or user navigates.
        return render_template_string("""
            <!DOCTYPE html>
            <html>
            <head><title>AI Presentation Generator</title></head>
            <body>
                <div style="padding: 10px; background-color: #f0f0f0; border-bottom: 1px solid #ccc; margin-bottom: 15px; display: flex; justify-content: space-between; align-items: center;">
                   <span>Welcome, {{ user_name }}!</span>
                   <a href="/logout" style="color: #dc3545; text-decoration: none;">Logout</a>
                </div>
                <p>Loading the presentation generator...</p>
                <p>If it doesn't load automatically, <a href="/gradio">click here to access the app</a>.</p>
                <iframe src="/gradio" width="100%" height="800px" style="border:none;"></iframe>
            </body>
            </html>
        """, user_name=user_name)

@app.route('/login')
def login():
    """Redirects user to Google for authentication."""
    # Ensure redirect_uri matches exactly what's in Google Cloud Console
    # Use _external=True for absolute URL needed by Google
    redirect_uri = url_for('authorize', _external=True)
    flask_logger.info(f"Redirecting to Google for login. Callback URI: {redirect_uri}")
    # Add prompt='consent' if refresh token isn't consistently granted
    return google.authorize_redirect(redirect_uri)
    # return google.authorize_redirect(redirect_uri, prompt='consent')


@app.route('/authorize') # This must match the Redirect URI in Google Cloud Console
def authorize():
    """Handles the callback from Google after user authentication."""
    try:
        # Fetch token using the authorization code Google sends back
        token = google.authorize_access_token()
        # Fetch user profile info using the obtained token
        # resp = google.get('userinfo') # Older way
        # resp.raise_for_status()
        # user_info = resp.json()
        user_info = oauth.google.userinfo(token=token) # Use Authlib's built-in method

        # Store token and user info in session
        session['user_token'] = token
        session['user'] = user_info
        flask_logger.info(f"User authorized successfully: {user_info.get('email')}")

    except Exception as e:
        flask_logger.error(f"Error during Google OAuth callback: {e}", exc_info=True)
        # Redirect to an error page or show message
        return "Authentication failed during callback.", 400

    # Redirect to the main page after successful login
    return redirect('/')

@app.route('/logout')
def logout():
    """Logs the user out by clearing the session."""
    flask_logger.info(f"User logging out: {session.get('user', {}).get('email')}")
    session.pop('user_token', None)
    session.pop('user', None)
    # Redirect to main page, which will now show the login prompt
    return redirect('/')


# --- API Endpoint for Gradio to Trigger Upload ---
@app.route('/upload_presentation_to_user_drive', methods=['POST'])
def handle_upload_to_user_drive():
    """Endpoint called by Gradio to upload file using user's session credentials."""
    if 'user' not in session:
        flask_logger.warning("Upload attempt failed: User not logged in.")
        return jsonify({"error": "User not logged in."}), 401

    user_email = session.get('user', {}).get('email', 'Unknown User')
    flask_logger.info(f"Received upload request from user: {user_email}")

    data = request.get_json()
    file_path_to_upload = data.get('file_path') if data else None

    if not file_path_to_upload or not os.path.exists(file_path_to_upload):
        flask_logger.warning(f"Upload request from {user_email} failed: file path '{file_path_to_upload}' is invalid or missing.")
        return jsonify({"error": f"Presentation file path missing or invalid: {file_path_to_upload}"}), 400

    try:
        # --- Get valid credentials (handles refresh) ---
        user_credentials = get_valid_user_credentials_from_session(session)

        if not user_credentials:
            flask_logger.warning(f"Upload request from {user_email} failed: Could not obtain valid credentials.")
            # Maybe token needs refresh and failed, or login is truly invalid
            return jsonify({"error": "Authentication invalid or expired. Please log out and log in again."}), 401

        # --- Call the upload function (imported from gradio_app) ---
        flask_logger.info(f"Calling upload_to_gdrive for user {user_email}, file: {file_path_to_upload}")
        result = gradio_app.upload_to_gdrive(file_path_to_upload, user_credentials)

        if result:
            file_id, view_link = result
            flask_logger.info(f"Successfully uploaded file {file_path_to_upload} for user {user_email}. File ID: {file_id}")
            return jsonify({"success": True, "view_link": view_link, "file_id": file_id})
        else:
            # The upload_to_gdrive function should have logged the specific error
            flask_logger.error(f"gradio_app.upload_to_gdrive function failed for user {user_email}, file: {file_path_to_upload}")
            return jsonify({"error": "Failed to upload file to your Google Drive. Check server logs for details."}), 500

    except Exception as e:
        flask_logger.error(f"Exception during user drive upload handling for {user_email}: {e}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred during upload: {str(e)}"}), 500


# --- Create and Mount Gradio App ---
flask_logger.info("Creating Gradio UI...")
# Pass the base URL of the Flask app so Gradio knows where to send API calls
flask_base_url = "http://127.0.0.1:5000" # Change if running on different host/port
gradio_ui = gradio_app.create_ui(flask_app_url=flask_base_url)

# Mount the Gradio app onto the Flask app at the /gradio path
# Use app=app (Flask instance), blocks=gradio_ui (Gradio Blocks), path="/gradio"
# Ensure allow_embedding is True if you plan to iframe it as in the index route
app = gr.mount_gradio_app(app, gradio_ui, path="/gradio")
flask_logger.info("Gradio UI mounted at /gradio")


# --- Run Flask App ---
if __name__ == '__main__':
    # Use host='0.0.0.0' to make it accessible on your network
    # Debug=True is useful for development, REMOVE for production
    # Choose a port (e.g., 5000)
    flask_logger.info(f"Starting Flask server on {flask_base_url}")
    app.run(host='0.0.0.0', port=5000, debug=True)