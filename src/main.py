# main.py
import json
import os
import pathlib
import sqlite3
import requests
import datetime # Thêm nếu dùng trong /api/status

# Internal imports
# Đảm bảo các file này tồn tại trong cùng thư mục hoặc PYTHONPATH
from db import init_app as init_db_app
from user import User

# Third party libraries
from flask import Flask, redirect, request, url_for, session, abort, jsonify, render_template # Thêm render_template
from flask_login import (
    LoginManager,
    current_user,
    login_required,
    login_user,
    logout_user,
)
from oauthlib.oauth2 import WebApplicationClient
from dotenv import load_dotenv # Thêm import dotenv
# Dùng cachecontrol đã cài đặt
from cachecontrol import CacheControl
from google.auth.transport.requests import Request as GoogleAuthRequest

load_dotenv() # Gọi load_dotenv ở đầu

# Configuration
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", None)
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", None)
GOOGLE_DISCOVERY_URL = (
    "https://accounts.google.com/.well-known/openid-configuration"
)

# Flask app setup
app = Flask(__name__) # Thư mục template mặc định là 'templates'
app.secret_key = os.environ.get("SECRET_KEY") or os.urandom(24)
if not os.environ.get("SECRET_KEY"):
    print("Cảnh báo: Biến môi trường SECRET_KEY chưa được đặt, đang sử dụng key ngẫu nhiên tạm thời.")

# User session management setup using Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)

# Đăng ký các hàm quản lý DB với app
init_db_app(app)

@login_manager.unauthorized_handler
def unauthorized():
    return "You must be logged in to access this content.", 401

# OAuth2 client setup
if not GOOGLE_CLIENT_ID:
    raise ValueError("GOOGLE_CLIENT_ID không được đặt trong môi trường.")
client = WebApplicationClient(GOOGLE_CLIENT_ID)

# Flask-Login helper to retrieve a user from our db
@login_manager.user_loader
def load_user(user_id):
    """Callback được Flask-Login sử dụng để tải user từ ID trong session."""
    return User.get(user_id)

# --- Routes ---

@app.route("/")
def index():
    """Hiển thị trang chủ."""
    if current_user.is_authenticated:
        return render_template('index.html', logged_in=True, current_user=current_user)
    else:
        return render_template('index.html', logged_in=False)

@app.route("/login")
def login():
    """Bắt đầu luồng đăng nhập Google."""
    google_provider_cfg = get_google_provider_cfg()
    authorization_endpoint = google_provider_cfg["authorization_endpoint"]
    redirect_uri = url_for("callback", _external=True)
    print(f"Login redirect URI: {redirect_uri}")
    request_uri = client.prepare_request_uri(
        authorization_endpoint,
        redirect_uri=redirect_uri,
        scope=["openid", "email", "profile"],
        # access_type="offline", # Có thể thêm nếu muốn refresh token
        # prompt="consent" # Có thể thêm nếu muốn luôn hiện màn hình đồng ý
    )
    print(f"Redirecting user to: {request_uri}")
    return redirect(request_uri)


@app.route("/login/callback")
def callback():
    """Xử lý callback từ Google."""
    code = request.args.get("code")
    if not code:
        return "Error: Missing authorization code.", 400

    # Kiểm tra state (oauthlib thường tự xử lý khi parse token nếu dùng đúng cách)
    # state_from_request = request.args.get("state") # Lấy state nếu cần kiểm tra thủ công

    google_provider_cfg = get_google_provider_cfg()
    token_endpoint = google_provider_cfg["token_endpoint"]

    token_url, headers, body = client.prepare_token_request(
        token_endpoint,
        authorization_response=request.url,
        redirect_url=url_for("callback", _external=True),
        code=code,
        # state=state_from_session # oauthlib dùng state nội bộ, thường không cần truyền lại
    )
    print("Requesting token from Google...")
    try:
        token_response = requests.post(
            token_url,
            headers=headers,
            data=body,
            auth=(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET),
            timeout=10
        )
        token_response.raise_for_status()
        print("Token request successful.")
    except requests.exceptions.RequestException as e:
         print(f"Failed to fetch token: {e}")
         return f"Failed to fetch token: {e}", 500

    try:
        client.parse_request_body_response(json.dumps(token_response.json()))
        # Lấy refresh token nếu có (cần scope offline và user đồng ý)
        # refresh_token = client.token.get('refresh_token')
        # if refresh_token:
        #    print("!!! Received Refresh Token !!!") # Cần lưu vào DB
    except Exception as e:
        print(f"Failed to parse token response: {e}")
        print(f"Token response content: {token_response.text}")
        return f"Failed to parse token response: {e}", 500

    userinfo_endpoint = google_provider_cfg["userinfo_endpoint"]
    uri, headers, body = client.add_token(userinfo_endpoint)
    print("Requesting userinfo from Google...")
    try:
        userinfo_response = requests.get(uri, headers=headers, data=body, timeout=10)
        userinfo_response.raise_for_status()
        userinfo_json = userinfo_response.json()
        print(f"Userinfo request successful{userinfo_json}.")
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch userinfo: {e}")
        return f"Failed to fetch userinfo: {e}", 500

    if not userinfo_json.get("email_verified"):
        return "User email not available or not verified by Google.", 400

    unique_id = userinfo_json.get("sub")
    users_email = userinfo_json.get("email")
    picture = userinfo_json.get("picture")
    users_name = userinfo_json.get("given_name") or userinfo_json.get("name")

    if not all([unique_id, users_email, users_name]):
         print(f"Missing user info: id={unique_id}, email={users_email}, name={users_name}")
         return "Could not retrieve all required user information from Google.", 500

    print(f"User info retrieved: ID={unique_id}, Name={users_name}, Email={users_email}")

    # Logic Update hoặc Create User trong DB
    existing_user = User.get(unique_id)
    if existing_user:
        User.update(unique_id, users_name, users_email, picture)
    else:
        User.create(unique_id, users_name, users_email, picture)

    # Đăng nhập user vào session sử dụng Flask-Login
    user_to_login = User.get(unique_id)
    if user_to_login:
        login_user(user_to_login)
        print(f"User {users_name} logged in via Flask-Login.")
    else:
        print(f"CRITICAL: Could not fetch user {unique_id} right after create/update!")
        return "Login failed due to internal user processing error.", 500

    # Chuyển hướng về trang chủ
    return redirect(url_for("index"))


@app.route("/logout")
@login_required
def logout():
    """Đăng xuất người dùng."""
    print(f"Logging out user: {current_user.name}")
    logout_user()
    return redirect(url_for("index"))


def get_google_provider_cfg():
    """Lấy cấu hình OIDC từ Google Discovery URL."""
    try:
        response = requests.get(GOOGLE_DISCOVERY_URL, timeout=5)
        response.raise_for_status()
        print(f"12343323233{response.json}")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Failed to get Google OIDC discovery document: {e}")
        abort(503, description=f"Could not contact Google Discovery endpoint: {e}")


@app.route("/protected")
@login_required
def protected():
     return f"Hello {current_user.name}! This is a protected area."

# --- Chạy ứng dụng ---
if __name__ == "__main__":
    print("--- Starting Flask Development Server (Flask-Login, OAuthlib, SQLite) ---")

    # Chạy HTTPS trên localhost
    app.run(port=5000, debug=True, ssl_context="adhoc")