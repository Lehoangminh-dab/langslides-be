import pymysql
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# MySQL configuration
mysql_config = {
    'host': '127.0.0.1',
    'database': 'rag_database',
    'user': 'root',
    'password': '',
    'autocommit': True,
    'cursorclass': pymysql.cursors.DictCursor
}

class DatabaseHandler:
    def __init__(self, config=None):
        self.config = config or mysql_config
        self.connection = None
    
    def get_connection(self):
        """Get a database connection, creating one if needed"""
        try:
            if self.connection is None or not self.connection.open:
                self.connection = pymysql.connect(**self.config)
            return self.connection
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
    
    def close_connection(self):
        """Close the database connection"""
        if self.connection and self.connection.open:
            self.connection.close()
    
    def init_db(self):
        """Initialize database tables if they don't exist"""
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                # Create users table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id VARCHAR(255) PRIMARY KEY,
                    email VARCHAR(255) NOT NULL UNIQUE,
                    name VARCHAR(255),
                    picture TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
                # Create sessions table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id VARCHAR(36) PRIMARY KEY,
                    user_id VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
                )
                """)
                
                # Create chat_messages table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    session_id VARCHAR(36) NOT NULL,
                    is_user BOOLEAN NOT NULL,
                    message LONGTEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
                """)
                
                # Create files table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS file_uploads (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    session_id VARCHAR(36) NOT NULL,
                    user_id VARCHAR(255),
                    file_name VARCHAR(255) NOT NULL,
                    file_path TEXT,
                    file_type VARCHAR(50),
                    file_hash VARCHAR(64),
                    collection_name VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
                )
                """)
                
                # Create presentations table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS presentations (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    session_id VARCHAR(36) NOT NULL,
                    file_path TEXT NOT NULL,
                    template VARCHAR(50),
                    slide_count INT DEFAULT 10,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    gdrive_link TEXT,
                    download_count INT DEFAULT 0,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
                """)
                
            conn.commit()
            logger.info("Database tables initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    # User management methods
    def save_user(self, user_id: str, email: str, name: str = None, picture: str = None) -> bool:
        """Save or update a user in the database"""
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                # Check if user exists
                cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
                existing_user = cursor.fetchone()
                
                if existing_user:
                    # Update existing user
                    cursor.execute("""
                    UPDATE users SET email = %s, name = %s, picture = %s
                    WHERE id = %s
                    """, (email, name, picture, user_id))
                else:
                    # Insert new user
                    cursor.execute("""
                    INSERT INTO users (id, email, name, picture)
                    VALUES (%s, %s, %s, %s)
                    """, (user_id, email, name, picture))
                
                return True
        except Exception as e:
            logger.error(f"Error saving user: {e}")
            return False
    
    def get_user(self, user_id: str) -> Optional[Dict]:
        """Get user information by ID"""
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
                return cursor.fetchone()
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            return None
    
    # Session management methods - FIXED VERSION
    def create_session(self, session_id: str, user_id: Optional[str] = None) -> bool:
        """Create a new session in the database, or update if it exists"""
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                # Check if the session already exists
                cursor.execute(
                    "SELECT session_id FROM sessions WHERE session_id = %s", 
                    (session_id,)
                )
                existing_session = cursor.fetchone()
                
                if existing_session:
                    # Update the existing session with the new user_id and refresh last_activity
                    cursor.execute("""
                    UPDATE sessions 
                    SET user_id = %s, last_activity = CURRENT_TIMESTAMP
                    WHERE session_id = %s
                    """, (user_id, session_id))
                    logger.info(f"Updated existing session: {session_id}")
                else:
                    # Insert new session
                    cursor.execute("""
                    INSERT INTO sessions (session_id, user_id)
                    VALUES (%s, %s)
                    """, (session_id, user_id))
                    logger.info(f"Created new session: {session_id}")
                
                return True
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return False
    
    def update_session_activity(self, session_id: str) -> bool:
        """Update the last activity timestamp for a session"""
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute("""
                UPDATE sessions SET last_activity = CURRENT_TIMESTAMP
                WHERE session_id = %s
                """, (session_id,))
                
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error updating session activity: {e}")
            return False
    
    # Chat message methods
    def save_message(self, session_id: str, is_user: bool, message: str) -> bool:
        """Save a chat message to the database"""
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute("""
                INSERT INTO chat_messages (session_id, is_user, message)
                VALUES (%s, %s, %s)
                """, (session_id, is_user, message))
                
                return True
        except Exception as e:
            logger.error(f"Error saving message: {e}")
            return False
    
    def get_session_messages(self, session_id: str) -> List[Tuple[Optional[str], Optional[str]]]:
        """Get all messages for a session in the format [(user_msg, ai_msg), ...]"""
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute("""
                SELECT is_user, message FROM chat_messages
                WHERE session_id = %s
                ORDER BY created_at ASC
                """, (session_id,))
                
                results = cursor.fetchall()
                
                # Convert to the expected format
                formatted_messages = []
                for row in results:
                    if row['is_user']:
                        # User message: (user_msg, None)
                        formatted_messages.append((row['message'], None))
                    else:
                        # AI message: (None, ai_msg)
                        # For AI-only messages like greetings, the previous entry might not exist
                        if formatted_messages and formatted_messages[-1][1] is None:
                            # Complete the previous user message with this AI response
                            user_msg = formatted_messages[-1][0]
                            formatted_messages[-1] = (user_msg, row['message'])
                        else:
                            # This is an AI-only message (like greeting)
                            formatted_messages.append((None, row['message']))
                
                return formatted_messages
        except Exception as e:
            logger.error(f"Error getting session messages: {e}")
            return []
    
    def clear_session_messages(self, session_id: str) -> bool:
        """Clear all messages for a session"""
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute("""
                DELETE FROM chat_messages
                WHERE session_id = %s
                """, (session_id,))
                
                return True
        except Exception as e:
            logger.error(f"Error clearing session messages: {e}")
            return False
    
    # File upload methods
    def save_file_upload(self, session_id: str, file_name: str, file_path: str,
                        file_type: str, file_hash: str, collection_name: str,
                        user_id: Optional[str] = None) -> int:
        """Save a file upload record and return its ID"""
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute("""
                INSERT INTO file_uploads (session_id, user_id, file_name, file_path,
                                         file_type, file_hash, collection_name)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (session_id, user_id, file_name, file_path,
                     file_type, file_hash, collection_name))
                
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error saving file upload: {e}")
            return -1
    
    def get_file_by_hash(self, file_hash: str) -> Optional[Dict]:
        """Get file information by hash"""
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute("""
                SELECT * FROM file_uploads
                WHERE file_hash = %s
                ORDER BY created_at DESC
                LIMIT 1
                """, (file_hash,))
                
                return cursor.fetchone()
        except Exception as e:
            logger.error(f"Error getting file by hash: {e}")
            return None
    
    def get_user_uploads(self, user_id: str) -> List[Dict]:
        """Get all file uploads for a user"""
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute("""
                SELECT fu.id, fu.file_name, fu.file_hash, fu.collection_name, 
                       fu.created_at, s.session_id
                FROM file_uploads fu
                JOIN sessions s ON fu.session_id = s.session_id
                WHERE s.user_id = %s
                ORDER BY fu.created_at DESC
                """, (user_id,))
                
                results = cursor.fetchall()
                
                # Format dates for JSON serialization
                for row in results:
                    if 'created_at' in row and row['created_at']:
                        row['created_at'] = row['created_at'].isoformat()
                
                return results
        except Exception as e:
            logger.error(f"Error getting user uploads: {e}")
            return []

    # Presentation methods
    def save_presentation(self, session_id: str, file_path: str,
                         template: str = "Basic", slide_count: int = 10) -> int:
        """Save a presentation record and return its ID"""
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute("""
                INSERT INTO presentations (session_id, file_path, template, slide_count)
                VALUES (%s, %s, %s, %s)
                """, (session_id, file_path, template, slide_count))
                
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error saving presentation: {e}")
            return -1
    
    def update_presentation_gdrive(self, presentation_id: int, gdrive_link: str) -> bool:
        """Update the Google Drive link for a presentation"""
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute("""
                UPDATE presentations SET gdrive_link = %s
                WHERE id = %s
                """, (gdrive_link, presentation_id))
                
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error updating presentation Google Drive link: {e}")
            return False
    
    def get_latest_presentation(self, session_id: str) -> Optional[Dict]:
        """Get the latest presentation for a session"""
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute("""
                SELECT * FROM presentations
                WHERE session_id = %s
                ORDER BY created_at DESC
                LIMIT 1
                """, (session_id,))
                
                return cursor.fetchone()
        except Exception as e:
            logger.error(f"Error getting latest presentation: {e}")
            return None
    
    def increment_download_count(self, presentation_id: int) -> bool:
        """Increment download count for a presentation"""
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute("""
                UPDATE presentations 
                SET download_count = download_count + 1
                WHERE id = %s
                """, (presentation_id,))
                
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error incrementing download count: {e}")
            return False
    
    def get_user_presentations(self, user_id: str) -> List[Dict]:
        """Get all presentations for a user"""
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute("""
                SELECT p.id, p.file_path, p.template, p.slide_count, 
                       p.created_at, p.gdrive_link, p.download_count, s.session_id
                FROM presentations p
                JOIN sessions s ON p.session_id = s.session_id
                WHERE s.user_id = %s
                ORDER BY p.created_at DESC
                """, (user_id,))
                
                results = cursor.fetchall()
                
                # Format dates for JSON serialization
                for row in results:
                    if 'created_at' in row and row['created_at']:
                        row['created_at'] = row['created_at'].isoformat()
                
                return results
        except Exception as e:
            logger.error(f"Error getting user presentations: {e}")
            return []

    def get_user_sessions(self, user_id: str) -> List[Dict]:
        """Get all sessions for a user with message counts"""
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute("""
                SELECT s.session_id, s.created_at, s.last_activity,
                       COUNT(DISTINCT cm.id) as message_count
                FROM sessions s
                LEFT JOIN chat_messages cm ON s.session_id = cm.session_id
                WHERE s.user_id = %s
                GROUP BY s.session_id, s.created_at, s.last_activity
                ORDER BY s.last_activity DESC
                """, (user_id,))
                
                results = cursor.fetchall()
                
                # Format dates for JSON serialization
                for row in results:
                    if 'created_at' in row and row['created_at']:
                        row['created_at'] = row['created_at'].isoformat()
                    if 'last_activity' in row and row['last_activity']:
                        row['last_activity'] = row['last_activity'].isoformat()
                
                return results
        except Exception as e:
            logger.error(f"Error getting user sessions: {e}")
            return []