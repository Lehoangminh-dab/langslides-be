# db.py
import sqlite3
import click
from flask import current_app, g
from flask.cli import with_appcontext

DATABASE = 'users.sqlite3' # Tên file database SQLite

def get_db():
    """Mở kết nối DB mới nếu chưa có trong context hiện tại."""
    if 'db' not in g:
        g.db = sqlite3.connect(
            DATABASE,
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row # Giúp truy cập cột bằng tên
    return g.db

def close_db(e=None):
    """Đóng kết nối DB khi context kết thúc."""
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    """Khởi tạo database, tạo bảng nếu chưa tồn tại."""
    db = get_db()
    # Dùng IF NOT EXISTS để an toàn khi chạy lại
    print("Initializing the database...")
    db.execute('''
        CREATE TABLE IF NOT EXISTS user (
            id TEXT PRIMARY KEY, -- Google ID (sub) là TEXT
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL, -- Email nên là duy nhất
            profile_pic TEXT
        )
    ''')
    db.commit()
    print("Database initialized.")

@click.command('init-db')
@with_appcontext
def init_db_command():
    """Tạo mới các bảng trong database."""
    init_db()
    click.echo('Initialized the database.')

def init_app(app):
    """Đăng ký các hàm quản lý DB với ứng dụng Flask."""
    # Đảm bảo DB được đóng sau mỗi request
    app.teardown_appcontext(close_db)
    # Thêm lệnh 'flask init-db' vào flask CLI
    app.cli.add_command(init_db_command)