# user.py
from flask_login import UserMixin # Kế thừa từ UserMixin để có sẵn các thuộc tính cần thiết
from db import get_db # Import hàm get_db từ file db.py

class User(UserMixin):
    def __init__(self, id_, name, email, profile_pic):
        self.id = id_ # id phải là thuộc tính bắt buộc cho Flask-Login (thường là primary key)
        self.name = name
        self.email = email
        self.profile_pic = profile_pic

    @staticmethod
    def get(user_id):
        """Lấy thông tin user từ DB bằng user_id (Google ID)."""
        db = get_db()
        user_row = db.execute(
            "SELECT * FROM user WHERE id = ?", (user_id,)
        ).fetchone()
        if not user_row:
            return None
        # Tạo đối tượng User từ dữ liệu lấy được
        user = User(
            id_=user_row["id"], name=user_row["name"], email=user_row["email"], profile_pic=user_row["profile_pic"]
        )
        return user

    @staticmethod
    def create(id_, name, email, profile_pic):
        """Tạo user mới trong DB."""
        db = get_db()
        try:
            print(f"Attempting to create user: ID={id_}, Name={name}")
            db.execute(
                "INSERT INTO user (id, name, email, profile_pic) VALUES (?, ?, ?, ?)",
                (id_, name, email, profile_pic),
            )
            db.commit()
            print(f"User {name} ({id_}) created successfully.")
        except db.IntegrityError: # Bắt lỗi nếu ID hoặc Email đã tồn tại (do constraint UNIQUE)
            db.rollback()
            print(f"User creation failed for {name}. User ID or Email might already exist. Attempting update instead.")
            # Thay vì chỉ báo lỗi, chúng ta gọi hàm update
            User.update(id_, name, email, profile_pic)
        except db.Error as e:
            db.rollback()
            print(f"Database error during user creation: {e}")


    @staticmethod
    def update(id_, name, email, profile_pic):
        """Cập nhật thông tin user trong DB."""
        db = get_db()
        try:
            print(f"Attempting to update user: ID={id_}, Name={name}")
            db.execute(
                "UPDATE user SET name = ?, email = ?, profile_pic = ? WHERE id = ?",
                (name, email, profile_pic, id_)
            )
            db.commit()
            print(f"User {name} ({id_}) updated successfully.")
        except db.Error as e:
            db.rollback()
            print(f"Failed to update user {name}. Error: {e}")