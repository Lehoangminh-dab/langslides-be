import chromadb
import os
import sys # Thêm sys để sử dụng sys.modules

# Đoạn code để thử sử dụng pysqlite3-binary nếu có vấn đề với phiên bản SQLite hệ thống
try:
    # Ưu tiên pysqlite3-binary nếu đã cài đặt, để đảm bảo tương thích phiên bản SQLite
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("Đã chuyển sang sử dụng pysqlite3-binary.")
except ImportError:
    print("pysqlite3-binary không được tìm thấy hoặc không cần thiết. Sử dụng sqlite3 mặc định của hệ thống.")
except KeyError:
    # Xử lý trường hợp 'pysqlite3' có thể đã được import nhưng không nằm trong sys.modules đúng cách
    # hoặc đã được pop ra trước đó.
    print("Có vẻ như pysqlite3 đã được xử lý hoặc không thể hoán đổi với sqlite3. Tiếp tục với cấu hình hiện tại.")


def print_all_data_from_chroma(db_directory_path: str):
    """
    Kết nối đến một cơ sở dữ liệu ChromaDB lưu trữ cục bộ và in ra tất cả dữ liệu.

    Args:
        db_directory_path (str): Đường dẫn đến thư mục chứa dữ liệu của ChromaDB
                                 (nơi file chroma.sqlite3 và các file khác được lưu).
    """
    if not os.path.isdir(db_directory_path):
        print(f"Lỗi: Đường dẫn '{db_directory_path}' không phải là một thư mục hoặc không tồn tại.")
        return

    print(f"Đang kết nối tới ChromaDB tại thư mục: {db_directory_path}")
    try:
        # Khởi tạo một PersistentClient để kết nối tới cơ sở dữ liệu đã tồn tại
        # ChromaDB sẽ tìm file chroma.sqlite3 (và các file khác) trong thư mục này.
        client = chromadb.PersistentClient(path=db_directory_path)
        print("Kết nối thành công!")

        # Lấy danh sách tất cả các collections
        collections = client.list_collections()

        if not collections:
            print("Không tìm thấy collection nào trong cơ sở dữ liệu.")
            return

        print(f"\nTìm thấy {len(collections)} collection(s):")
        for i, collection_obj in enumerate(collections):
            # Tên collection có thể lấy từ thuộc tính .name của đối tượng Collection
            collection_name = collection_obj.name
            print(f"\n--- Collection {i+1}: {collection_name} (ID: {collection_obj.id}) ---")

            try:
                # Bạn có thể sử dụng trực tiếp collection_obj hoặc lấy lại bằng tên/ID
                # current_collection = client.get_collection(name=collection_name) # Cách này cũng được
                current_collection = collection_obj # Sử dụng đối tượng đã có từ list_collections()

                # Lấy tất cả dữ liệu từ collection
                # Bao gồm: "ids", "embeddings", "metadatas", "documents", "uris", "data"
                # Mặc định (nếu không có `include`): trả về ids, metadatas, documents (và distances cho query)
                # embeddings thường không được trả về mặc định trong `get()` để tiết kiệm tài nguyên.
                results = current_collection.get(include=["embeddings", "metadatas", "documents"])

                if not results or not results.get('ids'): # Kiểm tra xem có ID nào trả về không
                    print("  Collection này trống hoặc không có dữ liệu phù hợp với yêu cầu `include`.")
                    continue

                num_items = len(results['ids'])
                print(f"  Số lượng mục: {num_items}")

                for j in range(num_items):
                    print(f"\n  Mục {j+1}:")
                    print(f"    ID: {results['ids'][j]}")

                    # Kiểm tra sự tồn tại của key và giá trị không phải None trước khi truy cập
                    if results.get('documents') and j < len(results['documents']) and results['documents'][j] is not None:
                        print(f"    Document: {results['documents'][j]}")
                    else:
                        print("    Document: (không có hoặc null)")

                    if results.get('metadatas') and j < len(results['metadatas']) and results['metadatas'][j] is not None:
                        print(f"    Metadata: {results['metadatas'][j]}")
                    else:
                        print("    Metadata: (không có hoặc null)")

                    # ... (các dòng print ID, Document, Metadata ở trên) ...

                    # Lấy danh sách tất cả embeddings một cách an toàn
                    all_embeddings_list = results.get('embeddings')

                    # Kiểm tra xem danh sách embeddings có tồn tại không,
                    # và có embedding cho mục hiện tại (j) không.
                    if all_embeddings_list is not None and \
                       j < len(all_embeddings_list) and \
                       all_embeddings_list[j] is not None:
                        
                        current_embedding = all_embeddings_list[j]
                        embedding_preview = current_embedding[:5] # Lấy 5 giá trị đầu tiên để xem trước
                        
                        # Kiểm tra xem current_embedding có hỗ trợ lấy len() không (thường là có với vector)
                        if hasattr(current_embedding, '__len__'):
                            total_dims = len(current_embedding)
                            # Đảm bảo embedding_preview cũng không vượt quá số chiều thực tế nếu embedding ngắn
                            preview_dims = min(len(embedding_preview), total_dims)
                            print(f"    Embedding (xem trước {preview_dims}/{total_dims} chiều): {embedding_preview}...")
                        else:
                            # Trường hợp hiếm gặp: embedding không có độ dài (không phải list/array)
                            print(f"    Embedding (xem trước): {embedding_preview}...")
                    else:
                        print("    Embedding: (không có, null, hoặc không được yêu cầu)")
            except Exception as e:
                print(f"  Lỗi khi lấy dữ liệu từ collection '{collection_name}': {e}")
                import traceback
                traceback.print_exc()


    except Exception as e:
        print(f"Đã xảy ra lỗi khi kết nối hoặc đọc từ ChromaDB: {e}")
        print("Hãy đảm bảo rằng ChromaDB đã được thiết lập đúng và đường dẫn là chính xác.")
        print("Nếu gặp lỗi về phiên bản SQLite, hãy chắc chắn bạn đã cài đặt 'pysqlite3-binary' và nó được import đúng cách (đã thử tự động ở đầu script).")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # ---- HƯỚNG DẪN SỬ DỤNG ----
    # 1. Thay thế 'duong_dan_den_thu_muc_chroma_db' bằng đường dẫn thực tế
    #    đến thư mục chứa cơ sở dữ liệu ChromaDB của bạn.
    #    Ví dụ: "./my_chroma_data" hoặc "/path/to/your/chroma_db_directory"
    #    Đây là thư mục bạn đã cung cấp cho `chromadb.PersistentClient(path="...")`
    #    khi tạo hoặc làm việc với DB. File `chroma.sqlite3` sẽ nằm trong thư mục này.

    # Ví dụ: nếu thư mục chứa file chroma.sqlite3 của bạn tên là 'db_storage'
    # và nó nằm cùng cấp với file script Python này:
    path_to_db_directory = "."  # <--- THAY ĐỔI ĐƯỜNG DẪN NÀY CHO PHÙ HỢP

    # Tạo một thư mục và một collection mẫu nếu chưa có để kiểm thử
    # Điều này hữu ích nếu bạn muốn chạy script mà không có sẵn DB
    if not os.path.exists(path_to_db_directory):
        print(f"Thư mục '{path_to_db_directory}' không tồn tại. Đang tạo dữ liệu mẫu...")
        os.makedirs(path_to_db_directory, exist_ok=True)
        try:
            # Tạo client tạm thời để thêm dữ liệu
            temp_client = chromadb.PersistentClient(path=path_to_db_directory)
            # Lấy hoặc tạo collection
            collection = temp_client.get_or_create_collection(
                name="sample_collection",
                metadata={"description": "Một collection mẫu để kiểm thử"}
            )
            # Thêm dữ liệu mẫu
            if collection.count() == 0: # Chỉ thêm nếu collection trống
                collection.add(
                    documents=[
                        "Đây là tài liệu đầu tiên về AI.",
                        "Tài liệu thứ hai nói về học máy.",
                        "Một tài liệu khác không có embedding."
                    ],
                    metadatas=[
                        {"source": "web", "year": 2023},
                        {"source": "book", "year": 2022},
                        {"source": "internal"}
                    ],
                    ids=["doc1_ai", "doc2_ml", "doc3_noembed"]
                )
                # Ví dụ thêm dữ liệu có embedding (ChromaDB sẽ tự tạo nếu không cung cấp embedding_function)
                # collection.add(
                #     embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                #     documents=["Tài liệu có embedding 1", "Tài liệu có embedding 2"],
                #     metadatas=[{"type": "custom_embed"}, {"type": "custom_embed"}],
                #     ids=["embed_1", "embed_2"]
                # )
                print(f"Đã tạo dữ liệu mẫu trong collection 'sample_collection' tại '{path_to_db_directory}'.")
            else:
                print(f"Collection 'sample_collection' đã có dữ liệu tại '{path_to_db_directory}'. Bỏ qua tạo mẫu.")

            # Quan trọng: Đảm bảo client được giải phóng để tránh lock file,
            # đặc biệt nếu bạn tạo và đọc ngay lập tức.
            # Trong Python, việc này thường xảy ra khi biến ra khỏi scope hoặc dùng `del`.
            # Hoặc, nếu client hỗ trợ context manager (with ... as ...), thì nên dùng.
            # ChromaDB client hiện tại không trực tiếp là context manager.
            del temp_client
            del collection
            # Chờ một chút để đảm bảo file đã được ghi hoàn toàn (thường không cần thiết)
            # import time
            # time.sleep(0.5)

        except Exception as e:
            print(f"Lỗi khi tạo dữ liệu mẫu: {e}")
            import traceback
            traceback.print_exc()

    # Gọi hàm để in dữ liệu từ thư mục đã chỉ định
    print_all_data_from_chroma(path_to_db_directory)

    # Một ví dụ khác nếu bạn biết chắc file chroma.sqlite3 nằm ở thư mục hiện tại
    # và ChromaDB cũng được cấu hình để dùng thư mục hiện tại (path="."):
    # print("\n--- Thử đọc từ thư mục hiện tại ('.') ---")
    # print_all_data_from_chroma(".")