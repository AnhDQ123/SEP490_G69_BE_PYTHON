from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import faiss
import numpy as np
from flask import Flask, request, jsonify
import pandas as pd
import os
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader
# Dữ liệu huấn luyện (query, product, similarity)
df = pd.read_csv('training.csv')
train_data_from_csv = [
    InputExample(texts=[row['query'], row['product']], label=row['label'])
    for _, row in df.iterrows()
]
# Tải mô hình Sentence-BERT
model_save_path = 'saved_model/'

# Kiểm tra xem mô hình đã tồn tại chưa
if not os.path.exists(model_save_path):  # Nếu mô hình chưa được lưu
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Tạo Dataloader từ dữ liệu huấn luyện
    train_dataloader = DataLoader(train_data_from_csv, batch_size=8)

    # Loss function cho việc huấn luyện
    train_loss = losses.CosineSimilarityLoss(model)

    # Huấn luyện mô hình
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=10, warmup_steps=100)

    # Lưu mô hình sau khi huấn luyện
    model.save(model_save_path)
    print("Mô hình đã được huấn luyện và lưu tại:", model_save_path)

else:
    # Nếu mô hình đã tồn tại, tải mô hình đã lưu
    model = SentenceTransformer(model_save_path)
    print("Mô hình đã được tải từ thư mục:", model_save_path)

# Dữ liệu sản phẩm



file_path = 'products.csv'
df = pd.read_csv(file_path)
product_descriptions = df['name'].tolist()
product_embeddings = model.encode(product_descriptions)

# Chuyển embeddings thành numpy array
product_embeddings = np.array(product_embeddings)
# Khởi tạo FAISS index
dimension = product_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Thêm embeddings vào FAISS index
index.add(product_embeddings)

# Kiểm tra số lượng vectors trong index
print(f"Number of vectors in index: {index.ntotal}")

# API dựa trên truy vấn của người dùng
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def search():
    try:
        products_df = pd.read_csv("products.csv")  # Đảm bảo đường dẫn chính xác
        products = products_df.to_dict(orient="records")
        data = request.get_json()  # Lấy dữ liệu JSON từ POST request
        query = data.get("query", "").strip()

        if not query:
            return jsonify({"error": "Missing query parameter"}), 400

        # Tạo embedding cho truy vấn người dùng
        query_embedding = model.encode([query])

        # Tìm kiếm sản phẩm tương tự trong FAISS
        D, I = index.search(query_embedding, k=10)  # Lấy 10 kết quả gần nhất

        # Chuyển đổi các giá trị D thành float64 (Python float)
        results = [{"name": products[i]['name'], "similarity_score": float(D[0][j])} for j, i in enumerate(I[0])]

        return jsonify({"query": query, "results": results})
    except Exception as e:
        # In lỗi chi tiết ra console và trả về mã lỗi 500
        print(f"Error: {str(e)}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


# Chạy Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
