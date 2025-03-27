import pandas as pd
import requests
import requests_cache
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, jsonify, request
from flask_cors import CORS

API_BASE_URL = "http://localhost:8080/api/product"
requests_cache.install_cache('api_cache', expire_after=0)

app = Flask(__name__)
CORS(app)

# === BƯỚC 1: GIẢ LẬP DỮ LIỆU ===
def fetch_data():
    response = requests.get(f"{API_BASE_URL}/data")
    return response.json() if response.status_code == 200 else {}

# Lấy dữ liệu từ API
data = fetch_data()

# === BƯỚC 2: XỬ LÝ DỮ LIỆU ===
df_orders = pd.DataFrame(data["orders"])
df_order_details = pd.DataFrame(data["order_details"])
df_products = pd.DataFrame(data["products"])
df_feedback = pd.DataFrame(data["feedbacks"])

# Gộp dữ liệu feedback vào order_details
df_order_details = df_order_details.merge(df_feedback, on=['userId', 'productId'], how='left').fillna(0)

# Tạo ma trận ratings
ratings_matrix = df_order_details.pivot_table(
    index='userId',
    columns='productId',
    values='rate',
    aggfunc='mean'  # Hoặc 'max', 'sum' tùy mục đích
).fillna(0)

# SVD - Collaborative Filtering
U, sigma, Vt = svds(ratings_matrix.values.astype(float), k=min(10, len(ratings_matrix)-1))
sigma = np.diag(sigma)
predicted_ratings = np.dot(np.dot(U, sigma), Vt)
predicted_df = pd.DataFrame(predicted_ratings, index=ratings_matrix.index, columns=ratings_matrix.columns)

# Chuyển đổi dữ liệu sản phẩm
products_dict = df_products.set_index("id")["name"].to_dict()
descriptions_dict = df_products.set_index("id")["description"].to_dict()

# Content-Based Filtering
df_products["combined_text"] = df_products["name"] + " " + df_products["description"].fillna("")
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df_products["combined_text"].fillna(""))
similarity_matrix = cosine_similarity(tfidf_matrix)

# === BƯỚC 3: HÀM GỢI Ý ===
def get_collab_recommendations(user_id, top_n=10):
    if user_id not in predicted_df.index:
        return []
    user_predictions = predicted_df.loc[user_id].sort_values(ascending=False)
    return [{"productId": pid, "name": products_dict[pid], "description": descriptions_dict.get(pid, ""), "score": score} for pid, score in user_predictions.items() if pid in products_dict][:top_n]

def get_content_recommendations(user_id, top_n=10, weight_purchased=0.3, weight_content=0.5):
    user_purchases = df_order_details[df_order_details['userId'] == user_id]
    purchased_items = set(user_purchases['productId'].values)
    items_to_predict = df_products.loc[~df_products['id'].isin(purchased_items), 'id'].unique()

    if len(items_to_predict) == 0:
        return []

    scores_list = []
    for item in items_to_predict:
        item_index = list(products_dict.keys()).index(item)
        content_score = float(np.mean(similarity_matrix[item_index]))
        bought_scores = [float(similarity_matrix[item_index][list(products_dict.keys()).index(p)]) *
                         user_purchases[user_purchases['productId'] == p]['quantity'].sum() for p in purchased_items if
                         p in products_dict]
        bought_score = np.mean(bought_scores) if bought_scores else 0.0
        total_score = content_score * weight_content + weight_purchased * bought_score
        total_score = 0.0 if np.isnan(total_score) else total_score
        scores_list.append((item, total_score))
    if scores_list:
        scores = np.array([score for _, score in scores_list]).reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, float(np.max(predicted_df.values))))
        scaled_scores = scaler.fit_transform(scores).flatten().tolist()
    else:
        scaled_scores = []
    recommendations = sorted([{"productId": int(pid), "name": str(products_dict[pid]),
                               "description": str(descriptions_dict.get(pid, "")), "score": float(scaled_scores[i])} for
                              i, (pid, _) in enumerate(scores_list)], key=lambda x: x["score"], reverse=True)[:top_n]
    return recommendations

def get_hybrid_recommendations(user_id, top_n=10, product_type="FRESH"):
    num_orders = len(df_order_details[df_order_details["userId"] == user_id])
    alpha = 0.3 if num_orders < 3 else 0.6

    if user_id not in predicted_df.index or len(predicted_df) == 0:
        return get_popular_products(top_n, product_type)

    filtered_products = df_products[df_products["type"] == product_type]
    filtered_product_ids = set(filtered_products["id"])

    collab_recs = get_collab_recommendations(user_id, top_n)
    content_recs = get_content_recommendations(user_id, top_n)

    collab_recs = [rec for rec in collab_recs if rec["productId"] in filtered_product_ids]
    content_recs = [rec for rec in content_recs if rec["productId"] in filtered_product_ids]

    if not collab_recs and not content_recs:
        return get_popular_products(top_n, product_type)

    combined_scores = {}
    for rec in content_recs:
        combined_scores[rec["productId"]] = rec.get("score", 0) * alpha
    for rec in collab_recs:
        combined_scores[rec["productId"]] = combined_scores.get(rec["productId"], 0) + rec.get("score", 0) * (1 - alpha)

    # Nếu có sản phẩm bị trùng, sắp xếp giảm dần theo điểm số
    final_recs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

    return [{"productId": pid} for pid, _ in final_recs[:top_n]]

#name
#input
#out
def get_popular_products(top_n, product_type):
    if "type" not in df_products.columns or "id" not in df_products.columns:
        return []

    # Lọc sản phẩm theo loại
    filtered_products = df_products[df_products["type"] == product_type]
    if filtered_products.empty:
        return []

    # Gộp dữ liệu số lượng mua và đánh giá trung bình
    popularity_df = (
        df_order_details.groupby("productId")
        .agg(total_quantity=("quantity", "sum"), avg_rating=("rate", "mean"))
        .reset_index()
    )

    # Kết hợp với danh sách sản phẩm lọc theo loại
    merged_df = popularity_df.merge(filtered_products, left_on="productId", right_on="id")

    if merged_df.empty:
        return []

    # Xếp hạng sản phẩm dựa trên tổng số lượng mua (ưu tiên) và điểm đánh giá trung bình
    merged_df["score"] = merged_df["total_quantity"] * 0.7 + merged_df["avg_rating"] * 0.3

    # Sắp xếp theo điểm số giảm dần, lấy top_n
    top_products = merged_df.sort_values(by="score", ascending=False).head(top_n)

    return [{"productId": int(pid)} for pid in top_products["productId"]]


# === BƯỚC 4: API Flask ===
@app.route("/recommend/collab", methods=["GET"])
def recommend_collab():
    user_id = int(request.args.get("userId", 101))
    return jsonify(get_collab_recommendations(user_id))


@app.route("/recommend/content", methods=["GET"])
def recommend_content():
    user_id = int(request.args.get("userId", 101))
    return jsonify(get_content_recommendations(user_id))


@app.route("/recommend/hybrid", methods=["GET"])
def recommend_hybrid():
    user_id = int(request.args.get("userId", 0))
    return jsonify(get_hybrid_recommendations(user_id,10, "COOKED"))


# API trả về danh sách sản phẩm gợi ý
@app.route("/recommend/<int:user_id>/<string:product_type>/<int:top_n>", methods=["GET"])
def get_recommendations(user_id, product_type, top_n):
    recommended_products = get_hybrid_recommendations(user_id, top_n, product_type)
    return jsonify(recommended_products if recommended_products else [])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
