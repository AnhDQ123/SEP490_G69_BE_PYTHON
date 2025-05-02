import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import requests
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, jsonify, request
from flask_cors import CORS
import jwt
from functools import wraps

# === CONFIG ===
load_dotenv()
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8080/api/product")
JWT_SECRET = os.getenv("JWT_SECRET", "dev_secret")

app = Flask(__name__)
CORS(app)

# === JWT Middleware ===
def require_jwt(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid Authorization header"}), 401

        token = auth_header.replace("Bearer ", "")
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            request.user = payload
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401

        return f(*args, **kwargs)
    return decorated

# === DATA FETCHING WITH RETRY ===
def fetch_data(retries=3, delay=2):
    for attempt in range(retries):
        try:
            response = requests.get(f"{API_BASE_URL}/data", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"[Attempt {attempt+1}] API error: {response.status_code}")
        except Exception as e:
            print(f"[Attempt {attempt+1}] Fetch error: {e}")
        time.sleep(delay)
    return {}

# === DATA PREPARATION ===
def prepare_data():
    data = fetch_data()

    df_orders = pd.DataFrame(data.get("orders", []))
    df_order_details = pd.DataFrame(data.get("order_details", []))
    df_products = pd.DataFrame(data.get("products", []))
    df_feedbacks = pd.DataFrame(data.get("feedbacks", []))

    if df_order_details.empty or df_products.empty:
        return None

    df_order_details = df_order_details.merge(df_feedbacks, on=["userId", "productId"], how="left").fillna(0)

    ratings_matrix = df_order_details.pivot_table(index="userId", columns="productId", values="rate", aggfunc="mean").fillna(0)
    U, sigma, Vt = svds(ratings_matrix.values.astype(float), k=min(10, len(ratings_matrix)-1))
    sigma = np.diag(sigma)
    predicted_df = pd.DataFrame(np.dot(np.dot(U, sigma), Vt), index=ratings_matrix.index, columns=ratings_matrix.columns)

    df_products["combined_text"] = df_products["name"].fillna("") + " " + df_products["description"].fillna("")
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_products["combined_text"])
    similarity_matrix = cosine_similarity(tfidf_matrix)

    products_dict = df_products.set_index("id")["name"].to_dict()
    descriptions_dict = df_products.set_index("id")["description"].to_dict()

    return {
        "df_orders": df_orders,
        "df_order_details": df_order_details,
        "df_products": df_products,
        "predicted_df": predicted_df,
        "similarity_matrix": similarity_matrix,
        "products_dict": products_dict,
        "descriptions_dict": descriptions_dict
    }

# === HYBRID RECOMMENDER ===
def generate_hybrid_recommendation(user_id, top_n, product_type, data_dict):
    df_orders = data_dict["df_orders"]
    df_order_details = data_dict["df_order_details"]
    df_products = data_dict["df_products"]
    predicted_df = data_dict["predicted_df"]
    similarity_matrix = data_dict["similarity_matrix"]
    products_dict = data_dict["products_dict"]
    descriptions_dict = data_dict["descriptions_dict"]

    num_orders = len(df_order_details[df_order_details["userId"] == user_id])
    alpha = 0.3 if num_orders < 3 else 0.6

    filtered_product_ids = set(df_products[df_products["type"] == product_type]["id"])

    def collab_recs():
        if user_id not in predicted_df.index:
            return []
        preds = predicted_df.loc[user_id].sort_values(ascending=False)
        return [{"productId": int(pid), "score": float(score)} for pid, score in preds.items() if pid in filtered_product_ids]

    def content_recs():
        purchases = df_order_details[df_order_details['userId'] == user_id]
        bought_ids = set(purchases['productId'].values)
        items_to_predict = df_products.loc[~df_products['id'].isin(bought_ids), 'id'].unique()

        scores = []
        for pid in items_to_predict:
            if pid not in products_dict:
                continue
            idx = list(products_dict.keys()).index(pid)
            content_score = float(np.mean(similarity_matrix[idx]))
            bought_scores = [float(similarity_matrix[idx][list(products_dict.keys()).index(p)]) *
                             purchases[purchases['productId'] == p]['quantity'].sum()
                             for p in bought_ids if p in products_dict]
            bought_score = np.mean(bought_scores) if bought_scores else 0.0
            final_score = 0.5 * content_score + 0.3 * bought_score
            scores.append((pid, final_score))

        if not scores:
            return []

        scaler = MinMaxScaler()
        scores_array = scaler.fit_transform(np.array([s for _, s in scores]).reshape(-1, 1)).flatten()
        return [{"productId": int(pid), "score": float(scores_array[i])}
                for i, (pid, _) in enumerate(scores) if pid in filtered_product_ids]

    collab = collab_recs()
    content = content_recs()

    if not collab and not content:
        # fallback popular products
        popularity_df = df_order_details.groupby("productId").agg(
            total_quantity=("quantity", "sum"),
            avg_rating=("rate", "mean")
        ).reset_index()

        merged_df = popularity_df.merge(df_products, left_on="productId", right_on="id")
        merged_df = merged_df[merged_df["type"] == product_type]

        merged_df["score"] = merged_df["total_quantity"] * 0.7 + merged_df["avg_rating"] * 0.3
        top_products = merged_df.sort_values(by="score", ascending=False).head(top_n)

        return [{"productId": int(pid)} for pid in top_products["productId"]]

    combined_scores = {}
    for rec in content:
        combined_scores[int(rec["productId"])] = float(rec["score"] * alpha)
    for rec in collab:
        combined_scores[int(rec["productId"])] = combined_scores.get(int(rec["productId"]), 0) + float(rec["score"] * (1 - alpha))

    sorted_recs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    return [{"productId": int(pid)} for pid, _ in sorted_recs[:top_n]]

# === API ENDPOINT ===
@app.route("/recommend/<int:user_id>/<string:product_type>/<int:top_n>", methods=["GET"])
@require_jwt
def get_recommendation_api(user_id, product_type, top_n):
    try:
        data_dict = prepare_data()
        if not data_dict:
            return jsonify([])

        recs = generate_hybrid_recommendation(user_id, top_n, product_type, data_dict)
        return jsonify(recs)
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))