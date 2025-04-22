import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import requests
import requests_cache
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, jsonify, request
from flask_cors import CORS
import jwt
from functools import wraps

# === CONFIG ===
# Load .env file
load_dotenv()

# Get config
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8080/api/product")
JWT_SECRET = os.getenv("JWT_SECRET", "dev_secret")
requests_cache.install_cache('api_cache', expire_after=300)

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


# === HELPER FUNCTION ===
def fetch_data():
    try:
        response = requests.get(f"{API_BASE_URL}/data", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API response error: {response.status_code}")
    except Exception as e:
        print(f"Error fetching data: {e}")
    return {}

# === DATA LOAD & PROCESS ===
data = fetch_data()
try:
    df_orders = pd.DataFrame(data.get("orders", []))
    df_order_details = pd.DataFrame(data.get("order_details", []))
    df_products = pd.DataFrame(data.get("products", []))
    df_feedbacks = pd.DataFrame(data.get("feedbacks", []))

    # Merge feedback
    df_order_details = df_order_details.merge(df_feedbacks, on=["userId", "productId"], how="left").fillna(0)

    # Ratings Matrix
    ratings_matrix = df_order_details.pivot_table(index="userId", columns="productId", values="rate", aggfunc="mean").fillna(0)

    # SVD - Collaborative Filtering
    U, sigma, Vt = svds(ratings_matrix.values.astype(float), k=min(10, len(ratings_matrix)-1))
    sigma = np.diag(sigma)
    predicted_ratings = np.dot(np.dot(U, sigma), Vt)
    predicted_df = pd.DataFrame(predicted_ratings, index=ratings_matrix.index, columns=ratings_matrix.columns)

    # TF-IDF Content Filtering
    df_products["combined_text"] = df_products["name"].fillna("") + " " + df_products["description"].fillna("")
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_products["combined_text"])
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Product dictionaries
    products_dict = df_products.set_index("id")["name"].to_dict()
    descriptions_dict = df_products.set_index("id")["description"].to_dict()

except Exception as e:
    print(f"Error preparing data: {e}")
    df_orders, df_order_details, df_products, df_feedbacks = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    predicted_df = pd.DataFrame()
    similarity_matrix = np.array([])

# === RECOMMENDER FUNCTIONS ===
def get_collab_recommendations(user_id, top_n=10):
    try:
        if user_id not in predicted_df.index:
            return []
        user_predictions = predicted_df.loc[user_id].sort_values(ascending=False)
        return [{"productId": pid, "name": products_dict.get(pid, "Unknown"),
                 "description": descriptions_dict.get(pid, ""), "score": score}
                for pid, score in user_predictions.items() if pid in products_dict][:top_n]
    except Exception as e:
        print(f"Error in collaborative recommendation: {e}")
        return []

def get_content_recommendations(user_id, top_n=10, weight_purchased=0.3, weight_content=0.5):
    try:
        user_purchases = df_order_details[df_order_details['userId'] == user_id]
        purchased_items = set(user_purchases['productId'].values)
        items_to_predict = df_products.loc[~df_products['id'].isin(purchased_items), 'id'].unique()

        scores_list = []
        for item in items_to_predict:
            if item not in products_dict:
                continue
            item_index = list(products_dict.keys()).index(item)
            content_score = float(np.mean(similarity_matrix[item_index]))
            bought_scores = [float(similarity_matrix[item_index][list(products_dict.keys()).index(p)]) *
                             user_purchases[user_purchases['productId'] == p]['quantity'].sum()
                             for p in purchased_items if p in products_dict]
            bought_score = np.mean(bought_scores) if bought_scores else 0.0
            total_score = content_score * weight_content + weight_purchased * bought_score
            total_score = 0.0 if np.isnan(total_score) else total_score
            scores_list.append((item, total_score))

        if scores_list:
            scores = np.array([score for _, score in scores_list]).reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, float(np.max(predicted_df.values)) if not predicted_df.empty else 1))
            scaled_scores = scaler.fit_transform(scores).flatten().tolist()
        else:
            return []

        recommendations = sorted([{"productId": int(pid), "name": str(products_dict.get(pid, "")),
                                    "description": str(descriptions_dict.get(pid, "")), "score": float(scaled_scores[i])}
                                   for i, (pid, _) in enumerate(scores_list)],
                                  key=lambda x: x["score"], reverse=True)[:top_n]
        return recommendations
    except Exception as e:
        print(f"Error in content recommendation: {e}")
        return []

def get_popular_products(top_n, product_type):
    try:
        filtered_products = df_products[df_products["type"] == product_type]
        if filtered_products.empty:
            return []

        popularity_df = df_order_details.groupby("productId").agg(
            total_quantity=("quantity", "sum"), avg_rating=("rate", "mean")
        ).reset_index()

        merged_df = popularity_df.merge(filtered_products, left_on="productId", right_on="id")
        merged_df["score"] = merged_df["total_quantity"] * 0.7 + merged_df["avg_rating"] * 0.3

        top_products = merged_df.sort_values(by="score", ascending=False).head(top_n)
        return [{"productId": int(pid)} for pid in top_products["productId"]]
    except Exception as e:
        print(f"Error getting popular products: {e}")
        return []

def get_hybrid_recommendations(user_id, top_n=10, product_type="FRESH"):
    try:
        if df_order_details.empty or df_products.empty:
            return []

        num_orders = len(df_order_details[df_order_details["userId"] == user_id])
        alpha = 0.3 if num_orders < 3 else 0.6

        filtered_product_ids = set(df_products[df_products["type"] == product_type]["id"])
        collab = [rec for rec in get_collab_recommendations(user_id, top_n*top_n) if rec["productId"] in filtered_product_ids]
        content = [rec for rec in get_content_recommendations(user_id, top_n*top_n) if rec["productId"] in filtered_product_ids]

        if not collab and not content:
            return get_popular_products(top_n, product_type)

        combined_scores = {}
        for rec in content:
            combined_scores[rec["productId"]] = rec.get("score", 0) * alpha
        for rec in collab:
            combined_scores[rec["productId"]] = combined_scores.get(rec["productId"], 0) + rec.get("score", 0) * (1 - alpha)

        final_recs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return [{"productId": pid} for pid, _ in final_recs[:top_n]]
    except Exception as e:
        print(f"Error in hybrid recommendation: {e}")
        return []

# === API ENDPOINTS ===
@app.route("/recommend/<int:user_id>/<string:product_type>/<int:top_n>", methods=["GET"])
@require_jwt
def get_recommendation_api(user_id, product_type, top_n):
    try:
        recs = get_hybrid_recommendations(user_id, top_n, product_type)
        return jsonify(recs if recs else [])
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
