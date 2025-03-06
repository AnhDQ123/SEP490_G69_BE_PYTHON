import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler

# Import dữ liệu từ file CSV
file_path = "food_recommendation_dataset.csv"
df = pd.read_csv(file_path)

# Sắp xếp dữ liệu theo user_id, item_id và rating (rating giảm dần)
df = df.sort_values(by=["user_id", "item_id", "rating"], ascending=[True, True, False])
# Loại bỏ các đánh giá trùng lặp, giữ lại đánh giá mới nhất
df = df.drop_duplicates(subset=["user_id", "item_id"], keep="first")

# Tạo ma trận đánh giá từ dữ liệu
ratings_matrix = df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
# Áp dụng SVD để giảm chiều dữ liệu
U, sigma, Vt = svds(ratings_matrix.values.astype(float), k=3)  # k=3 có thể điều chỉnh
sigma = np.diag(sigma)
predicted_ratings = np.dot(np.dot(U, sigma), Vt)
predicted_df = pd.DataFrame(predicted_ratings, index=ratings_matrix.index, columns=ratings_matrix.columns)

# Tạo từ điển ánh xạ item_id -> product_name
products_en = df.set_index("item_id")["product_name"].to_dict()


# Hàm lấy gợi ý dựa trên Collaborative Filtering
def get_collab_recommendations(user_id, top_n=10):
    if user_id not in predicted_df.index:  # Kiểm tra user có tồn tại trong dữ liệu không
        return []
    user_predictions = predicted_df.loc[user_id].sort_values(ascending=False)
    recommendation_products = [(products_en[item], scores) for item, scores in user_predictions.items() if item in products_en][
                      :top_n]
    return recommendation_products


# Tạo từ điển ánh xạ item_id -> product_name để sử dụng cho Content-Based Filtering
product_categories = df[["item_id", "product_name"]].drop_duplicates().set_index("item_id")["product_name"].to_dict()
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(list(product_categories.values()))
similarity_matrix = cosine_similarity(tfidf_matrix)


# Hàm lấy gợi ý dựa trên Content-Based Filtering
def get_content_recommendations(user_id, top_n=10, weight_purchased=0.3, weight_content=0.5):
    user_purchases = df[df['user_id'] == user_id]
    purchased_items = user_purchases.loc[user_purchases['purchased'] == 1, 'item_id'].values
    items_to_predict = df.loc[~df['item_id'].isin(purchased_items), 'item_id'].unique()

    scores_list = []

    for item in items_to_predict:
        item_index = list(product_categories.keys()).index(item)
        content_score = max(similarity_matrix[item_index]) # Lấy độ tương đồng cao nhất

        # Tính điểm cộng thêm dựa trên sản phẩm đã mua
        bought_score = max(
            (similarity_matrix[item_index][list(product_categories.keys()).index(p)]
             for p in purchased_items if p in product_categories),
            default=0
        )

        scores = content_score * weight_content + weight_purchased * bought_score
        scores_list.append((item, scores))

    # Chuẩn hóa điểm số để đảm bảo sự cân bằng khi kết hợp với Collaborative Filtering
    if scores_list:
        scores = np.array([scores for _, scores in scores_list]).reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, np.max(predicted_df.values)))  # Scale về khoảng của Collaborative
        scaled_scores = scaler.fit_transform(scores).flatten()

        recommendation_products = [(product_categories[item], scaled_scores[i]) for i, (item, _) in enumerate(scores_list)]
    else:
        recommendation_products = []

    return sorted(recommendation_products, key=lambda x: x[1], reverse=True)[:top_n]

# Hàm gợi ý kết hợp giữa Collaborative Filtering và Content-Based Filtering
def get_hybrid_recommendations(user_id, top_n=10, alpha=0.5):
    collab_recs = get_collab_recommendations(user_id, top_n)
    content_recs = get_content_recommendations(user_id, top_n)

    if not collab_recs:
        return content_recs  # Nếu không có dữ liệu Collaborative, chỉ dùng Content-Based

    content_recs_dict = dict(content_recs)
    collab_recs_dict = dict(collab_recs)

    combined_recs = {}
    for item, scores in content_recs_dict.items():
        combined_recs[item] = scores * alpha
    for item, scores in collab_recs_dict.items():
        combined_recs[item] = combined_recs.get(item, 0) + scores * (1 - alpha)

    final_recs = sorted(combined_recs.items(), key=lambda x: x[1], reverse=True)
    return final_recs[:top_n]


# In kết quả gợi ý theo từng phương pháp
print("Hybrid Filtering")
recommendations = get_hybrid_recommendations(1)
for i, (product, score) in enumerate(recommendations, 1):
    print(f"{i}. {product} - {score:.2f}")

print("\n")
print("Content-Based Filtering")
recommendations1 = get_content_recommendations(1)
for i, (product, score) in enumerate(recommendations1, 1):
    print(f"{i}. {product} - {score:.2f}")

print("\n")
print("Collaborative Filtering")
recommendations2 = get_collab_recommendations(1)
for i, (product, score) in enumerate(recommendations2, 1):
    print(f"{i}. {product} - {score:.2f}")
