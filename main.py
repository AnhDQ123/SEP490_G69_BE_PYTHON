#
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.sparse.linalg import svds

#data import and preparation from csv file
file_path = "food_recommendation_dataset.csv"
df = pd.read_csv(file_path)

#
df = df.sort_values(by=["user_id", "item_id", "rating"], ascending=[True, True, False])
df = df.drop_duplicates(subset=["user_id", "item_id"], keep="first")

#
ratings_matrix = df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
U, sigma, Vt = svds(ratings_matrix.values.astype(float), k=3)
sigma = np.diag(sigma)
predicted_ratings = np.dot(np.dot(U, sigma), Vt)
predicted_df = pd.DataFrame(predicted_ratings, index=ratings_matrix.index, columns=ratings_matrix.columns)

#
products_en = df.set_index("item_id")["product_name"].to_dict()

#
def get_collab_recommendations(user_id, top_n=10):
    if user_id not in predicted_df.index:
        return []
    user_predictions = predicted_df.loc[user_id].sort_values(ascending=False)
    recommendations = [(products_en[item], score) for item, score in user_predictions.items() if item in products_en][
                      :top_n]
    return recommendations

#
product_categories = df[["item_id", "product_name"]].drop_duplicates().set_index("item_id")["product_name"].to_dict()
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(list(product_categories.values()))
similarity_matrix = cosine_similarity(tfidf_matrix)

#
def get_content_recommendations(user_id, top_n=10, weight_purchased=0.3, weight_content=0.5):
    user_purchases = df[df['user_id'] == user_id]
    purchased_items = user_purchases.loc[user_purchases['purchased'] == 1, 'item_id'].values
    items_to_predict = df.loc[~df['item_id'].isin(purchased_items), 'item_id'].unique()

    recommendations = []

    for item in items_to_predict:
        # if item not in product_categories:
        #     continue

        item_index = list(product_categories.keys()).index(item)
        content_score = max(similarity_matrix[item_index])

        similarity_bonus = max(
            (similarity_matrix[item_index][list(product_categories.keys()).index(p)]
             for p in purchased_items if p in product_categories),
            default=0
        )

        score = content_score * weight_content + weight_purchased * similarity_bonus
        recommendations.append((product_categories[item], score))

    return sorted(recommendations, key=lambda x: x[1], reverse=True)[:top_n]

#
def get_hybrid_recommendations(user_id, top_n=10, alpha=0.5):
    collab_recs = get_collab_recommendations(user_id, top_n)
    content_recs = get_content_recommendations(user_id, top_n)

    if not collab_recs:
        return content_recs

    content_recs_dict = dict(content_recs)
    collab_recs_dict = dict(collab_recs)

    combined_recs = {}
    for item, score in content_recs_dict.items():
        combined_recs[item] = score * alpha
    for item, score in collab_recs_dict.items():
        combined_recs[item] = combined_recs.get(item, 0) + score * (1 - alpha)

    final_recs = sorted(combined_recs.items(), key=lambda x: x[1], reverse=True)
    return final_recs[:top_n]

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