import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from TorchCRF import CRF
from torch.utils.data import DataLoader, TensorDataset
from flask import Flask, request, jsonify
import pandas as pd
import os
from torch.utils.data import TensorDataset, DataLoader
# ========== Define BERT-CRF Model ==========
# Lớp fc ánh xạ đầu ra BERT sang không gian nhãn.
class BERTCRFModel(nn.Module):
    def __init__(self, bert_model_name, num_labels):
        super(BERTCRFModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.hidden_size = self.bert.config.hidden_size
        self.fc = nn.Linear(self.hidden_size, num_labels)
        self.crf = CRF(num_labels)
    # Dữ liệu đi qua BERT, trả về vector ngữ nghĩa của từng từ.
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        emissions = self.fc(outputs.last_hidden_state)
        # Nếu có labels:
        # Tính loss để huấn luyện mô hình.
        # Nếu không có labels:
        # Dự đoán nhãn tốt nhất bằng CRF.
        if labels is not None:
            loss = (-self.crf(emissions, labels, mask=attention_mask.byte()) / input_ids.size(0)).mean()
            return loss
        else:
            predictions = self.crf.viterbi_decode(emissions, mask=attention_mask.byte())
            return predictions


# ========== Load Tokenizer & Labels ==========
# Tokenizer của BERT giúp biến đổi văn bản thành số để mô hình có thể xử lý.
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# Label Mapping
label2id = {
    "B-CATEGORY": 1, "I-CATEGORY": 2,
    "B-BRAND": 3, "I-BRAND": 4,
    "B-DISCOUNT": 5, "I-DISCOUNT": 6,
    "B-ATTRIBUTE": 7, "I-ATTRIBUTE": 8,
    "B-ORIGIN": 9, "I-ORIGIN": 10,
    "B-WEIGHT": 11, "I-WEIGHT": 12,
    "B-SIZE": 13, "I-SIZE": 14,
    "B-VOLUME": 15, "I-VOLUME": 16,
    "B-POWER": 17, "I-POWER": 18,
    "B-COUNT": 19, "I-COUNT": 20,  # Thêm nhãn COUNT vào danh sách
    "B-GENDER": 21, "I-GENDER": 22,
    "O": 0  # Nhãn 'O' dành cho token không có nhãn
}

# Gán số cho từng nhãn, giúp mô hình học và dự đoán dễ dàng hơn.
# Tạo ánh xạ ngược (id2label) để chuyển từ số về nhãn.
id2label = {v: k for k, v in label2id.items()}

# Chuyển đổi văn bản (query) thành input_ids và attention_mask.
# Xử lý subword tokenization của BERT để đảm bảo nhãn (labels) khớp với token.
# Trả về tensor chứa input và nhãn để mô hình huấn luyện hoặc dự đoán.
def encode_query(query, labels, max_length=32):
    tokens = tokenizer(query, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")

    tokenized_words = tokenizer.tokenize(query)
    adjusted_labels = ["O"] * len(tokenized_words)

    word_idx = 0
    for idx, token in enumerate(tokenized_words):
        if token.startswith("##"):  # Token phụ thuộc vào từ trước
            adjusted_labels[idx] = adjusted_labels[idx - 1]
        else:
            if word_idx < len(labels):
                adjusted_labels[idx] = labels[word_idx]
                word_idx += 1

    label_ids = [label2id[label] for label in adjusted_labels] + [0] * (max_length - len(adjusted_labels))

    return tokens["input_ids"], tokens["attention_mask"], torch.tensor(label_ids)
# Example Training Data
df = pd.read_csv("training_data.csv")
train_data = []
for _, row in df.iterrows():
    query = row["query"]
    labels = row["labels"].split()
    train_data.append((query, labels))

# Prepare Data for Training
# train_inputs: Mã hóa của câu (BERT token IDs).
# train_masks: Mask tensor để chỉ định token nào là thật (1), token nào là padding (0).
# train_labels: Nhãn thực thể của từng token trong câu.
train_inputs, train_masks, train_labels = [], [], []

# Lặp qua từng câu (query) và nhãn thực thể (label) trong tập huấn luyện (train_data).
# Gọi encode_query(query, label) để chuyển câu thành tensor:
# input_ids: Tokenized ID của câu.
# attention_mask: Mask tensor cho BERT.
# label_ids: Nhãn thực thể dạng số.
# Thêm các giá trị vào danh sách train_inputs, train_masks, train_labels.
for query, label in train_data:
    input_ids, attention_mask, label_ids = encode_query(query, label)
    train_inputs.append(input_ids)
    train_masks.append(attention_mask)
    train_labels.append(label_ids)

# Chuyển danh sách thành tensor
train_inputs = torch.cat(train_inputs, dim=0)
train_masks = torch.cat(train_masks, dim=0)
train_labels = torch.stack(train_labels, dim=0)

# Create DataLoader
batch_size = 8
train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTCRFModel("bert-base-uncased", num_labels=len(label2id)).to(device)

model_path = "bert_crf_model.pth"

# Load trọng số nếu có
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("✅ Đã load mô hình đã huấn luyện trước đó.")
else:
    print("⚠️ Không tìm thấy mô hình. Bạn cần train từ đầu.")
    # Nếu không có, bạn nên khởi tạo model mới tại đây (hoặc báo lỗi)

# Tiếp tục huấn luyện
batch_size = 8
train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTCRFModel("bert-base-uncased", num_labels=len(label2id)).to(device)

model_path = "bert_crf_model.pth"

# Nếu model đã tồn tại => Load
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✅ Mô hình đã được tải từ file. Bỏ qua bước huấn luyện.")
else:
    print("🚀 Bắt đầu huấn luyện mô hình...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    epochs = 20

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_dataloader:
            batch = [b.to(device) for b in batch]
            input_ids, attention_mask, labels = batch

            optimizer.zero_grad()
            loss = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader):.4f}")

    torch.save(model.state_dict(), model_path)
    print("✅ Mô hình đã được huấn luyện và lưu thành công!")

# Prediction Function
def predict(query):
    model.eval()
    input_ids, attention_mask, _ = encode_query(query, ["O"] * len(query.split()))
    input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

    with torch.no_grad():
        predictions = model(input_ids, attention_mask)

    predicted_labels = [id2label[label] for label in predictions[0]]
    return list(zip(query.split(), predicted_labels))
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "Welcome to BERT-CRF NER API using Flask"})

@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Query is required"}), 400

    result = predict(query)
    return jsonify({"entities": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

# Example Prediction
query = "gà ri rán KFC khuyến mãi"
print(predict(query))

