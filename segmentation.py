import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from TorchCRF import CRF
from torch.utils.data import DataLoader, TensorDataset


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
train_data = [
    # 🍔 Đồ ăn nhanh
    ("gà rán KFC khuyến mãi", ["B-CATEGORY", "I-CATEGORY", "B-BRAND", "B-DISCOUNT"]),
    ("burger McDonald's siêu ngon", ["B-CATEGORY", "B-BRAND", "B-ATTRIBUTE"]),
    ("pizza Domino giảm giá", ["B-CATEGORY", "B-BRAND", "B-DISCOUNT"]),
    ("gà rán Lotteria khuyến mãi", ["B-CATEGORY", "I-CATEGORY", "B-BRAND", "B-DISCOUNT"]),
    ("hamburger Burger King combo ưu đãi", ["B-CATEGORY", "B-BRAND", "B-ATTRIBUTE", "B-DISCOUNT"]),
    ("khoai tây chiên McDonald's cỡ lớn", ["B-CATEGORY", "B-BRAND", "B-SIZE"]),
    ("gà sốt cay Texas Chicken size M", ["B-CATEGORY", "B-BRAND", "B-ATTRIBUTE", "B-SIZE"]),
    ("bánh mì Subway gà nướng", ["B-CATEGORY", "B-BRAND", "B-ATTRIBUTE"]),
    ("Cơm giá rẻ", ["B-CATEGORY", "B-DISCOUNT", "I-DISCOUNT"]),
    ("Phở giảm giá", ["B-CATEGORY", "B-DISCOUNT", "I-DISCOUNT"]),
    ("Cơm rang giá rẻ", ["B-CATEGORY", "B-ATTRIBUTE", "B-DISCOUNT","I-DISCOUNT"]),
    ("Gà rán giá rẻ", ["B-CATEGORY", "B-ATTRIBUTE","B-DISCOUNT","I-DISCOUNT"]),
    ("Gà rán giá rẻ", ["B-CATEGORY", "B-ATTRIBUTE","B-DISCOUNT","I-DISCOUNT"]),
    ("pizza Domino giảm giá", ["B-CATEGORY", "B-BRAND", "B-DISCOUNT"]),
    ("pizza Domino giảm giá", ["B-CATEGORY", "B-BRAND", "B-DISCOUNT"]),
    ("pizza Domino giảm giá", ["B-CATEGORY", "B-BRAND", "B-DISCOUNT"]),


    # 🥤 Đồ uống nhanh
    ("trà sữa Gongcha size M", ["B-CATEGORY", "B-BRAND", "B-SIZE"]),
    ("trà đào Highlands thơm ngon", ["B-CATEGORY", "B-BRAND", "B-ATTRIBUTE"]),
    ("cà phê sữa Highlands đậm đà", ["B-CATEGORY", "B-BRAND", "B-ATTRIBUTE"]),
    ("trà xanh Starbucks giảm giá", ["B-CATEGORY", "B-BRAND", "B-DISCOUNT"]),
    ("trà chanh Bụi Phố đặc biệt", ["B-CATEGORY", "B-BRAND", "B-ATTRIBUTE"]),
    ("sinh tố xoài Phúc Long khuyến mãi", ["B-CATEGORY", "B-BRAND", "B-DISCOUNT"]),

    # 🐟 Thực phẩm tươi sống
    ("cá hồi Nauy tươi sống 500g", ["B-CATEGORY", "B-ORIGIN", "B-ATTRIBUTE", "B-WEIGHT"]),
    ("thịt bò Kobe Nhật Bản 200g", ["B-CATEGORY", "B-ORIGIN", "B-ORIGIN", "B-WEIGHT"]),
    ("tôm hùm Alaska nhập khẩu", ["B-CATEGORY", "B-ORIGIN", "B-ATTRIBUTE"]),
    ("cua hoàng đế Na Uy 1kg", ["B-CATEGORY", "B-ORIGIN", "B-WEIGHT"]),
    ("hàu sữa Pháp tươi ngon", ["B-CATEGORY", "B-ORIGIN", "B-ATTRIBUTE"]),
    ("sò huyết Cà Mau loại 1", ["B-CATEGORY", "B-ORIGIN", "B-ATTRIBUTE"]),
    ("cá chẽm Việt Nam 1.2kg", ["B-CATEGORY", "B-ORIGIN", "B-WEIGHT"]),

    # 🥩 Thịt, trứng, gia cầm
    ("thịt heo Ba Lan đông lạnh 1kg", ["B-CATEGORY", "B-ORIGIN", "B-ATTRIBUTE", "B-WEIGHT"]),
    ("bò Mỹ phi lê mềm ngon", ["B-CATEGORY", "B-ORIGIN", "B-ATTRIBUTE"]),
    ("thịt gà ta nguyên con 2kg", ["B-CATEGORY", "B-ATTRIBUTE", "B-WEIGHT"]),
    ("trứng gà CP hộp 10 quả", ["B-CATEGORY", "B-BRAND", "B-COUNT"]),
    ("trứng vịt muối hộp 6 quả", ["B-CATEGORY", "B-ATTRIBUTE", "B-COUNT"]),

    # 🍎 Rau củ, trái cây
    ("nho Mỹ nhập khẩu", ["B-CATEGORY", "B-ORIGIN", "B-ATTRIBUTE"]),
    ("bơ sáp Đà Lạt loại 1", ["B-CATEGORY", "B-ORIGIN", "B-ATTRIBUTE"]),
    ("dâu tây Hàn Quốc hộp 500g", ["B-CATEGORY", "B-ORIGIN", "B-WEIGHT"]),
    ("xoài cát Hòa Lộc loại ngon", ["B-CATEGORY", "B-ORIGIN", "B-ATTRIBUTE"]),
    ("chuối tiêu Laba Đà Lạt 1kg", ["B-CATEGORY", "B-ORIGIN", "B-WEIGHT"]),
    ("táo Envy New Zealand size lớn", ["B-CATEGORY", "B-ORIGIN", "B-SIZE"]),

    # 🥬 Rau xanh tươi sống
    ("rau cải ngọt Đà Lạt 500g", ["B-CATEGORY", "B-ORIGIN", "B-WEIGHT"]),
    ("xà lách Mỹ hữu cơ 200g", ["B-CATEGORY", "B-ORIGIN", "B-ATTRIBUTE", "B-WEIGHT"]),
    ("cà chua bi Đà Lạt 1kg", ["B-CATEGORY", "B-ORIGIN", "B-WEIGHT"]),
    ("ớt chuông đỏ Hàn Quốc 3 trái", ["B-CATEGORY", "B-ORIGIN", "B-COUNT"]),
    ("hành lá Việt Nam 200g", ["B-CATEGORY", "B-ORIGIN", "B-WEIGHT"]),

    # 🍚 Thực phẩm khô
    ("mì Hảo Hảo thùng 30 gói", ["B-CATEGORY", "B-BRAND", "B-WEIGHT"]),
    ("gạo ST25 túi 5kg", ["B-CATEGORY", "B-BRAND", "B-WEIGHT"]),
    ("bún khô Phú Quốc 500g", ["B-CATEGORY", "B-ORIGIN", "B-WEIGHT"]),
    ("dầu ăn Neptune chai 1L", ["B-CATEGORY", "B-BRAND", "B-VOLUME"]),
    ("nước mắm Nam Ngư 500ml", ["B-CATEGORY", "B-BRAND", "B-VOLUME"]),

    # 🍪 Thực phẩm đóng gói
    ("bánh quy Oreo hộp 300g", ["B-CATEGORY", "B-BRAND", "B-WEIGHT"]),
    ("kẹo dẻo Haribo Đức 200g", ["B-CATEGORY", "B-BRAND", "B-ORIGIN", "B-WEIGHT"]),
    ("sô cô la KitKat Nhật Bản", ["B-CATEGORY", "B-BRAND", "B-ORIGIN"]),
    ("sữa Ensure Gold hộp 850g", ["B-CATEGORY", "B-BRAND", "B-WEIGHT"]),
    ("yến mạch Quaker Mỹ 1kg", ["B-CATEGORY", "B-BRAND", "B-ORIGIN", "B-WEIGHT"]),
    ("ngũ cốc Calbee Nhật Bản", ["B-CATEGORY", "B-BRAND", "B-ORIGIN"]),

    # 🍷 Đồ uống đóng chai
    ("bia Heineken lon 330ml", ["B-CATEGORY", "B-BRAND", "B-VOLUME"]),
    ("nước ép cam Tropicana Mỹ 1L", ["B-CATEGORY", "B-BRAND", "B-ORIGIN", "B-VOLUME"]),
    ("trà xanh C2 chai 500ml", ["B-CATEGORY", "B-BRAND", "B-VOLUME"]),
    ("sữa chua uống Yakult Nhật Bản", ["B-CATEGORY", "B-BRAND", "B-ORIGIN"]),

    ("Phở bò Nam Định ngon nhất", ["B-CATEGORY", "B-ATTRIBUTE", "B-ORIGIN", "I-ORIGIN", "B-ATTRIBUTE"]),
    ("Bún chả Hà Nội chính gốc", ["B-CATEGORY", "I-CATEGORY", "B-ORIGIN", "B-ATTRIBUTE"]),
    ("Mì Quảng Đà Nẵng đặc sản", ["B-CATEGORY", "I-CATEGORY", "B-ORIGIN", "I-ORIGIN", "B-ATTRIBUTE"]),
    ("Cháo lươn Nghệ An cay nồng", ["B-CATEGORY", "I-CATEGORY", "B-ORIGIN", "I-ORIGIN", "B-ATTRIBUTE"]),
    ("Phở gà Hà Nội hương vị truyền thống", ["B-CATEGORY", "B-ATTRIBUTE", "B-ORIGIN", "B-ATTRIBUTE", "I-ATTRIBUTE"]),
    ("Lẩu bò Sài Gòn đậm đà", ["B-CATEGORY", "B-ATTRIBUTE", "B-ORIGIN", "B-ATTRIBUTE"]),
    ("Bò Hầm", ["B-CATEGORY", "B-ATTRIBUTE",]),
    ("Gà Hầm Thuốc Bắc", ["B-CATEGORY", "B-ATTRIBUTE","I-ATTRIBUTE","I-ATTRIBUTE"]),
    ("Cơm Lươn sốt Teryaki", ["B-CATEGORY", "B-ATTRIBUTE","I-ATTRIBUTE","B-BRAND"]),
("Cơm Lươn sốt Teryaki", ["B-CATEGORY", "I-CATEGORY", "B-ATTRIBUTE", "B-BRAND"]),
("Cơm Lươn sốt cay", ["B-CATEGORY", "I-CATEGORY", "B-ATTRIBUTE"]),
("Cơm rang Lươn sốt đặc biệt", ["B-CATEGORY", "I-CATEGORY", "B-ATTRIBUTE", "B-ATTRIBUTE"])
]

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

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTCRFModel("bert-base-uncased", num_labels=len(label2id)).to(device)
model_save_path = "bert_crf_model.pth"
torch.save(model.state_dict(), model_save_path)
print("Mô hình đã được lưu thành công!")
# Load trọng số đã lưu trước đó
model_load_path = "bert_crf_model.pth"
model.load_state_dict(torch.load(model_load_path, map_location=device))
model.eval()  # Đặt mô hình vào chế độ đánh giá
print("Mô hình đã được tải thành công!")
# Training Loop
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)  # Giảm LR xuống từ 5e-5
epochs = 10  # Huấn luyện thêm để mô hình học kỹ hơn
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

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader)}")


# Prediction Function
def predict(query):
    model.eval()
    input_ids, attention_mask, _ = encode_query(query, ["O"] * len(query.split()))
    input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

    with torch.no_grad():
        predictions = model(input_ids, attention_mask)

    predicted_labels = [id2label[label] for label in predictions[0]]
    return list(zip(query.split(), predicted_labels))


# Example Prediction
query = "gà ri rán KFC khuyến mãi"
print(predict(query))

