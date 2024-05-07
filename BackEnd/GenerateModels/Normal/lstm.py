import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Embedding, LSTM, Dense  # type: ignore
import joblib

def train_model():
    # Load dữ liệu
    fake_news = pd.read_csv("./BackEnd/Datasets/fake.csv")
    true_news = pd.read_csv("./BackEnd/Datasets/true.csv")

    # Kết hợp các trường title, subject và text thành một trường duy nhất
    fake_news["combined_text"] = fake_news["title"] + " " + fake_news["subject"] + " " + fake_news["text"]
    true_news["combined_text"] = true_news["title"] + " " + true_news["subject"] + " " + true_news["text"]

    # Gán nhãn cho dữ liệu
    fake_news["label"] = 1
    true_news["label"] = 0

    # Lựa chọn các trường quan trọng cho việc phân loại
    data = pd.concat([fake_news[["combined_text", "label"]], true_news[["combined_text", "label"]]])

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(data["combined_text"], data["label"], test_size=0.2, random_state=42)

    # Mã hóa văn bản
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    # Tạo độ dài đồng nhất cho các chuỗi
    max_length = 100
    X_train_padded = pad_sequences(X_train_seq, maxlen=max_length)
    X_test_padded = pad_sequences(X_test_seq, maxlen=max_length)

    # Xây dựng mô hình LSTM
    model = Sequential()
    model.add(Embedding(10000, 128, input_length=max_length))
    model.add(LSTM(64))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Huấn luyện mô hình
    history = model.fit(X_train_padded, y_train, epochs=10, batch_size=32, validation_data=(X_test_padded, y_test))

    # Lấy độ chính xác từ lịch sử
    accuracy = history.history['val_accuracy'][-1]

    return model, tokenizer, accuracy

def save_model(model, tokenizer, accuracy):
    # Save model architecture and weights
    model.save('./BackEnd/Models/LSTM/lstm_model.keras')

    # Save tokenizer
    with open('./BackEnd/Models/LSTM/lstm_tokenizer.pkl', 'wb') as handle:
        joblib.dump(tokenizer, handle)

    # Export accuracy to a file
    with open('./BackEnd/algorithm_accuracy.txt', 'a') as f:
        f.write(f"\nLSTM - {accuracy}\n")    

# Train model and save
model, tokenizer, accuracy = train_model()
save_model(model, tokenizer, accuracy)
