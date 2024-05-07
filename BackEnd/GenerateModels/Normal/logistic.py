import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

def train_model():
    fake_news = pd.read_csv("./BackEnd/Datasets/fake.csv")
    true_news = pd.read_csv("./BackEnd/Datasets/true.csv")
    # Thêm cột label để phân biệt giữa tin giả (1) và tin thật (0)
    fake_news["label"] = 1
    true_news["label"] = 0

    # Kết hợp các trường title, subject, và text thành một trường duy nhất
    fake_news["combined_text"] = (
        fake_news["title"] + " " + fake_news["subject"] + " " + fake_news["text"]
    )
    true_news["combined_text"] = (
        true_news["title"] + " " + true_news["subject"] + " " + true_news["text"]
    )

    # Lựa chọn các trường quan trọng cho việc phân loại
    data = pd.concat(
        [fake_news[["combined_text", "label"]], true_news[["combined_text", "label"]]]
    )

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(
        data["combined_text"], data["label"], test_size=0.2, random_state=42
    )

    # Tiền xử lý văn bản và chuyển đổi thành ma trận TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Khởi tạo và huấn luyện mô hình Logistic Regression
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)

    # Calculate accuracy
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)

    return model, vectorizer, accuracy

def save_model(model, vectorizer, accuracy):
    # Lưu mô hình và vectorizer vào file
    joblib.dump(model, './BackEnd/Models/Logistic/logistic_model.pkl')
    joblib.dump(vectorizer, './BackEnd/Models/Logistic/logistic_vectorizer.pkl')

    # Export accuracy to a file
    with open('./BackEnd/algorithm_accuracy.txt', 'a') as f:
        f.write(f"\nNaive Bayes - {accuracy}\n")

# Gọi hàm
model, vectorizer, accuracy = train_model()
save_model(model, vectorizer, accuracy)
