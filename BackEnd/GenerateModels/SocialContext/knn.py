import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import joblib

def train_model():
    data = pd.read_csv('./BackEnd/Dataset/data.csv')

    # Replace missing values with empty strings
    data.fillna('', inplace=True)

    # Combine all fields into one column
    data['combined_data'] = data.apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)

    # Chọn cột "combined_data" làm đầu vào và cột "fake" làm nhãn
    X = data['combined_data']
    y = data['fake']

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Tiền xử lý văn bản và chuyển đổi thành ma trận TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Khởi tạo và huấn luyện mô hình k-Nearest Neighbors
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_tfidf, y_train)

    # Tính độ chính xác
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)

    return model, vectorizer, accuracy

def save_model(model, vectorizer, accuracy):
    # Lưu mô hình và vectorizer vào file
    joblib.dump(model, './BackEnd/Models/KNN/knn_model.pkl')
    joblib.dump(vectorizer, './BackEnd/Models/KNN/knn_vectorizer.pkl')

    # Xuất độ chính xác ra file
    with open('./BackEnd/algorithm_accuracy.txt', 'a') as f:
        f.write(f"\nk-Nearest Neighbors - {accuracy}\n")

# Gọi hàm train_model và save_model
model, vectorizer, accuracy = train_model()
save_model(model, vectorizer, accuracy)
