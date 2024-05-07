import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

def train_model():
    data = pd.read_csv('./BackEnd/Dataset/data.csv')

    # Tiền xử lý dữ liệu: thay thế giá trị thiếu và chuyển đổi tất cả các giá trị sang chuỗi
    data.fillna('', inplace=True)
    data = data.astype(str)

    # Kết hợp tất cả các cột thành một chuỗi duy nhất
    data['combined_data'] = data.apply(lambda row: ' '.join(row), axis=1)

    # Chọn tất cả các cột ngoại trừ cột "fake" làm đầu vào và cột "fake" làm nhãn
    X = data.drop(columns=['fake'])
    y = data['fake']

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Tiền xử lý văn bản và chuyển đổi thành ma trận TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    X_train_tfidf = vectorizer.fit_transform(X_train['combined_data'])
    X_test_tfidf = vectorizer.transform(X_test['combined_data'])

    # Khởi tạo và huấn luyện mô hình Logistic Regression (Softmax)
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    # Tính độ chính xác
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)

    return model, vectorizer, accuracy


def save_model(model, vectorizer, accuracy):
    # Lưu mô hình và vectorizer vào file
    joblib.dump(model, 'softmax_model.pkl')
    joblib.dump(vectorizer, 'softMax_vectorizer.pkl')

    # Xuất độ chính xác ra file
    with open('algorithm_accuracy.txt', 'a') as f:
        f.write(f"\nSoftMax - {accuracy}\n")

# Gọi hàm train_model và save_model
model, vectorizer, accuracy = train_model()
save_model(model, vectorizer, accuracy)
