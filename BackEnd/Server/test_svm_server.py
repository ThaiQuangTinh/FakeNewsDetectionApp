from datetime import datetime
import os
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

# Load mô hình và vectorizer từ các tệp đã lưu
model = joblib.load("./BackEnd/Models/SocialContext/SVM/svm_model.pkl")
vectorizer = joblib.load("./BackEnd/Models/SocialContext/SVM/svm_vectorizer.pkl")

# Định nghĩa Flask app và sử dụng CORS
app = Flask(__name__)
CORS(app)

# Xử lý yêu cầu API POST
from scipy.special import expit

# Xử lý yêu cầu API POST
@app.route("/api/predict_news", methods=["POST"])
def predict_news():
    # Nhận dữ liệu từ client
    news_data = request.json
    news_text = news_data["news_text"]
    news_tfidf = vectorizer.transform([news_text])

    # Dự đoán giá trị của hàm quyết định
    decision_values = model.decision_function(news_tfidf)

    # Chuyển đổi giá trị của hàm quyết định thành xác suất sử dụng hàm sigmoid
    probabilities = expit(decision_values)

    # Lấy chỉ số của lớp có xác suất cao nhất
    predicted_class = model.predict(news_tfidf)[0]

    # Trả về kết quả dưới dạng JSON
    if predicted_class == 1:
        return jsonify({"prediction": "fake", "probability": probabilities[0]})
    else:
        return jsonify({"prediction": "true", "probability": probabilities[0]})




# Khởi chạy Flask app
if __name__ == "__main__":
    app.run(debug=True)
