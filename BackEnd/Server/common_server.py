from datetime import datetime
import os
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

# Huấn luyện và lưu mô hình
model = joblib.load("./BackEnd/Models/Normal/SoftMax/softmax_model.pkl")
vectorizer = joblib.load("./BackEnd/Models/Normal/SoftMax/softmax_vectorizer.pkl")
    
# Định nghĩa Flask app và sử dụng CORS
app = Flask(__name__)
CORS(app)

# Xử lý yêu cầu API POST
@app.route("/api/predict_news", methods=["POST"])
def predict_news():
    # Nhận dữ liệu từ client
    news_data = request.json
    news_text = news_data["news_text"]
    news_tfidf = vectorizer.transform([news_text])

    # Dự đoán xác suất của từng lớp
    probabilities = model.predict_proba(news_tfidf)[0]
    # Lấy chỉ số của lớp có xác suất cao nhất
    predicted_class = model.predict(news_tfidf)[0]

    # Trả về kết quả dưới dạng JSON
    if predicted_class == 1:
        return jsonify({"prediction": "fake", "probability": probabilities[1]})
    else:
        return jsonify({"prediction": "true", "probability": probabilities[0]})


# API endpoint để nhận dữ liệu từ client
@app.route("/api/receive_news", methods=["POST"])
def receive_news():
    # Nhận dữ liệu từ client
    news_data = request.json
    
    # Trích xuất các trường dữ liệu
    title = news_data.get("title", "")
    text = news_data.get("text", "")
    subject = news_data.get("subject", "")
    news_type = news_data.get("type", "")

    # Lấy ngày tháng hiện tại
    date = datetime.now().strftime("%Y-%m-%d")

    # Chọn file CSV dựa trên loại tin tức
    if news_type == "fake":
        csv_path = "./BackEnd/FeedBackDatasets/fake.csv"
    elif news_type == "real":
        csv_path = "./BackEnd/FeedBackDatasets/true.csv"
    else:
        return jsonify({"message": "Invalid news type. Please specify either 'fake' or 'real'."}), 400

    # Tạo DataFrame từ dữ liệu mới
    new_data = pd.DataFrame({
        "title": [title],
        "text": [text],
        "subject": [subject],
        "date": [date]
    })

    # Ghi vào file CSV
    new_data.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)

    return jsonify({"message": "News received and saved successfully."}), 200

# Khởi chạy Flask app
if __name__ == "__main__":
    app.run(debug=True)


