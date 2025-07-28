# ❤️ Heart Disease Prediction

## 📌 Mục tiêu
Dự án này nhằm xây dựng mô hình Machine Learning để **dự đoán khả năng mắc bệnh tim** của bệnh nhân dựa trên các chỉ số y tế.

## 📂 Dữ liệu
- Nguồn: [Heart Disease Dataset on Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- Số mẫu: 918
- Các cột chính:
  - `Age`, `Sex`, `RestingBP`, `Cholesterol`, `FastingBS`, `MaxHR`, `Oldpeak`, `ChestPainType`, `ExerciseAngina`, `ST_Slope`,...
  - `HeartDisease`: nhãn đầu ra (0 = Không bệnh, 1 = Có bệnh)

## 🔧 Xử lý dữ liệu
- Thay thế các giá trị 0 bất hợp lý trong cột `Cholesterol` bằng trung vị
- Chuẩn hóa các đặc trưng số bằng `StandardScaler`
- Mã hóa các biến phân loại bằng `OneHotEncoder`
- Sử dụng `ColumnTransformer` để kết hợp xử lý nhiều loại biến
- Tách dữ liệu thành train/val/test theo tỷ lệ 60/20/20

## 🧠 Mô hình sử dụng
- Logistic Regression
- Random Forest
- Support Vector Machine (SVC)
- GridSearchCV để tối ưu siêu tham số

## 📈 Kết quả (SVC với `class_weight='balanced'`)
    precision  recall   f1-score  support

        0       0.92      0.70      0.80        87
        1       0.78      0.95      0.86        97

    accuracy                            0.83       184
    macro avg       0.85      0.82      0.83       184
    weighted avg    0.85      0.83      0.83       184


## ▶️ Hướng dẫn sử dụng

### 1. Clone project
```bash
git clone https://github.com/Hadanh0703/heart-disease-prediction.git
cd heart-disease-prediction
pip install -r requirements.txt
python src/model.py

📁 Cấu trúc thư mục
heart-disease-prediction/
├── data/
│   └── heart.csv
├── notebooks/
│   └── EDA_and_Modeling.ipynb
├── src/
│   └── model.py
├── results/
│   └── classification_report.txt
├── requirements.txt
├── .gitignore
└── README.md

👤 Tác giả
Hadanh0703
🔗 GitHub: Hadanh0703
