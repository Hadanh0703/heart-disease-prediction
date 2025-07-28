import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# 1. Load & xử lý dữ liệu
data = pd.read_csv("../data/heart.csv")

# Thay thế giá trị Cholesterol = 0 bằng trung vị (do không hợp lệ)
choles_median = data[data["Cholesterol"] != 0]["Cholesterol"].median()
data["Cholesterol"] = data["Cholesterol"].replace(0, choles_median)

# 2. Tách feature & label
target = "HeartDisease"
x = data.drop(columns=[target])
y = data[target]

# 3. Chia train, validation, test (60/20/20)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2005)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=2005)

# 4. Tiền xử lý dữ liệu: scale & one-hot encode
num_cols = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
cat_cols = ["Sex", "ChestPainType", "FastingBS", "RestingECG", "ExerciseAngina", "ST_Slope"]

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

x_train_pre = preprocessor.fit_transform(x_train)
x_val_pre = preprocessor.transform(x_val)
x_test_pre = preprocessor.transform(x_test)

# 5. Huấn luyện mô hình SVC với class_weight
model = SVC(class_weight="balanced", random_state=42)
model.fit(x_train_pre, y_train)

# 6. Đánh giá mô hình
y_pred = model.predict(x_test_pre)
print(classification_report(y_test, y_pred))

# 7. (Tuỳ chọn) GridSearchCV với Random Forest
# Uncomment để dùng
"""
from sklearn.ensemble import RandomForestClassifier

params = {
    "n_estimators": [100, 200, 300],
    "criterion": ["gini", "entropy", "log_loss"]
}
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=100),
    param_grid=params,
    cv=5,
    scoring="recall",
    verbose=2
)

grid_search.fit(x_train_pre, y_train)
print("Best Score:", grid_search.best_score_)
print("Best Params:", grid_search.best_params_)
"""
