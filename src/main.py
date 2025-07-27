import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
data = pd.read_csv("heart.csv")
choles_median = data[data["Cholesterol"] != 0]["Cholesterol"].median()
data["Cholesterol"] = data["Cholesterol"].replace(0, choles_median)

target = "HeartDisease"
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2005)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=0.25,random_state=2005)
num_cols = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
cat_cols = ["Sex", "ChestPainType", "FastingBS", "RestingECG", "ExerciseAngina", "ST_Slope"]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(), cat_cols)
    ]
)

x_train_preprocessed = preprocessor.fit_transform(x_train)
x_val_preprocessed = preprocessor.transform(x_val)
x_test_preprocessed = preprocessor.transform(x_test)

# params = {
#     "n_estimators" : [100,200,300],
#     "criterion" : ["gini", "entropy", "log_loss"]
# }
# grid_search = GridSearchCV(estimator= RandomForestClassifier(random_state=100), param_grid= params,cv = 5, scoring= "recall", verbose= 2)
# grid_search.fit(x_train_preprocessed, y_train)
# grid_search.score(x_val_preprocessed, y_val)
# print(grid_search.best_score_)
# print(grid_search.best_params_)




model = SVC(class_weight="balanced")
model.fit(x_train_preprocessed, y_train)
model.score(x_val_preprocessed, y_val)

y_predict = model.predict(x_test_preprocessed) #Dùng mô hình đã huấn luyện (model) để dự đoán nhãn đầu ra (y) cho tập dữ liệu kiểm tra (x_test_preprocessed).
for i, j in zip(y_predict, y_test):
    print("Prediction: {}. Actual value: {}".format(i,j))
print(classification_report(y_test,y_predict))

# LogisticRegression: recall cho phat hien benh nhan kha on
# precision    recall  f1-score   support
#
#            0       0.91      0.68      0.78        87
#            1       0.76      0.94      0.84        97
#
#     accuracy                           0.82       184
#    macro avg       0.84      0.81      0.81       184
# weighted avg       0.83      0.82      0.81       184





