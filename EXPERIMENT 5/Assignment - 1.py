import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

data= df = pd.read_csv("Student_data.csv")

X = df[["attendance", "study_hours"]]
y = df["pass"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
model=LogisticRegression(max_iter=1000,random_state=42)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("--- Logistic Regression Performance ---")
print(f"Accuracy : {accuracy_score(y_test,y_pred):.4f}")
print(f"Precision : {precision_score(y_test,y_pred):.4f}")
print(f"Recall : {recall_score(y_test,y_pred):.4f}")
print(f"F1 Score : {f1_score(y_test,y_pred):.4f}")

