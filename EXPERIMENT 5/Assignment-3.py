import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve,roc_auc_score

data={
    "Study_Hours":[2,3,5,6,7,9,4,8,11,10,
                   5,6,8,7,6,9,4,5,3,2,
                   4,6,8,7,5,3,1,6,7,9],
    "Attendance":[49,66,55,78,82,86,91,74,83,90,
                  78,36,52,94,63,74,76,89,82,90,
                  96,85,66,75,56,80,60,48,59,46],
    "Pass" : [0,0,0,0,0,1,1,1,1,1,
              0,0,1,1,0,1,1,1,0,0,
              0,1,1,1,1,1,0,0,1,1]
}
df=pd.DataFrame(data)
X=df[["Study_Hours","Attendance"]]
y=df["Pass"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
model=LogisticRegression(max_iter=1000,random_state=42)
model.fit(X_train,y_train)
y_prob=model.predict_proba(X_test)[:,1]
fpr,tpr,thresholds=roc_curve(y_test,y_prob)
auc_score=roc_auc_score(y_test,y_prob)
print("--- ROC-AUC Performance ---")
print(f"AUC Score : {auc_score:.4f}")

plt.figure(figsize=(7,5))
plt.plot(fpr,tpr,label=f"ROC Curve (AUC = {auc_score:.4f})")
plt.plot([0,1],[0,1],linestyle="--",label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Posivtive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True,linestyle="--",alpha=0.6)
plt.show()