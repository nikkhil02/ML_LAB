import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

df=pd.read_csv("House_Price.csv")
X=df[["area","bedrooms"]]
y=df["price"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

model=LinearRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_pred)

print("Slope(b1):",model.coef_)
print("Intercept (b0):",model.intercept_)
print("\n--- Evaluation Metrics ---")
print(f"MAE : {mae:.2f}")
print(f"MSE : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2 : {r2:.2f}")

plt.figure(figsize=(7,5))
plt.scatter(y_test,y_pred,label="Predicted Values")
plt.plot([y_test.min(),y_test.max()],
         [y_test.min(),y_test.max()],
         linewidth=2,label="Perfect Prediction")

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual v/s Predicted House Price")
plt.legend()
plt.grid(True,linestyle="--",alpha=0.6)
plt.show()

