import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df=pd.read_csv("House_Price.csv")

X=df[["area"]]
y=df["price"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

linear_model=LinearRegression()
linear_model.fit(X_train,y_train)
y_pred_linear=linear_model.predict(X_test)
r2_linear=r2_score(y_test,y_pred_linear)

poly=PolynomialFeatures(degree=2)
X_train_poly=poly.fit_transform(X_train)
X_test_poly=poly.transform(X_test)
poly_model=LinearRegression()
poly_model.fit(X_train_poly,y_train)
y_pred_poly=poly_model.predict(X_test_poly)
r2_poly=r2_score(y_test,y_pred_poly)
print("--- R2 Score Comparison ---")
print(f"Linear Regression R2 : {r2_linear:.2f}")
print(f"Polynomial Regression R2 : {r2_poly:.2f}")

plt.figure(figsize=(8,5))
plt.scatter(X,y,label="Actual Data")
X_plot=np.linspace(X["area"].min(),X["area"].max(),300).reshape(-1,1)
y_linear=linear_model.predict(pd.DataFrame(X_plot,columns=["area"]))
X_plot_poly=poly.transform(pd.DataFrame(X_plot,columns=["area"]))
y_poly=poly_model.predict(X_plot_poly)

plt.plot(X_plot,y_linear,label="Linear Regression")
plt.plot(X_plot,y_poly,label="Polynomial Regression")
plt.xlabel("House Area (sq ft.)")
plt.ylabel("House Price")
plt.title("Linear v/s Polynomial Regression ")
plt.legend()
plt.grid(True,linestyle="--",alpha=0.6)
plt.show()

