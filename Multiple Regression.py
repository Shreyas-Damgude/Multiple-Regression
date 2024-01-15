from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("reliance_data.csv")

x = df.iloc[:, 3:7]

print("**Independent variables**\n", x)
print()

y = df.iloc[:, 8]
print("**Dependent variable**\n", y)
print()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

model = LinearRegression()
model.fit(x_train, y_train)
intercept = model.intercept_

print(f"=> Intercept = {intercept}")

slopes = model.coef_
print(f"=> Slopes = {slopes}")
print()
print("**Equation of regression**")
equation = f"y = {intercept}"
for i in range(len(slopes)):
    coef = slopes[i]
    equation += f"- {abs(coef)}x{i+1} " if coef < 0 else f" + {coef}x{i+1} "

print(equation)
print()
y_prd = model.predict(x_test)

plt.scatter(y_test, y_prd)
plt.title('Comparison between test values and predicted values')
plt.xlabel("Actual value")
plt.ylabel("Predicted value")
plt.show()

y1_prd = y_prd[1111:1131]
y1_test = y_test[1111:1131]
a = []
for i in range(1, len(y1_prd) + 1):
    a.append(i)

plt.scatter(a, y1_prd, color="Blue")
plt.title('Comparison between test values and predicted values')
plt.ylabel("Predicted Value")
plt.plot(a, y1_prd, color="Red")
plt.plot(a, y1_test, color="Black")

plt.show()

print(f"Accuracy = {r2_score(y1_prd, y1_test)}")

print(f"Absolute Mean Error = {mean_absolute_error(y1_prd, y1_test)}")