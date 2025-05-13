import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = {
    'AlcoholIntake': [28, 3, 12, 46, 0, 35, 70, 80, 80, 24, 12, 21, 14, 42, 15, 68, 80, 98, 101, 76, 115, 88, 29, 28, 15, 68, 90],
    'FattyLiver': [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1]
}
df = pd.DataFrame(data)

X = df[['AlcoholIntake']]
y = df['FattyLiver']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f'Accuracy: {accuracy:.2f}%')

plt.scatter(X_train, y_train, marker='o', color='blue', label='Training Data')

beta_0 = model.intercept_[0]
beta_1 = model.coef_[0][0]
x_range = np.linspace(X_train['AlcoholIntake'].min(), X_train['AlcoholIntake'].max(), 100)

z = beta_0 + beta_1 * x_range
sigmoid_z = 1 / (1 + np.exp(-z))

plt.plot(x_range, sigmoid_z, color='red', label='Logistic Curve')
plt.xlabel('Alcohol Intake')
plt.ylabel('Probability of Fatty Liver')
plt.title('Logistic Regression: Alcohol Intake vs Fatty Liver (Training Data)')
plt.legend()
plt.grid(True)
plt.show()

user_input = float(input("Enter g/day of Alcohol Intake: "))
user_input_df = pd.DataFrame({'AlcoholIntake': [user_input]})

prediction = model.predict(user_input_df)
probability = model.predict_proba(user_input_df) * 100

print(f"Prediction: {'Fatty Liver' if prediction[0] == 1 else 'No Fatty Liver'}")
print(f"Probability of Fatty Liver: {probability[0][1]:.2f}%")

plt.scatter(X_train, y_train, marker='o', color='blue', label='Training Data')
plt.scatter(user_input, prediction[0], marker='x', color='green', label='User Input')

plt.plot(x_range, sigmoid_z, color='red', label='Logistic Curve')
plt.xlabel('Alcohol Intake')
plt.ylabel('Probability of Fatty Liver')
plt.title('Logistic Regression: Alcohol Intake vs Fatty Liver (Training Data + User Input)')
plt.legend()
plt.grid(True)
plt.show()

