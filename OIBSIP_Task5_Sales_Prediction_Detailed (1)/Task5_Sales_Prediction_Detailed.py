# 📊 Task 5: Sales Prediction - OIBSIP Data Science Internship

# This project predicts sales based on advertising spend on TV, Radio, and Newspaper.
# It's a basic linear regression example using the Advertising dataset.

# ✅ Libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ✅ Step 1: Load the dataset
# Make sure 'advertising.csv' is in your working directory.
# The dataset contains columns: TV, Radio, Newspaper, Sales
df = pd.read_csv("advertising.csv")

# ✅ Step 2: Display first few rows
print("Sample data:")
print(df.head())

# ✅ Step 3: Define features (X) and target (y)
X = df[['TV', 'Radio', 'Newspaper']]  # Independent variables
y = df['Sales']                       # Target variable

# ✅ Step 4: Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# ✅ Step 5: Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# ✅ Step 6: Make predictions on the test data
predictions = model.predict(X_test)

# ✅ Step 7: Evaluate the model using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, predictions)
print("\nModel Evaluation:")
print("Mean Squared Error (MSE):", mse)

# ✅ Step 8: Display model coefficients
print("\nModel Coefficients:")
print(f"Intercept: {model.intercept_}")
print(f"TV Coefficient: {model.coef_[0]}")
print(f"Radio Coefficient: {model.coef_[1]}")
print(f"Newspaper Coefficient: {model.coef_[2]}")