import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the cleaned dataset
df_cleaned = pd.read_csv('cleaned_weather_dataset.csv')

# Compute the correlation matrix
correlation_matrix = df_cleaned.corr()

# Display the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)

# Heatmap of the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix Heatmap')
plt.show()


# Define the features (independent variables) and target (dependent variable)
#1
# Predicting 'Outdoor Drybulb Temperature [C]' based on other parameters
X = df_cleaned[['6h Prediction Outdoor Drybulb Temperature [C]', '12h Prediction Outdoor Drybulb Temperature [C]', 
                 '24h Prediction Outdoor Drybulb Temperature [C]']]  # Features (multiple columns)
y = df_cleaned['Outdoor Drybulb Temperature [C]']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print model performance metrics
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

# Plot regression results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # Identity line
plt.xlabel('True Values (Outdoor Drybulb Temperature [C])')
plt.ylabel('Predictions')
plt.title('Multiple Regression Results')
plt.show()

#2
# Predicting 'Outdoor Relative Humidity [%]' based on other parameters
X = df_cleaned[['6h Prediction Outdoor Relative Humidity [%]', '12h Prediction Outdoor Relative Humidity [%]', 
                 '24h Prediction Outdoor Relative Humidity [%]']]  # Features (multiple columns)
y = df_cleaned['Outdoor Relative Humidity [%]']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print model performance metrics
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

# Plot regression results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # Identity line
plt.xlabel('True Values (Outdoor Relative Humidity [%])')
plt.ylabel('Predictions')
plt.title('Multiple Regression Results')
plt.show()

#3
# Predicting 'Diffuse Solar Radiation [W/m2]' based on other parameters
X = df_cleaned[['6h Prediction Diffuse Solar Radiation [W/m2]', '12h Prediction Diffuse Solar Radiation [W/m2]', 
                 '24h Prediction Diffuse Solar Radiation [W/m2]']]  # Features (multiple columns)
y = df_cleaned['Diffuse Solar Radiation [W/m2]']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print model performance metrics
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

# Plot regression results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # Identity line
plt.xlabel('True Values (Diffuse Solar Radiation [W/m2])')
plt.ylabel('Predictions')
plt.title('Multiple Regression Results')
plt.show()

#4
# Predicting 'Direct Solar Radiation [W/m2]' based on other parameters
X = df_cleaned[['6h Prediction Direct Solar Radiation [W/m2]', '12h Prediction Direct Solar Radiation [W/m2]', 
                 '24h Prediction Direct Solar Radiation [W/m2]']]  # Features (multiple columns)
y = df_cleaned['Direct Solar Radiation [W/m2]']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print model performance metrics
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

# Plot regression results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # Identity line
plt.xlabel('True Values (Direct Solar Radiation [W/m2])')
plt.ylabel('Predictions')
plt.title('Multiple Regression Results')
plt.show()
