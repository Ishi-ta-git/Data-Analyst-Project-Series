import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mstats

# Load the dataset
df = pd.read_csv("C:/Users/91990/Downloads/weather_data.csv")

# Display the first few rows and column names
print(df.head())
print(df.columns)

# Replace zeros with NaN for missing values
df.replace(0, np.nan, inplace=True)

# Drop rows with any missing values
df_cleaned = df.dropna()

# Check again to ensure missing values are handled
print(df.isnull().sum())

# Create box plots for each numeric column to identify outliers
for column in df.select_dtypes(include=['float64', 'int64']).columns:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=df[column])
    plt.title(f'Box Plot of {column}')
    plt.show()

# Define a function to remove outliers using IQR
def remove_outliers_iqr(df, numeric_columns):
    for column in numeric_columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        # Filter out outliers
        df = df[(df[column] >= (Q1 - 1.5 * IQR)) & (df[column] <= (Q3 + 1.5 * IQR))]
    return df

# List of numeric columns to clean
numeric_columns = ['Outdoor Drybulb Temperature [C]', 'Outdoor Relative Humidity [%]', '6h Prediction Outdoor Drybulb Temperature [C]', '12h Prediction Outdoor Drybulb Temperature [C]',
                   '24h Prediction Outdoor Drybulb Temperature [C]', '6h Prediction Outdoor Relative Humidity [%]', '12h Prediction Outdoor Relative Humidity [%]', '24h Prediction Direct Solar Radiation [W/m2]']  # Replace with actual column names

# Apply outlier removal
df_cleaned = remove_outliers_iqr(df_cleaned, numeric_columns)

# Save the cleaned DataFrame to a CSV file
df_cleaned.to_csv('cleaned_weather_dataset.csv', index=False)
print("Cleaned dataset saved to cleaned_weather_dataset.csv")

#Visualize all numeric columns in a box plot
plt.figure(figsize=(12, 8))
sns.boxplot(data=df_cleaned[numeric_columns])
plt.title('Cumulative Box Plot for All Numeric Columns')
plt.xlabel('Columns')
plt.ylabel('Values')
plt.show()

