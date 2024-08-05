import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

try:
    # Load the Iris dataset
    df = sns.load_dataset('iris')

    # Display the first few rows of the dataset
    print(df.head())

    # Basic statistics
    print(df.describe())

    # Check for missing values
    print(df.isnull().sum())

    # Visualize the distribution of each feature
    sns.pairplot(df, hue='species')
    plt.show()

    # Exclude the non-numeric column 'species' for correlation matrix
    numeric_df = df.drop(columns=['species'])

    # Correlation matrix
    corr_matrix = numeric_df.corr()
    print(corr_matrix)

    # Heatmap of the correlation matrix
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.show()

    # Save the dataset to a CSV file
    df.to_csv('iris_dataset.csv', index=False)
    print("Dataset saved to iris_dataset.csv")

    # Save the correlation matrix to a CSV file
    corr_matrix.to_csv('iris_correlation_matrix.csv')
    print("Correlation matrix saved to iris_correlation_matrix.csv")

except Exception as e:
    print(f"An error occurred: {e}")

