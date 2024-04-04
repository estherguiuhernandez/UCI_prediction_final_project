from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

def perform_pca(train_data, test_data, n_components=None):
    """
    Perform Principal Component Analysis (PCA) on the given training and testing DataFrames.

    Parameters:
    train_data (DataFrame): The input training DataFrame containing the features.
    test_data (DataFrame): The input testing DataFrame containing the features.
    n_components (int or None): Number of components to keep. If None, all components are kept. If specified as 0. it represents the variance explained, for example 0.85 would mean as many components as to explain 85% variance

    Returns:
    train_pca_result (DataFrame): DataFrame containing the transformed features after PCA for the training set.
    test_pca_result (DataFrame): DataFrame containing the transformed features after PCA for the testing set.
    pca (PCA): PCA object fitted on the training data.
    explained_variance_ratio (array): Explained variance ratio of each selected component.
    """

    # Step 1: Standardize the features separately for training and testing sets
    scaler = StandardScaler()
    scaled_train_data = scaler.fit_transform(train_data)
    scaled_test_data = scaler.transform(test_data)  # Use transform on testing set, don't fit again

    # Step 2: Perform PCA on the training set
    pca = PCA(n_components=n_components)
    train_pca_result = pca.fit_transform(scaled_train_data)

    # Transform the testing set using the fitted PCA model from the training set
    test_pca_result = pca.transform(scaled_test_data)

    # Convert the results to DataFrames
    train_pca_result_df = pd.DataFrame(data=train_pca_result, columns=[f"PC{i + 1}" for i in range(train_pca_result.shape[1])])
    test_pca_result_df = pd.DataFrame(data=test_pca_result, columns=[f"PC{i + 1}" for i in range(test_pca_result.shape[1])])

    # Print the explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Ratio Explained')  # Update y-axis label
    plt.title('Variance Ratio Explained by Each Principal Component')

    # Set y-axis limits dynamically based on maximum explained variance ratio
    plt.ylim(0, np.max(explained_variance_ratio) * 1.1)

    # Set y-axis ticks and labels
    y_ticks = np.linspace(0, np.max(explained_variance_ratio) * 1.1, num=11)
    plt.yticks(y_ticks)
    plt.gca().set_yticklabels(['{:.3f}'.format(x) for x in plt.gca().get_yticks()])

    plt.show()

     # Print the number of components required to achieve the specified variance explained
    if n_components is not None:
        cumulative_variance = np.cumsum(explained_variance_ratio)
        num_components_threshold = np.argmax(cumulative_variance >= n_components) + 1
        print(f"Number of components required to achieve {n_components:.2f} variance explained: {num_components_threshold}")


    return train_pca_result_df, test_pca_result_df, pca, explained_variance_ratio