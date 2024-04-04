
import pandas as pd


def one_hot_encode_categorical(data, categorical_columns):
    """
    Perform one-hot encoding for categorical variables.

    Parameters:
    data (pandas.DataFrame): The input data containing categorical variables.
    categorical_columns (list): List of column names corresponding to categorical variables.

    Returns:
    pandas.DataFrame: Dataframe with categorical variables replaced by one-hot encoded columns.
    """
    # Copy the original dataframe to avoid modifying the original data
    encoded_data = data.copy()
    
    # Perform one-hot encoding for each categorical column
    for column in categorical_columns:
        # Convert the categorical column to type 'category' to ensure proper encoding
        encoded_data[column] = encoded_data[column].astype('category')
        # Perform one-hot encoding and append to the dataframe
        encoded_data = pd.concat([encoded_data, pd.get_dummies(encoded_data[column], prefix=column)], axis=1)
        # Drop the original categorical column after encoding
        encoded_data.drop(columns=[column], inplace=True)
    
    return encoded_data