import pandas as pd
from sklearn.model_selection import train_test_split

class DataProcessor:
    def read_csv(self, file_path):
        return self._read_file(file_path, pd.read_csv, "CSV")

    def write_csv(self, data, file_path):
        self._write_file(data, file_path, pd.DataFrame.to_csv, index=False, file_type="CSV")

    def read_parquet(self, file_path):
        return self._read_file(file_path, pd.read_parquet, "Parquet")

    def write_parquet(self, data, file_path):
        self._write_file(data, file_path, pd.DataFrame.to_parquet, index=False, file_type="Parquet")

    def train_test_split(self, data, test_size=0.2, random_state=None, shuffle=True):
        try:
            train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state, shuffle=shuffle)
            print("Data split into training and testing sets successfully.")
            return train_data, test_data
        except Exception as e:
            print(f"An error occurred during train-test split: {str(e)}")
            return None, None

    def split_features_target(self, data, target_column):
        """
        Splits the data into features and target variable.

        Parameters:
        - data (pd.DataFrame): The input dataframe.
        - target_column (str): The name of the column to be used as the target variable.

        Returns:
        - X (pd.DataFrame): The features dataframe.
        - y (pd.Series): The target variable series.
        """
        try:
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data.")
            
            X = data.drop(columns=[target_column])
            y = data[target_column]
            print("Features and target variable split successfully.")
            return X, y
        except Exception as e:
            print(f"An error occurred while splitting features and target: {str(e)}")
            return None, None

    def _read_file(self, file_path, read_func, file_type):
        try:
            return read_func(file_path)
        except FileNotFoundError:
            print(f"{file_type} file not found: {file_path}")
        except Exception as e:
            print(f"An error occurred while reading the {file_type} file: {str(e)}")
        return None

    def _write_file(self, data, file_path, write_func, *args, file_type="file", **kwargs):
        try:
            write_func(data, file_path, *args, **kwargs)
            print(f"{file_type} file written successfully: {file_path}")
        except Exception as e:
            print(f"An error occurred while writing the {file_type} file: {str(e)}")
