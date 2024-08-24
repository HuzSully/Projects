import pandas as pd

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_csv(self):
        try:
            data = pd.read_csv(self.file_path)
            return data
        except FileNotFoundError:
            print("File not found.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

    def write_csv(self, data):
        try:
            data.to_csv(self.file_path, index=False)
            print("CSV file written successfully.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

    def read_parquet(self):
        try:
            data = pd.read_parquet(self.file_path)
            return data
        except FileNotFoundError:
            print("File not found.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

    def write_parquet(self, data):
        try:
            data.to_parquet(self.file_path, index=False)
            print("Parquet file written successfully.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")