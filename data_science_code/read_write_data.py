import pandas as pd

class DataProcessor:
    def read_csv(self, file_path):
        return self._read_file(file_path, pd.read_csv, "CSV")

    def write_csv(self, data, file_path):
        self._write_file(data, file_path, pd.DataFrame.to_csv, index=False, file_type="CSV")

    def read_parquet(self, file_path):
        return self._read_file(file_path, pd.read_parquet, "Parquet")

    def write_parquet(self, data, file_path):
        self._write_file(data, file_path, pd.DataFrame.to_parquet, index=False, file_type="Parquet")

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
