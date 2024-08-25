import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import category_encoders as ce

class CategoricalEncoder:
    def __init__(self, df):
        """
        Initializes the CategoricalEncoder with a pandas DataFrame.
        
        Parameters:
        - df: pandas DataFrame containing the data to be encoded.
        """
        self.df = df.copy()

    def label_encoding(self, columns):
        """
        Apply label encoding to specified columns.
        
        Parameters:
        - columns: A list of column names to encode.
        
        Returns:
        - A DataFrame with the label encoded columns.
        """
        for column in columns:
            encoder = LabelEncoder()
            self.df[column + '_label_encoded'] = encoder.fit_transform(self.df[column])
        return self.df

    def one_hot_encoding(self, columns):
        """
        Apply one-hot encoding to specified columns.
        
        Parameters:
        - columns: A list of column names to encode.
        
        Returns:
        - A DataFrame with the one-hot encoded columns.
        """
        self.df = pd.get_dummies(self.df, columns=columns)
        return self.df

    def target_encoding(self, columns, target):
        """
        Apply target encoding to specified columns.
        
        Parameters:
        - columns: A list of column names to encode.
        - target: The target variable used for encoding.
        
        Returns:
        - A DataFrame with the target encoded columns.
        """
        encoder = ce.TargetEncoder()
        for column in columns:
            self.df[column + '_target_encoded'] = encoder.fit_transform(self.df[column], self.df[target])
        return self.df

    def ordinal_encoding(self, columns, categories_list):
        """
        Apply ordinal encoding to specified columns.
        
        Parameters:
        - columns: A list of column names to encode.
        - categories_list: A list of category orders corresponding to each column.
        
        Returns:
        - A DataFrame with the ordinal encoded columns.
        """
        for column, categories in zip(columns, categories_list):
            encoder = OrdinalEncoder(categories=[categories])
            self.df[column + '_ordinal_encoded'] = encoder.fit_transform(self.df[[column]])
        return self.df

    def binary_encoding(self, columns):
        """
        Apply binary encoding to specified columns.
        
        Parameters:
        - columns: A list of column names to encode.
        
        Returns:
        - A DataFrame with the binary encoded columns.
        """
        for column in columns:
            encoder = ce.BinaryEncoder()
            binary_encoded_df = encoder.fit_transform(self.df[column])
            self.df = pd.concat([self.df, binary_encoded_df], axis=1)
        return self.df

    def frequency_encoding(self, columns):
        """
        Apply frequency encoding to specified columns.
        
        Parameters:
        - columns: A list of column names to encode.
        
        Returns:
        - A DataFrame with the frequency encoded columns.
        """
        for column in columns:
            freq = self.df[column].value_counts(normalize=True)
            self.df[column + '_frequency_encoded'] = self.df[column].map(freq)
        return self.df

    def hash_encoding(self, columns, n_components=3):
        """
        Apply hash encoding to specified columns.
        
        Parameters:
        - columns: A list of column names to encode.
        - n_components: Number of components to keep after hashing.
        
        Returns:
        - A DataFrame with the hash encoded columns.
        """
        for column in columns:
            encoder = ce.HashingEncoder(n_components=n_components)
            hash_encoded_df = encoder.fit_transform(self.df[column])
            self.df = pd.concat([self.df, hash_encoded_df], axis=1)
        return self.df

# Example usage:
# df = pd.DataFrame({
#     'category1': ['low', 'medium', 'high', 'medium', 'low'],
#     'category2': ['apple', 'banana', 'cherry', 'banana', 'apple'],
#     'target': [10, 20, 30, 20, 10]
# })
# encoder = CategoricalEncoder(df)
# df_encoded = encoder.label_encoding(['category1', 'category2'])
# df_encoded = encoder.one_hot_encoding(['category1', 'category2'])
# df_encoded = encoder.target_encoding(['category1', 'category2'], 'target')
# df_encoded = encoder.ordinal_encoding(['category1', 'category2'], [['low', 'medium', 'high'], ['apple', 'banana', 'cherry']])
# df_encoded = encoder.binary_encoding(['category1', 'category2'])
# df_encoded = encoder.frequency_encoding(['category1', 'category2'])
# df_encoded = encoder.hash_encoding(['category1', 'category2'], n_components=3)
