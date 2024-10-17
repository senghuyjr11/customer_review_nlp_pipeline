from sklearn.preprocessing import OneHotEncoder
import numpy as np

class OneHotEncoding:
    def __init__(self):
        self.encoder = OneHotEncoder(sparse_output=False)  # Set sparse=False to return dense array

    def fit_transform(self, data):
        data = np.array(data).reshape(-1, 1)
        one_hot_encoded = self.encoder.fit_transform(data)
        return one_hot_encoded