import numpy as np

class GloVeEmbedding:
    def __init__(self, glove_file_path='glove.6B.100d.txt'):
        self.embeddings_index = {}
        self.load_glove_embeddings(glove_file_path)

    def load_glove_embeddings(self, glove_file_path):
        """
        Load pre-trained GloVe embeddings into a dictionary.
        - glove_file_path: Path to the GloVe file
        """
        with open(glove_file_path, encoding='utf8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                self.embeddings_index[word] = coefs

    def get_word_vector(self, word):
        """
        Retrieve the vector for a given word from the GloVe embeddings.
        """
        return self.embeddings_index.get(word)