from gensim.models import Word2Vec

class Word2VecModel:
    def __init__(self):
        self.model = None

    def train(self, sentences, vector_size=100, window=5, min_count=1, workers=4):
        """
        Train Word2Vec model on input sentences.
        - sentences: A list of tokenized sentences.
        - vector_size: Size of word vectors.
        - window: Maximum distance between the current and predicted word.
        - min_count: Ignores all words with total frequency lower than this.
        - workers: Number of threads to use.
        """
        self.model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
        return self.model

    def get_word_vector(self, word):
        """
        Retrieve the vector representation of a word.
        """
        if word in self.model.wv:
            return self.model.wv[word].tolist()  # Return vector as a list
        else:
            return None

    def save_model(self, filepath='word2vec_trained.model'):
        """
        Save the trained Word2Vec model to disk.
        - filepath: Path where the model should be saved.
        """
        if self.model:
            self.model.save(filepath)
        else:
            raise ValueError("No model to save. Train the model first.")

    def load_model(self, filepath='word2vec_trained.model'):
        """
        Load a pre-trained Word2Vec model from disk.
        - filepath: Path to the saved model.
        """
        self.model = Word2Vec.load(filepath)
