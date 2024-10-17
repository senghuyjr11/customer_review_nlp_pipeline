from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfRepresentation:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit_transform(self, documents):
        """
        Fit the TF-IDF vectorizer and transform the documents into TF-IDF features.
        - documents: A list of strings (documents).
        """
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        return tfidf_matrix, self.vectorizer.get_feature_names_out()

    def transform(self, documents):
        """
        Transform new documents based on the fitted TF-IDF vectorizer.
        - documents: A list of strings (documents).
        """
        return self.vectorizer.transform(documents)