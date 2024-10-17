from flask import request, jsonify

from app.use_cases.text_representation.glove_embedding import GloVeEmbedding
from app.use_cases.text_representation.one_hot_encoding import OneHotEncoding
from app.use_cases.text_representation.tfidf_representation import TfidfRepresentation
from app.use_cases.text_representation.word2vec import Word2VecModel


class TextRepresentationController:
    def __init__(self):
        self.one_hot_encoding = OneHotEncoding()
        self.word2vec_model = Word2VecModel()
        self.glove_embedding = GloVeEmbedding()
        self.tfidf_representation = TfidfRepresentation()

    def one_hot_encode(self):
        text = request.json['text']
        words = text.split()
        one_hot= self.one_hot_encoding.fit_transform(words)
        return jsonify({"one_hot": one_hot.tolist()})

    def train_word2vec(self):
        text = request.json['text']
        # Tokenize the input text into sentences and words
        sentences = [sentence.split() for sentence in text.split('.')]
        # Train Word2Vec model
        self.word2vec_model.train(sentences)
        return jsonify({"message": "Word2Vec model trained successfully"})

    def get_word_vector(self):
        word = request.json['word']
        vector = self.word2vec_model.get_word_vector(word)
        if vector:
            return jsonify({"word": word, "vector": vector})
        else:
            return jsonify({"error": f"Word '{word}' not found in vocabulary"}), 404

    def get_glove_vector(self):
        word = request.json.get('word')
        vector = self.glove_embedding.get_word_vector(word)
        if vector is not None:
            return jsonify({"word": word, "vector": vector.tolist()})
        else:
            return jsonify({"error": f"Word '{word}' not found in GloVe vocabulary"}), 404

    def get_tfidf_representation(self):
        text = request.json.get('text')  # Extract the text from the request
        # Split the text into sentences/documents
        documents = text.split('.')
        tfidf_matrix, feature_names = self.tfidf_representation.fit_transform(documents)

        # Convert the TF-IDF matrix to a dense format and convert to a list
        tfidf_dense = tfidf_matrix.todense().tolist()

        # Return the TF-IDF vectors with feature names
        return jsonify({
            "tfidf_vectors": tfidf_dense,
            "feature_names": feature_names.tolist()
        })