from fastapi.encoders import jsonable_encoder

from app.models import TextInput
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

    def one_hot_encode(self, input_text: TextInput):
        words = input_text.text.split()
        one_hot = self.one_hot_encoding.fit_transform(words)
        return {"one_hot": one_hot.tolist()}

    def train_word2vec(self, input_text: TextInput):
        sentences = [sentence.split() for sentence in input_text.text.split('.')]
        self.word2vec_model.train(sentences)
        return {"message": "Word2Vec model trained successfully"}

    def get_word_vector(self, input_text: TextInput):
        vector = self.word2vec_model.get_word_vector(input_text.text)
        if vector:
            return {"word": input_text.text, "vector": vector}
        else:
            return {"error": f"Word '{input_text.text}' not found in vocabulary"}, 404

    def get_glove_vector(self, input_text: TextInput):
        # Assuming you are passing the full text and need to extract words
        word = input_text.text  # This gets the word or text from the Pydantic model
        vector = self.glove_embedding.get_word_vector(word)
        if vector is not None:
            # Convert to JSON-compatible format using jsonable_encoder
            return jsonable_encoder({"word": word, "vector": vector.tolist()})
        else:
            return {"error": f"Word '{word}' not found in GloVe vocabulary"}, 404

    def get_tfidf_representation(self, input_text: TextInput):
        documents = input_text.text.split('.')
        tfidf_matrix, feature_names = self.tfidf_representation.fit_transform(documents)
        tfidf_dense = tfidf_matrix.todense().tolist()
        return {
            "tfidf_vectors": tfidf_dense,
            "feature_names": feature_names.tolist()
        }
