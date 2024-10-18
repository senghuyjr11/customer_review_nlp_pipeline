from fastapi.encoders import jsonable_encoder

from app.models import SummarizeInput, TextInput
from app.use_cases.review_analysis.named_entity_recognition import NamedEntityRecognition
from app.use_cases.review_analysis.sentiment_analysis import SentimentAnalysis
from app.use_cases.review_analysis.text_summarization import TextSummarization
from app.use_cases.text_representation.glove_embedding import GloVeEmbedding
from app.use_cases.text_representation.one_hot_encoding import OneHotEncoding
from app.use_cases.text_representation.tfidf_representation import TfidfRepresentation
from app.use_cases.text_representation.word2vec import Word2VecModel


class NLPPipelineController:
    def __init__(self):
        self.one_hot_encoding = OneHotEncoding()
        self.word2vec_model = Word2VecModel()
        self.glove_embedding = GloVeEmbedding()
        self.tfidf_representation = TfidfRepresentation()
        self.ner = NamedEntityRecognition()
        self.sentiment_analysis = SentimentAnalysis()
        self.text_summarization = TextSummarization()

    def process_text(self, input_text: TextInput, method: str = 'tfidf'):
        text = input_text.text
        sentences = [sentence.strip() for sentence in text.split('.') if sentence]

        if method == 'one_hot':
            words = text.split()
            one_hot_encoded = self.one_hot_encoding.fit_transform(words)
            return {"one_hot": one_hot_encoded.tolist()}

        elif method == 'word2vec':
            sentences = [sentence.split() for sentence in input_text.text.split('.')]
            self.word2vec_model.train(sentences)  # Ensure the model is trained here
            vector = self.word2vec_model.get_word_vector(input_text.text)
            if vector:
                return {"word": input_text.text, "vector": vector}
            else:
                return {"error": f"Word '{input_text.text}' not found in vocabulary"}, 404


        elif method == 'glove':
            word = input_text.text  # Get the word
            vector = self.glove_embedding.get_word_vector(word)

            if vector is not None:
                # Ensure the vector is converted to a JSON-serializable format (e.g., list)
                return jsonable_encoder(
                    {"word": word, "vector": vector.tolist() if hasattr(vector, 'tolist') else vector})
            else:
                return {"error": f"Word '{word}' not found in GloVe vocabulary"}, 404

        elif method == 'tfidf':
            tfidf_matrix, feature_names = self.tfidf_representation.fit_transform(sentences)
            return {
                "tfidf_vectors": tfidf_matrix.todense().tolist(),
                "feature_names": feature_names.tolist()
            }

        else:
            return {"error": "Invalid method"}, 400

    def apply_ner(self, input_text: TextInput):
        text = input_text.text
        entities = self.ner.extract_entities(text)
        return {"entities": entities}

    def analyze_sentiment(self, input_text: TextInput):
        text = input_text.text
        sentiment = self.sentiment_analysis.analyze_sentiment(text)
        return sentiment

    def summarize_text(self, input_text: SummarizeInput):
        text = input_text.text
        max_length = input_text.max_length
        min_length = input_text.min_length

        summary = self.text_summarization.summarize_text(text, max_length=max_length, min_length=min_length)
        return {"summary": summary}
