from flask import request, jsonify

from app.use_cases.review_analysis.named_entity_recognition import NamedEntityRecognition
from app.use_cases.review_analysis.sentiment_analysis import SentimentAnalysis
from app.use_cases.review_analysis.text_summarization import TextSummarization
from app.use_cases.text_representation.one_hot_encoding import OneHotEncoding
from app.use_cases.text_representation.word2vec import Word2VecModel
from app.use_cases.text_representation.glove_embedding import GloVeEmbedding
from app.use_cases.text_representation.tfidf_representation import TfidfRepresentation


class NLPPipelineController:
    def __init__(self):
        self.one_hot_encoding = OneHotEncoding()
        self.word2vec_model = Word2VecModel()
        self.glove_embedding = GloVeEmbedding()
        self.tfidf_representation = TfidfRepresentation()
        self.ner = NamedEntityRecognition()
        self.sentiment_analysis = SentimentAnalysis()
        self.text_summarization = TextSummarization()

    def process_text(self):
        data = request.json
        text = data.get('text')
        method = data.get('method', 'tfidf')  # Default to TF-IDF if no method is provided

        # Tokenize text
        sentences = [sentence.strip() for sentence in text.split('.') if sentence]

        if method == 'one_hot':
            words = text.split()
            one_hot_encoded = self.one_hot_encoding.fit_transform(words)
            return jsonify({"one_hot": one_hot_encoded.tolist()})

        elif method == 'word2vec':
            self.word2vec_model.train(sentences)  # Train the model on input
            vectors = {word: self.word2vec_model.get_word_vector(word) for word in set(text.split())}
            return jsonify({"word2vec": vectors})

        elif method == 'glove':
            vectors = {word: self.glove_embedding.get_word_vector(word) for word in set(text.split())}
            return jsonify({"glove": vectors})

        elif method == 'tfidf':
            tfidf_matrix, feature_names = self.tfidf_representation.fit_transform(sentences)
            return jsonify({
                "tfidf_vectors": tfidf_matrix.todense().tolist(),
                "feature_names": feature_names.tolist()
            })

        else:
            return jsonify({"error": "Invalid method"}), 400

    def apply_ner(self):
        text = request.json.get('text')
        entities = self.ner.extract_entities(text)
        return jsonify({"entities": entities})

    def analyze_sentiment(self):
        text = request.json.get('text')
        sentiment = self.sentiment_analysis.analyze_sentiment(text)
        return jsonify(sentiment)

    def summarize_text(self):
        text = request.json.get('text')  # Get the input text from the request
        max_length = request.json.get('max_length', 130)  # Optional max_length parameter
        min_length = request.json.get('min_length', 30)  # Optional min_length parameter

        # Summarize the text
        summary = self.text_summarization.summarize_text(text, max_length=max_length, min_length=min_length)

        return jsonify({"summary": summary})