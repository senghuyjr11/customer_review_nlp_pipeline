from flask import Flask

from app.adapters.controllers.nlp_pipeline_controller import NLPPipelineController
from app.adapters.controllers.text_representation_controller import TextRepresentationController

app = Flask(__name__)
text_representation_controller = TextRepresentationController()
nlp_pipeline_controller = NLPPipelineController()

@app.route("/one_hot", methods=["POST"])
def one_hot_encode():
    return text_representation_controller.one_hot_encode()

@app.route('/train_word2vec', methods=['POST'])
def train_word2vec():
    return text_representation_controller.train_word2vec()

@app.route('/get_word_vector', methods=['POST'])
def get_word_vector():
    return text_representation_controller.get_word_vector()

@app.route('/get_glove_vector', methods=['POST'])
def get_glove_vector():
    return text_representation_controller.get_glove_vector()

@app.route('/get_tfidf_vector', methods=['POST'])
def get_tfidf_vector():
    return text_representation_controller.get_tfidf_representation()

# Route for processing text with selected method (One-Hot, Word2Vec, GloVe, TF-IDF)
@app.route('/process_text', methods=['POST'])
def process_text():
    """
    Usage example:
    {
      "text": "AI",
      "method": "tfidf"
    }
    """
    return nlp_pipeline_controller.process_text()

# Route for Named Entity Recognition (NER)
@app.route('/ner', methods=['POST'])
def apply_ner():
    return nlp_pipeline_controller.apply_ner()

# Route for Sentiment Analysis
@app.route('/sentiment', methods=['POST'])
def analyze_sentiment():
    return nlp_pipeline_controller.analyze_sentiment()

@app.route('/summarize', methods=['POST'])
def summarize():
    return nlp_pipeline_controller.summarize_text()