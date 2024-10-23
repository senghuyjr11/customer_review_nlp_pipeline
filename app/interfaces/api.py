from fastapi import FastAPI
from app.adapters.controllers.nlp_pipeline_controller import NLPPipelineController
from app.adapters.controllers.text_representation_controller import TextRepresentationController
from app.models import SummarizeInput, TextInput

app = FastAPI()
text_representation_controller = TextRepresentationController()
nlp_pipeline_controller = NLPPipelineController()

@app.post("/one_hot")
async def one_hot_encode(input_text: TextInput):
    return text_representation_controller.one_hot_encode(input_text)

@app.post("/train_word2vec")
async def train_word2vec(input_text: TextInput):
    return text_representation_controller.train_word2vec(input_text)

@app.post("/get_word_vector")
async def get_word_vector(input_word: TextInput):
    return text_representation_controller.get_word_vector(input_word)

@app.post("/get_glove_vector")
async def get_glove_vector(input_text: TextInput):
    return text_representation_controller.get_glove_vector(input_text)

@app.post("/get_tfidf_representation")
async def get_tfidf_representation(input_text: TextInput):
    return text_representation_controller.get_tfidf_representation(input_text)


@app.post("/process_text/{method}")
async def process_text(input_text: TextInput, method: str):
    return nlp_pipeline_controller.process_text(input_text, method)

@app.post("/ner")
async def apply_ner(input_text: TextInput):
    return nlp_pipeline_controller.apply_ner(input_text)

@app.post("/sentiment")
async def analyze_sentiment(input_text: TextInput):
    return nlp_pipeline_controller.analyze_sentiment(input_text)

@app.post("/summarize")
async def summarize(input_text: SummarizeInput):
    return nlp_pipeline_controller.summarize_text(input_text)
