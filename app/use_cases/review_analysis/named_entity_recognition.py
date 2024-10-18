import spacy
import torch

class NamedEntityRecognition:
    def __init__(self):
        # Check if GPU is available and load the appropriate model
        try:
            if torch.cuda.is_available():
                self.nlp = spacy.load('en_core_web_trf')  # Transformer-based model for GPU
            else:
                self.nlp = spacy.load('en_core_web_sm')   # CPU-based small model
        except OSError:
            self.nlp = spacy.load('en_core_web_sm')       # Fallback to small model

    def extract_entities(self, text):
        """
        Extract named entities from the text.
        :param text: Input text (string)
        :return: List of named entities and their labels
        """
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]  # Extract entities and their labels
        return entities