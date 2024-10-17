import spacy


class NamedEntityRecognition:
    def __init__(self):
        # Load the English NLP model
        self.nlp = spacy.load('en_core_web_sm')

    def extract_entities(self, text):
        """
        Extract named entities from the text.
        :param text: Input text (string)
        :return: List of named entities and their labels
        """
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]  # Extract entities and their labels
        return entities