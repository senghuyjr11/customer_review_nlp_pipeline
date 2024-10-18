from textblob import TextBlob
import torch
from transformers import pipeline


class SentimentAnalysis:
    def __init__(self):
        # Check if a GPU is available, and if so, set device=0 (for the first GPU)
        device = 0 if torch.cuda.is_available() else -1

        # Load a pre-trained sentiment analysis model on the GPU if available
        self.sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased", device=device)

    def analyze_sentiment(self, text):
        """
        Analyze the sentiment of the input text.
        :param text: Input text (string)
        :return: Dictionary containing polarity, subjectivity, and sentiment label (positive/neutral/negative)
        """
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        # Determine sentiment label based on polarity
        if polarity > 0:
            sentiment = "positive"
        elif polarity < 0:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return {
            "polarity": polarity,
            "subjectivity": subjectivity,
            "sentiment": sentiment
        }