from textblob import TextBlob


class SentimentAnalysis:
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