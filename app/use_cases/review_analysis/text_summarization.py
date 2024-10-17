from transformers import pipeline

class TextSummarization:
    def __init__(self):
        # Load a pre-trained summarization model (BART)
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    def summarize_text(self, text, max_length=130, min_length=30, do_sample=False):
        """
        Summarize the input text using a pre-trained model.
        :param text: Input text (string)
        :param max_length: The maximum length of the summary.
        :param min_length: The minimum length of the summary.
        :param do_sample: Whether to sample from the output.
        :return: Summary of the text
        """
        summary = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=do_sample)
        return summary[0]['summary_text']
