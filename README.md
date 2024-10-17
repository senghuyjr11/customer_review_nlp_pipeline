# Customer Review NLP Pipeline

This project implements a comprehensive NLP pipeline that processes customer reviews with various text representation methods, sentiment analysis, named entity recognition (NER), and text summarization. The pipeline adheres to **Clean Architecture** principles for maintainability and scalability.

## Features

- **One-Hot Encoding**: Represents each word as a binary vector.
- **Word2Vec**: Generates word vectors using `gensim`.
- **GloVe**: Uses pre-trained GloVe embeddings.
- **TF-IDF**: Extracts text features using Term Frequency-Inverse Document Frequency.
- **NER (Named Entity Recognition)**: Extracts entities like people, organizations, and locations using `spaCy`.
- **Sentiment Analysis**: Determines text polarity and subjectivity using `TextBlob`.
- **Summarization**: Generates summaries using Hugging Face’s `BART` model.

## Clean Architecture

This project follows **Clean Architecture**:
- **Entities (Business Logic)**: Core NLP functions (e.g., text representation, sentiment analysis).
- **Use Cases**: Handlers for applying different NLP techniques.
- **Controllers**: Interfaces between the API and the use cases.
- **Frameworks**: Flask-based API for exposing endpoints.

## Installation

### Prerequisites:
- **Python 3.9+**
- **Pipenv** or **virtualenv** for environment management

### Steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/customer_review_nlp_pipeline.git
    cd customer_review_nlp_pipeline
    ```

2. **Set up a virtual environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use .venv\Scripts\activate
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Download the required NLP models**:
    ```bash
    python -m spacy download en_core_web_sm
    ```

5. **Download Glove**:
    ```bash
    https://nlp.stanford.edu/projects/glove/
   # extract the file and paste file .txt
    ```

6. **Run the application**:
    ```bash
    python run.py
    ```

## API Endpoints

1. **`/process_text`** (POST): Apply text representation (One-Hot, Word2Vec, GloVe, TF-IDF).
    - **Request**:
      ```json
      {
        "text": "Input text here",
        "method": "one_hot or word2vec, glove, tfidf"
      }
      ```

2. **`/ner`** (POST): Extract named entities (NER) from the input text.
    - **Request**:
      ```json
      {
        "text": "Apple is buying a startup in the U.K."
      }
      ```

3. **`/sentiment`** (POST): Perform sentiment analysis on the input text.
    - **Request**:
      ```json
      {
        "text": "I love this product!"
      }
      ```

4. **`/summarize`** (POST): Summarize the input text using BART.
    - **Request**:
      ```json
      {
        "text": "Artificial Intelligence is transforming industries..."
      }
      ```

## Usage Examples

You can interact with the API via `curl` or Postman. Here are examples using `curl`:

- **One-Hot Encoding**:
    ```bash
    curl -X POST http://127.0.0.1:5000/process_text \
         -H "Content-Type: application/json" \
         -d '{"text": "AI is amazing.", "method": "one_hot"}'
    ```

- **Named Entity Recognition (NER)**:
    ```bash
    curl -X POST http://127.0.0.1:5000/ner \
         -H "Content-Type: application/json" \
         -d '{"text": "Apple is buying a startup in the U.K."}'
    ```

- **Sentiment Analysis**:
    ```bash
    curl -X POST http://127.0.0.1:5000/sentiment \
         -H "Content-Type: application/json" \
         -d '{"text": "I love this product!"}'
    ```

- **Text Summarization**:
    ```bash
    curl -X POST http://127.0.0.1:5000/summarize \
         -H "Content-Type: application/json" \
         -d '{"text": "Artificial Intelligence is transforming industries..."}'
    ```

## Project Structure

```plaintext
customer_review_nlp_pipeline/
│
├── app/
│   ├── adapters/
│   │   └── controllers/
│   │       ├── nlp_pipeline_controller.py          # Main controller for NLP Review Pipeline
│   │       └── text_representation_controller.py   # Main controller for NLP Text Representation
│   ├── interfaces/
│   │   └── api.py                                  # API routes
│   ├── use_cases/
│   │   ├── text_representation/
│   │   │   ├── one_hot_encoding.py
│   │   │   ├── word2vec.py
│   │   │   ├── glove_embedding.py
│   │   │   └── tfidf_representation.py
│   │   └── review_analysis/
│   │       ├── named_entity_recognition.py
│   │       ├── sentiment_analysis.py
│   │       └── text_summarization.py
│
├── run.py                                          # Main entry point for running the app
├── requirements.txt                                # Python dependencies
└── README.md                                       # This file