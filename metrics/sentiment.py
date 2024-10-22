from langkit import sentiment

# Defina a classe SentimentAnalyzer
class SentimentAnalyzer:
    def __init__(self):
        pass

    def analyze(self, data):
        # Aplica a análise de sentimento nas colunas 'prompt' e 'response'
        data['prompt_sentiment'] = data['prompt'].apply(sentiment.sentiment_nltk)
        data['response_sentiment'] = data['response'].apply(sentiment.sentiment_nltk)
        return data

