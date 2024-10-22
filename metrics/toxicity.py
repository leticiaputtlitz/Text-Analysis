from langkit import toxicity

# Defina a classe ToxicityAnalyzer
class ToxicityAnalyzer:
    def __init__(self):
        pass

    def analyze(self, data):
        data['prompt_toxicity'] = data['prompt'].apply(toxicity.toxicity)
        data['response_toxicity'] = data['response'].apply(toxicity.toxicity)
        return data