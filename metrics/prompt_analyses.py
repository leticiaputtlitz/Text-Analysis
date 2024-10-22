from langkit import themes 
from langkit import injections 

# Defina a classe PromptAnalyzer
class PromptAnalyzer:
    def __init__(self):
        pass

    def analyze(self, data):
        # Cria colunas para armazenar os resultados das análises
        data['prompt_injection'] = data['prompt'].apply(lambda x: injections.injection({'prompt': [x]}))
        data['prompt_jailbreak'] = data['prompt'].apply(lambda x: themes.group_similarity(x, 'jailbreak'))
        return data
