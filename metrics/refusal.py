from langkit import themes 

# Defina a classe RefusalAnalyzer
class RefusalAnalyzer:
    def __init__(self):
        pass

    def analyze(self, data):
        data['refusal'] = data['response'].apply(lambda x: themes.group_similarity(x, 'refusal'))
        return data

