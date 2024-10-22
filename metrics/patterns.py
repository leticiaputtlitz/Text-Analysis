from langkit import regexes

# Defina a classe RegexAnalyzer
class RegexAnalyzer:
    def __init__(self, pattern_file_path="pattern_groups.json"):
        regexes.init(pattern_file_path=pattern_file_path)

    def analyze(self, data):
        data['prompt_patterns'] = data['prompt'].apply(lambda x: regexes.has_patterns(x))
        data['response_patterns'] = data['response'].apply(lambda x: regexes.has_patterns(x))
        return data
