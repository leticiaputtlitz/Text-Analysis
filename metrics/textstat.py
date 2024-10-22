import textstat
import pandas as pd

# Definição da classe TextStatAnalyzer
class TextStatAnalyzer:
    def __init__(self, language='en'):
        self.language = language

    def analyze_text(self, text):
        # Verifica se o texto está vazio ou None
        if not text:
            return {
                "Flesch Reading Ease": None,
                "SMOG Index": None,
                "Flesch-Kincaid Grade Level": None,
                "Coleman-Liau Index": None,
                "Automated Readability Index": None,
                "Dale-Chall Readability Score": None,
                "Difficult Words": None,
                "Linsear Write Formula": None,
                "Gunning Fog Index": None,
                "Text Standard": None,
                "Lexicon Count": None,
                "Sentence Count": None,
                "Syllable Count": None,
                "Character Count": None,
                "Polysyllable Count": None,
                "Monosyllable Count": None
            }

        # Aplica as funções de análise de textstat
        results = {
            "Flesch Reading Ease": textstat.flesch_reading_ease(text),
            "SMOG Index": textstat.smog_index(text),
            "Flesch-Kincaid Grade Level": textstat.flesch_kincaid_grade(text),
            "Coleman-Liau Index": textstat.coleman_liau_index(text),
            "Automated Readability Index": textstat.automated_readability_index(text),
            "Dale-Chall Readability Score": textstat.dale_chall_readability_score(text),
            "Difficult Words": textstat.difficult_words(text),
            "Linsear Write Formula": textstat.linsear_write_formula(text),
            "Gunning Fog Index": textstat.gunning_fog(text),
            "Text Standard": textstat.text_standard(text),
            "Lexicon Count": textstat.lexicon_count(text),
            "Sentence Count": textstat.sentence_count(text),
            "Syllable Count": textstat.syllable_count(text),
            "Character Count": textstat.char_count(text),
            "Polysyllable Count": textstat.polysyllabcount(text),
            "Monosyllable Count": textstat.monosyllabcount(text)
        }
        return results

    def analyze(self, data):
        # Aplica a função analyze_text para cada texto na coluna 'prompt' e adiciona novas colunas
        prompt_stats = data['prompt'].apply(lambda x: pd.Series(self.analyze_text(x)))
        prompt_stats.columns = ["Prompt " + col for col in prompt_stats.columns]
        
        # Aplica a função analyze_text para cada texto na coluna 'response' e adiciona novas colunas
        response_stats = data['response'].apply(lambda x: pd.Series(self.analyze_text(x)))
        response_stats.columns = ["Response " + col for col in response_stats.columns]
        
        # Concatena as novas colunas ao DataFrame original
        analyzed_data = pd.concat([data, prompt_stats, response_stats], axis=1)
        
        return analyzed_data
