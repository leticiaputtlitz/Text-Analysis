from presidio_analyzer import AnalyzerEngine

# Inicialize o motor de análise
analyzer = AnalyzerEngine()

# Definição da classe PIIAnalyzer
class PIIAnalyzer:
    def __init__(self):
        self.analyzer = AnalyzerEngine()

    def analyze_pii(self, text, entities=None):
        # Se nenhuma entidade for especificada ou a lista de entidades estiver vazia, analisar todas as entidades
        if not entities:
            entities = None
        
        results = self.analyzer.analyze(text=text, entities=entities, language='en')
        entities_info = []

        for result in results:
            entity_info = {
                "Entidade": result.entity_type,
                "Texto": text[result.start:result.end],
                "Confiança": result.score
            }
            entities_info.append(entity_info)
        
        return entities_info

    def analyze(self, data, selected_entities=None):
        data['prompt_pii'] = data['prompt'].apply(lambda x: self.analyze_pii(x, entities=selected_entities))
        data['response_pii'] = data['response'].apply(lambda x: self.analyze_pii(x, entities=selected_entities))
        return data
