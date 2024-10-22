from langkit import topics

# Defina a classe TopicsAnalyzer
class TopicsAnalyzer:
    def __init__(self):
        pass

    def analyze(self, data, topics_list=None):
        if topics_list is not None:
            topics.init(topics=topics_list)
        data['prompt_topics'] = data['prompt'].apply(lambda x: topics.closest_topic(x))
        data['response_topics'] = data['response'].apply(lambda x: topics.closest_topic(x))
        return data