import pandas as pd
import os, django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "AiEditor.settings")  # project_name 项目名称
django.setup()
from backend.models import Sentence


def load_csv():
    df = pd.read_csv(r'data/sentence_pairs_web.csv')[['sen1', 'sen2', 'type']]
    sentences = []
    for row in df.values:
        question1, question2, question_type = row
        obj = Sentence(sentence=question1, similar_sentence=question2, sentence_type=question_type)
        sentences.append(obj)
    Sentence.objects.bulk_create(sentences)


if __name__ == '__main__':
    load_csv()
