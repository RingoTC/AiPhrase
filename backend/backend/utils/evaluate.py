from ml_models.words_evaluation import sentenceScore as words_score
from ml_models.sentence_check import sentence_correction
from ml_models.FleschReadingEaseScore import fresScore
from backend.models import Sentence, GoodAnswer, RecordDetail
from ml_models.similarity import inferencePairsFromGraph
from ml_models.sentenceComplexity import sentenceComplex
import json
from ml_models.mying_similarity import my_similarity_serving
#from ml_models.ltt_similarity import bimpmPred as ltt_similarity_serving

# 语义相似性，句法复杂性，语法准确性，词汇常见性，句子易读性
id_category = {
    1: '语义相似性',
    2: '词汇常见性',
    3: '句子易读性',
    4: '句法复杂性',
    5: '语法准确性'
}
id_description = {
    1: ['语义不相似', '语义不相似', '语义不相似', '语义基本相似', '语义相似', '语义非常相似'],
    2: ['句子所用词汇很生僻', '句子所用词汇生僻', '句子所用词汇生僻', '句子所用词汇常见', '句子所用词汇常见', '句子所用词汇很常见'],
    3: ['语句复杂难懂，适合研究生阅读', '语句复杂难懂，适合本科生阅读', '语句简明，语义易懂', '语句简明，语义易懂', '语句自然流畅，语义比较容易理解', '语句简洁流畅，语义非常容易理解'],
    4: ['相比原句，句子成分间的依存关系很简单', '相比原句，句子成分间的依存关系很简单', '相比原句，句子成分间的依存关系简单', '相比原句，句子成分间的依存关系简单', '相比原句，句子成分间的依存关系简单',
        '相比原句，句子成分间的依存关系复杂'],
    5: ['有大量语法、拼写等错误', '语法、拼写等错误较多', '有少量语法、拼写等错误', '有少量语法、拼写等错误', '语法、拼写等错误极少', '没有语法、拼写等错误']
}


def evaluate_sentence_wordscore(sentence):
    '''
     判断句子中单词的生僻性，
     数值越小，越生僻，
     返回一个数值，
     score < 0.2，句子所用词汇很生僻；
     0.2 =< score < 0.4，句子所用词汇比较生僻；
     0.4 =< score < 0.6，句子所用词汇常见；
     0.6 =< score < 0.7，句子所用词汇比较常见；
     score >= 0.7，句子所用词汇很常见；
     输入：句子，
     输出：分数
     '''
    # top 5000 English words were downloaded from
    # https://www.oxfordlearnersdictionaries.com/wordlists/oxford3000-5000
    sentence = sentence.lower()
    wordscore = words_score(sentence)

    def get_label(num):
        bins = [-100, 0.1, 0.2, 0.4, 0.6, 0.7, 100]
        for i in range(0, 6):
            if bins[i] <= num < bins[i + 1]:
                return i

    wordscore = get_label(wordscore)
    wordscore_detail = {'id': 2, 'value': str(wordscore), 'name': id_category[2],
                        'description': id_description[2][wordscore]}
    return wordscore, wordscore_detail


def evaluate_readbility(sentence):
    readable_score = fresScore(sentence)

    def get_label(num):
        bins = [-1e4, 30, 50, 60, 70, 80, 1e4]
        for i in range(0, 6):
            if bins[i] <= num < bins[i + 1]:
                return i

    readable_score = get_label(readable_score)

    readable_detail = {'id': 3, 'value': str(readable_score), 'name': id_category[3],
                       'description': id_description[3][readable_score]}
    return readable_score, readable_detail


import numpy as np
import math


def model_ensemble_predict(x, Beta):
    x = np.append(x, [1])
    res = 1 / (1 + math.exp(-x.dot(Beta)))
    return res


def evaluate_similarity(sentence, customer_answer):
    '''
    句子相似程度
    :param problem_id:
    :param customer_answer:
    :return:
    '''

    bert_similarity = inferencePairsFromGraph(customer_answer, sentence)
    my_similarity = my_similarity_serving(customer_answer, sentence)
    # ltt_similarity = ltt_similarity_serving(customer_answer, sentence)
    # # 1.5497187   0.63752652  2.23683518 -2.53611195
    # # ltt my cuiz bias
    # X = np.array([ltt_similarity, my_similarity, bert_similarity])
    # Beta = np.array([1.5497187, 0.63752652, 2.23683518, -2.53611195])
    # similarity_score_float = model_ensemble_predict(X, Beta)
    similarity_score_float = bert_similarity + my_similarity
   # similarity_score_float = ltt_similarity * 1.5497187 + my_similarity * 0.63752652 + bert_similarity * 2.23683518 - 2.53611195

    def get_label(num):
        bins = [-100, 0.2, 0.4, 0.5, 0.6, 0.8, 100]
        for i in range(0, 6):
            if bins[i] <= num < bins[i + 1]:
                return i

    similarity_score = get_label(similarity_score_float)

    similarity_detail = {'id': 1, 'value': str(similarity_score), 'name': id_category[1],
                         'description': id_description[1][similarity_score]}

    return similarity_score, similarity_detail, similarity_score * 2 / 10


# 从各个指标评价句子
def evaluate_sentence_total(sentence, customer_answer):
    wordscore, wordscore_detail = evaluate_sentence_wordscore(customer_answer)
    similarity_score, similarity_detail, similarity_score_float = evaluate_similarity(sentence, customer_answer)
    readable_score, readable_detail = evaluate_readbility(customer_answer)
    complex_score, complex_detail = evaluate_sentence_complexity(sentence, customer_answer)
    correction_score, correction_detail = sentence_grammer_score(customer_answer)
    # 计算总分
    total_score = ceil(
        wordscore * 1.5 + similarity_score_float * 70 + readable_score * 1.5 + complex_score + correction_score * 2)
    return int(total_score), [wordscore_detail, similarity_detail, readable_detail, complex_detail, correction_detail]


# 更新好答案
def updateGoodAnswer(sentence_id, record):
    record_set = GoodAnswer.objects.filter(record_id__problem_id=sentence_id).order_by('record_id__score')
    isExc = False
    if len(record_set) < 3:
        GoodAnswer.objects.create(record_id=record).save()
        isExc = True
    else:
        min_score_good_record = record_set[0]
        if min_score_good_record.record_id.score < record.score:
            isExc = True
            GoodAnswer.objects.create(record_id=record).save()
            min_score_good_record.delete()
    return isExc


def evaluate_sentence_complexity(problem_sentence, customer_sentence):
    complex_score = sentenceComplex(problem_sentence, customer_sentence)

    def get_label(num):
        bins = [-100, 0.2, 0.4, 0.6, 0.8, 0.999999, 100]
        for i in range(0, 6):
            if bins[i] <= num < bins[i + 1]:
                return i

    complex_score = get_label(complex_score)
    print('complex_score:', complex_score)
    complex_detail = {'id': 4, 'value': str(complex_score), 'name': id_category[4],
                      'description': id_description[4][complex_score]}
    return complex_score, complex_detail


def save_details(record_instance, details):
    detail_instances = []
    for detail in details:
        value = float(detail['value'])
        category_id = detail['id']
        if category_id == 5:
            detail_instances.append(RecordDetail(category_id=category_id, value=value, problem_record=record_instance,
                                                 info=json.dumps(detail['erros'])))
        else:
            detail_instances.append(RecordDetail(category_id=category_id, value=value, problem_record=record_instance))
    RecordDetail.objects.bulk_create(detail_instances)


def change_record_detail_to_dict(record_detail):
    #  readable_detail = {'id': 3, 'value': str(readable_score), 'name': id_category[3], 'description': id_description[3]}
    id = record_detail.category_id
    detail = {'id': id, 'value': str(int(record_detail.value)), 'name': id_category[id],
              'description': id_description[id][int(record_detail.value)]}
    if id == 5:
        try:
            detail['erros'] = json.loads(record_detail.info)
        except Exception as e:
            detail['erros'] = []
            # raise e
    return detail


from numpy import ceil


def sentence_grammer_score(sentence):
    correction_score, errors = sentence_correction(sentence)
    correction_score = 5 - correction_score
    if correction_score <= 0:
        correction_score = 0

    correction_detail = {'id': 5, 'value': str(correction_score), 'name': id_category[5],
                         'description': id_description[5][correction_score], 'erros': errors}
    return correction_score, correction_detail


def get_errors(record_id):
    try:
        detail_correct = RecordDetail.objects.filter(problem_record=record_id, category_id=5)
        error = detail_correct[0]['info']
        error = json.loads(error)
    except:
        error = []
    return error
