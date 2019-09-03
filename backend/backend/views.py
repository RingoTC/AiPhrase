from django.shortcuts import render

# Create your views here.
from django.core import serializers
from backend.utils.evaluate import evaluate_sentence_total, updateGoodAnswer, save_details, \
    change_record_detail_to_dict, sentence_correction, sentenceComplex, get_errors
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.template import loader
from .models import User, Sentence, ProblemRecord, GoodAnswer, RecordDetail
import random
from ml_models.similarity import inferencePairsFromGraph
from backend.utils.stastics import get_stastics, get_stastics_by_list
import json
from ml_models.synonym import get_syn_words
from ml_models.paraphraser_full import paraphraser_full
from numpy import ceil
from backend.utils.user_level import get_user_level



# 同义改写
@csrf_exempt
def get_paraphrase(request):
    assert (request.method == 'POST')
    info = json.loads(request.body)
    print(info)
    sentence = info['sentence']
    sentences = paraphraser_full(sentence)
    return JsonResponse({'paraphrase': sentences})


def backend_view(request):
    return JsonResponse({'info': "Hello, You're at the backend view."})


# 获取用户信息
def get_user_info(request, tel_number):
    '''
    功能1：获取用户信息

    输入：phoneNumber
    输出：(是否注册：signed, 用户id：userID, 手机号码：phoneNumber)
    {
        phoneNumber
    }
    {
        signed: true,
        userID: 1,
        phoneNumber: 17162713333
    }
    :param request:
    :param user_id:
    :return:
    '''
    query_set = User.objects.filter(tel_number=tel_number)
    if len(query_set) == 1:
        user = query_set[0]
        response_dict = {'signed': True, 'userID': user.id, 'phoneNumber': user.tel_number}
    elif len(query_set) == 0:
        new_user = User.objects.create(tel_number=tel_number)
        response_dict = {'signed': True, 'userID': new_user.id, 'phoneNumber': new_user.tel_number}
        print('id is :', new_user.id)
        new_user.save()
        print('id is :', new_user.id)
    else:
        raise Exception('Error when query user info.')
    return JsonResponse(response_dict)


# 随机获取句子
def get_sentence_for_user(request, problem_type, user_id):
    '''
    功能2：获取题目
输入：typeOfQue：长中短
输出：
题目id：queID
{
    typeOfQue:2//长句联系
}
{
    typeOfQue:2,
    question:"Paraphrasing is extremely important, so it should be learned!",
    queID:0//题目id
}
功能2获取题目api: [
api: http://52.80.106.20:8000/backend/problem/length(012分别代表短中长)
]
    :param request:
    :return:
    '''

    query_set = Sentence.objects.filter(sentence_type=problem_type)

    if query_set:
        sentence = random.choice(query_set)
        experienced = ProblemRecord.objects.filter(problem_id=sentence.id, user_id=user_id).exists()
        response_dict = {'problem_type': problem_type, 'sentence': sentence.sentence, 'problem_id': sentence.id,
                         'experienced': experienced}
    else:
        response_dict = {'error': 'no appropriate sentence'}

    return JsonResponse(response_dict)


# 根据id获取句子
def get_sentence_by_id(request, problem_index, user_id, problem_type):
    '''
    功能2：获取题目
输入：typeOfQue：长中短
输出：
题目id：queID
{
    typeOfQue:2//长句联系
}
{
    typeOfQue:2,
    question:"Paraphrasing is extremely important, so it should be learned!",
    queID:0//题目id
}
功能2获取题目api: [
api: http://52.80.106.20:8000/backend/problem/length(012分别代表短中长)
]
    :param request:
    :return:
    '''

    query_set = Sentence.objects.filter(sentence_type=problem_type).order_by('id')

    if len(query_set) >= problem_index:
        sentence = query_set[problem_index]
        experienced = ProblemRecord.objects.filter(problem_id=sentence.id, user_id=user_id).exists()
        response_dict = {'problem_type': sentence.sentence_type, 'sentence': sentence.sentence,
                         'problem_id': sentence.id,
                         'problem_index': problem_index,
                         'experienced': experienced
                         }
    else:
        response_dict = {'error': 'problem not exists'}

    return JsonResponse(response_dict)


def get_aspect_detail(id, value, name, description):
    return {'id': id, 'value': str(value), 'name': name, 'description': description}


# 评价句子/post
@csrf_exempt
def evaluate_sentence(request):
    assert (request.method == 'POST')
    info = json.loads(request.body)
    print(info)
    sentence_id, customer_answer, user_id = int(info['queID']), info['ans'], int(info['user_id'])
    # 取句子和用户
    try:
        sentence_instance = Sentence.objects.filter(id=sentence_id)[0]
        user_instance = User.objects.filter(id=user_id)[0]
    except:
        return {'error': 'no such problem or user'}
    # 评估句子
    total_score, details = evaluate_sentence_total(sentence_instance.sentence, customer_answer)
    # 保存结果
    record = ProblemRecord.objects.create(user_id=user_instance, problem_id=sentence_instance,
                                          answer=customer_answer,
                                          score=total_score)
    record.save()
    # 如果评分为最高的三个之一，则更新goodAnswer表
    isExc = updateGoodAnswer(sentence_id, record)

    # 往数据库中存储答题细节
    save_details(record, details)

    rs = {'queID': sentence_id,
          'record_id': record.id,
          'rate': str(total_score),
          'isExc': isExc,
          'detail': details,
          'customer_answer': customer_answer
          }

    return JsonResponse(rs)


def index(request):
    result = 0.99999
    template = loader.get_template('polls/index.html')
    context = {
        'latest_question_list': 1,
        'result': result,
    }
    return HttpResponse(template.render(context, request))


@csrf_exempt
def predict(request):
    x1 = request.GET.get("v1")
    x2 = request.GET.get("v2")
    print(x1)
    print(x2)
    import time
    start = time.time()
    res = inferencePairsFromGraph(x1, x2)
    end = time.time()
    data = {"result": str(res), 'time': end - start}
    return JsonResponse(data)


def get_good_answers(request, problem_id):
    rs = GoodAnswer.objects.filter(record_id__problem_id__id=problem_id).values('record_id__answer',
                                                                                'record_id__score')[:3]
    answers = [dic['record_id__answer'] for dic in rs]
    scores = [str(dic['record_id__score']) for dic in rs]
    rs_dict = {
        'queID': problem_id,
        'excAns_scores': list(zip(scores, answers))
    }
    return JsonResponse(rs_dict)


# 根据用户id和题号返回所有该用户在该题下的所有答案
def get_history_answers_by_problem(request, user_id, problem_id):
    query_set = ProblemRecord.objects.filter(user_id=user_id, problem_id=problem_id).order_by('-id')
    rs_list = []
    for record in query_set:
        details = [change_record_detail_to_dict(record_detail) for record_detail in record.recorddetail_set.all()]
        instance_rs = {'queID': record.problem_id.id,
                       'record_id': record.id,
                       'rate': str(int(record.score)),
                       'detail': details,
                       'customer_answer': record.answer
                       }
        rs_list.append(instance_rs)

    return JsonResponse({'rs': rs_list})


# 返回用户的历史回答，分页
def get_history_answers_by_page(request, user_id, page_index):
    range_start = (page_index - 1) * 6
    range_end = page_index * 6
    query_set = ProblemRecord.objects.filter(user_id=user_id).order_by('-id')
    problem_ids = [record.problem_id.id for record in query_set]
    problem_id_set = list(set(problem_ids))
    problem_id_set.sort(key=problem_ids.index)
    sentences = {}
    sentences['s'] = [a[0] for a in Sentence.objects.filter(sentence_type='s').order_by('id').values_list('id')]
    sentences['m'] = [a[0] for a in Sentence.objects.filter(sentence_type='m').order_by('id').values_list('id')]
    sentences['l'] = [a[0] for a in Sentence.objects.filter(sentence_type='l').order_by('id').values_list('id')]

    rs = []
    for pid in problem_id_set[range_start:range_end]:
        sentence = Sentence.objects.filter(id=pid)[0]
        problem_rs = {'queID': pid, 'problem': sentence.sentence, 'problem_type': sentence.sentence_type,
                      'problem_index': sentences[sentence.sentence_type].index(sentence.id)}
        record_query_rs = ProblemRecord.objects.filter(problem_id__id=pid, user_id=user_id).order_by('-score')[:3]
        problem_rs['history'] = [(q.answer, int(q.score)) for q in record_query_rs]
        rs.append(problem_rs)

    return JsonResponse({'rs': rs, 'page_number': int(ceil(len(problem_id_set) / 6))})


# 获取数据统计
def get_staistic_for_user(request, user_id):
    rs = get_stastics_by_list(user_id)
    return JsonResponse({'rs': rs})


# 获取同义词
def syn_words(request, word):
    words = get_syn_words(word)
    return JsonResponse({'syn': words})


def get_user_level_by_id(request, user_id):
    level, score = get_user_level(user_id)
    return JsonResponse({'score': int(score), 'level': int(level)})
