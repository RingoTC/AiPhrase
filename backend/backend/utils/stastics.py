from backend.models import ProblemRecord
import datetime
import numpy as np
from backend.utils.evaluate import id_category

from django.utils import timezone

tz = timezone.get_current_timezone()


# 获取用户当天的题目数量
def get_user_problem_record_num(user_id, start=None, end=None, days=None, unique=True):
    if days is None:
        query_rs = ProblemRecord.objects.filter(user_id=user_id).values('problem_id')
    else:
        if end is None:
            end = datetime.datetime.now().astimezone(tz=tz)
        if start is None:
            start = datetime.date.today() + datetime.timedelta(days=-days)
        start = datetime.datetime(start.year, start.month, start.day, 0, 0, 0).astimezone(tz=tz)
        print(start, '=====>', end)
        query_rs = ProblemRecord.objects.filter(user_id=user_id, add_date__range=(start, end)).values('problem_id')
    ids = [q['problem_id'] for q in query_rs]
    if unique:
        ids = set(ids)
    return len(ids)


def get_day_week_monthly_num_problem(user_id, unique=True):
    num_day = get_user_problem_record_num(user_id=user_id, days=1, unique=unique)
    num_week = get_user_problem_record_num(user_id=user_id, days=7, unique=unique)
    num_month = get_user_problem_record_num(user_id=user_id, days=30, unique=unique)
    num_all = get_user_problem_record_num(user_id=user_id, days=None, unique=unique)
    if unique:
        rs = {'num_problems': {'day': num_day, 'week': num_week, 'month': num_month, 'all': num_all}}

    else:
        rs = {'num_problem_records': {'day': num_day, 'week': num_week, 'month': num_month, 'all': num_all}}
    return rs


def average_scores(user_id):
    records = ProblemRecord.objects.filter(user_id=user_id)

    if records:
        scores = np.array([r.score for r in records])
        num_85 = len(scores[[scores > 85]])
        num_70_85 = len(scores[(scores <= 85) & (scores > 70)])
        num_70 = len(scores[scores <= 70])
        mean_score = np.mean(scores)
        rs = {'num_scores': {'mean_scores': str(round(mean_score, 1)), '85': num_85, '70_85': num_70_85, '70': num_70}}
    else:
        rs = {'num_scores': {'mean_scores': 0, '85': 0, '70_85': 0, '70': 0}}
    return rs


def avegrage_details(user_id):
    records = ProblemRecord.objects.filter(user_id=user_id)

    details = {}
    for record in records:
        for detail in record.recorddetail_set.all():
            category_id = detail.category_id
            if category_id not in details:
                details[category_id] = []
            details[category_id].append(detail.value)
    rs = []
    for key in range(1, 6):
        # details[key] = {'name': id_category[key], 'average_value': str(round(np.average(details[key]), 2)), 'id': key}
        if key in details:
            rs.append(str(round(np.average(details[key]), 2)))
        else:
            rs.append(str(0))

    return {'avegrage_details': rs}


def get_stastics(user_id):
    rs = {}
    problems = get_day_week_monthly_num_problem(user_id)
    rs.update(problems)
    records = get_day_week_monthly_num_problem(user_id, unique=False)
    rs.update(records)
    scores = average_scores(user_id)
    rs.update(scores)
    details = avegrage_details(user_id)
    rs.update(details)

    return rs


def get_stastics_by_list(user_id):
    rs = []
    # 各指标数据
    details = avegrage_details(user_id)
    rs.append(details['avegrage_details'])
    # 分数数据
    scores = average_scores(user_id)['num_scores']
    score_list = [scores['mean_scores'], scores['85'], scores['70_85'], scores['70']]
    rs.append(score_list)
    # 做题次数
    records = get_day_week_monthly_num_problem(user_id, unique=False)['num_problem_records']
    record_list = [records['all'], records['day'], records['week'], records['month']]
    rs.append(record_list)
    # 题目数量
    problems = get_day_week_monthly_num_problem(user_id)['num_problems']
    problem_list = [problems['all'], problems['day'], problems['week'], problems['month']]
    rs.append(problem_list)

    return rs


if __name__ == '__main__':
    print('user 1 problems num:', get_user_problem_record_num(1))
