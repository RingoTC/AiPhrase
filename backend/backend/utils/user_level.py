from backend.models import User, ProblemRecord
import numpy as np


def get_user_level(user_id):
    records = ProblemRecord.objects.filter(user_id=user_id)
    score = 0
    if records:
        score += np.sum([r.score for r in records])
    level = np.ceil(np.log2(score / 500 + 1))
    print('score:', score, 'level:', level)
    return level, score


if __name__ == '__main__':
    for score in [1, 100, 200, 500, 1000, 1500, 2000, 2001, 3000, 4000, 4001, 7000, 7001, 8000]:
        level = np.ceil(np.log2(score / 500 + 1))
        print('score:', score, 'level:', level)
