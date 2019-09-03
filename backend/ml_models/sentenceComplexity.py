# 句式复杂度，句子结构相似度
import numpy as np
from stanfordcorenlp import StanfordCoreNLP

# 导入模型，耗时较长
nlp = StanfordCoreNLP('/home/fyyc/codes/deecamp/MinistAiCompose/ykd/stanford/stanford-corenlp-full-2016-10-31')


# 基于依存分析判断句式相似度,返回相似度百分比0-1
def sentenceSim(sentence1, sentence2):
    depecy1 = nlp.dependency_parse(sentence1)
    print(depecy1)
    depecy1 = np.array(depecy1)[:, 0]
    depecy2 = nlp.dependency_parse(sentence2)
    print(depecy2)
    depecy2 = np.array(depecy2)[:, 0]
    union = set(depecy1) & set(depecy2)
    union = [i for i in list(union) if i not in ['ROOT', 'punc', 'punct']]
    depecy1 = [i for i in list(depecy1) if i not in ['ROOT', 'punc', 'punct']]
    print(union)
    print("dependency result(similar)")
    result = float(len(union)) / max(len(depecy1), len(depecy2))
    print("====================")
    return result


# 基于句法分析树判断句式结构复杂度，返回复杂度百分比0-1
#基于句法分析树判断句式结构复杂度，返回复杂度百分比0-1
def sentenceComplex(sentence1,sentence2):
#    print("Constituency result(complex)")
    tree1 = nlp.parse(sentence1)
    tree2 = nlp.parse(sentence2)
#    print(tree1)
#    print(tree2)
    tree1 = tree1.split('\n ')
    tree2 = tree2.split("\n ")
    result1 = 0
    for theight in tree1:
        tcount1 = 0
        for tmp in theight:
            if (tmp == '('):
                tcount1 += 1
        result1 = max(result1,tcount1)
    result2 = 0
    for theight in tree2:
        tcount2 = 0
        for tmp in theight:
            if (tmp == '('):
                tcount2 += 1
        result2 = max(result2,tcount2)
#    print(float(result2))
#    ans =abs(result2-result1)/min(result1,result2)
    ans = result2 / result1
    if ans > 1:
        return 1
    else:
        return ans


# 用于测试
if __name__ == '__main__':
    sentence1 = "what do you mean"
    sentence2 = "what are you talking about"
    print(sentenceSim(sentence1, sentence2))
    print(sentenceComplex(sentence1, sentence2))
