import  numpy as np
from stanfordcorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('stanford-corenlp-full-2016-10-31')
while(True):
    sentence1 = input("sentence1:")
    sentence2 = input("sentence2:")
    depecy1 = nlp.dependency_parse(sentence1)
    print(depecy1)
    depecy1 = np.array(depecy1)[:, 0]
    depecy2 = nlp.dependency_parse(sentence2)
    print(depecy2)
    depecy2 = np.array(depecy2)[:, 0]
    union = set(depecy1) & set(depecy2)
    union = [i for i in list(union) if i not in ['ROOT','punc','punct']]
    depecy1 = [i for i in list(depecy1) if i not in ['ROOT','punc','punct']]
    print(union)
    print("dependency result(similar)")
    print(float(len(union)) / len(depecy1))
    ## add get sentence type
    ## GuideWord = [that, which, who, whom, whose, as, when, where, why, whether]
    sentence_map = {"IP":"Simple clause",
                    "WDT":"Clause",
                    "advcl": "Adverbial clause",
                    "csubj ": "Clause",
                    "xsubj":"Clause",
                    "ccomp": "Clause",
                    "complm": "Clause",
                    "mark":"Clause", # that,whether,because,when
                    "pass":"Passive form",
                    "infmod": "infinitival modifier", # to do
                    }
    sentence_type = [sentence_map[i] for i in list(depecy2) if i in sentence_map.keys()]
    sentence_type = list(set(sentence_type))
    print(sentence_type)

    print("====================")
    print("Constituency result(complex)")
    tree1 = nlp.parse(sentence1)
    tree2 = nlp.parse(sentence2)
    print(tree1)
    print(tree2)
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
    print(float(result2) / (result1))



