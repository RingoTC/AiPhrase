# -*- coding: UTF-8 -*-

# import curses
# from curses.ascii import isdigit
import nltk
from nltk.corpus import cmudict
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import pandas as pd
# import string
import spacy

d = cmudict.dict()
sp = spacy.load('en_core_web_sm')


def syllables(word):
    '''
    利用规则判断单词的音节数,
    输入：单词，
    输出：数字
    '''
    # referred from stackoverflow.com/questions/14541303/count-the-number-of-syllables-in-a-word
    count = 0
    vowels = 'aeiouy'
    word = word.lower()
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith('e'):
        count -= 1
    if word.endswith('le'):
        count += 1
    if count == 0:
        count += 1
    return count


def cmusyl(word):
    '''
    利用 cmu dataset 判断单词的音节数;
    如果单词不在 cmu dataset 里面则利用规则判断（函数：syllables）
    输入：单词，
    输出：数字list
    '''

    try:
        return [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]]
    except KeyError:
        # if word not found in cmudict
        return [syllables(word)]


def fresDescr(score):
    '''
    Flesch reading ease score 的含义
    输入：分数
    输出：描述语句
    '''
    score_range = {'Very easy to read. Easily understood by an average 11-year-old student.': 90,
                   'Easy to read. Conversational English for consumers.': 80,
                   'Fairly easy to read.': 70,
                   'Plain English. Easily understood by 13- to 15-year-old students.': 60,
                   'Fairly difficult to read.': 50,
                   'Difficult to read.': 30,
                   'Very difficult to read. Best understood by university graduates.': -1000}
    for k, v in score_range.items():
        if score >= v:
            return k


def fresScore(sentence):
    '''
    Flesch reading ease score 的含义
    输入：段落或单个句子
    输出：分数
    '''
    total_syl = 0
    punct = 0
    sen = sp(sentence)
    for i in range(len(sen)):
        if (sen[i].pos_ == "PUNCT"):
            punct = punct + 1
            continue
        total_syl = total_syl + cmusyl(sen[i].text)[0]

    score = 206.835 - 1.015 * ((len(sen) - punct) / len(sent_tokenize(sentence))) - 84.6 * (
            total_syl / (len(sen)))
    # result = {'No. of sentences':len(sent_tokenize(sentence)),
    #         'No. of words':len(sen)-punct,
    #         'Average syllables per word':total_syl/(len(sen)-punct),
    #         'Readability score':score}
    return score
    # print(total_syl, sent_tokenize(sentence), len(sen), -punct)


if __name__ == '__main__':
    import time

    now = time.time()
    sentence = ' hello world i love listen the song'
    print(sentence, fresScore(sentence))
    end = time.time()
    print('time:', end - now)
