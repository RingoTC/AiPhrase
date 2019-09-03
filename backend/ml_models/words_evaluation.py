# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import string
import nltk
# nltk.download(['wordnet', 'averaged_perceptron_tagger', 'punkt'])
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import spacy

# def removePunctuation(sentence):
#     '''
#     去掉英语句子中的标点符号,
#     返回没有标点符号的句子，
#     输入：句子，
#     输出：句子
#     '''
#     tokens = word_tokenize(sentence)
#     temp = []
#     for c in tokens:
#         if c not in string.punctuation:
#             temp.append(c)
#     newSentence = ' '.join(temp)
#     return newSentence

sp = spacy.load('en_core_web_sm')


def removePunctuation(sentence):
    '''
    去掉英语句子中的标点符号,
    返回没有标点符号的句子，
    输入：句子，
    输出：句子
    '''
    tokens = sp(sentence)
    temp = []
    for c in tokens:
        if c.pos_ == "PUNCT":
            continue
        temp.append(c.text)
    newSentence = ' '.join(temp)
    return newSentence


def get_word_lemma(tag):
    '''
    获取英语单词的原型，
    输入：[单词，词性]，
    输出：单词原型
    '''

    if tag[1].startswith('J'):
        word_pos = wordnet.ADJ
    elif tag[1].startswith('V'):
        word_pos = wordnet.VERB
    elif tag[1].startswith('R'):
        word_pos = wordnet.ADV
    else:
        word_pos = wordnet.NOUN

    wnl = WordNetLemmatizer()
    newWord = wnl.lemmatize(tag[0], pos=word_pos)
    return newWord


wordTop5000 = pd.read_csv('ml_models/Oxford_5000.csv', sep='\t')
wordTop5000List = list(wordTop5000['Word'].values)


def sentenceScore(sentence):
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

    score = 0.0
    length = len(sentence.split())
    if length <= 1:
        length = 1
    newSentence = removePunctuation(sentence)
    tokens = word_tokenize(newSentence)
    tags = pos_tag(tokens)
    for tag in tags:
        word_lemma = get_word_lemma(tag)
        if word_lemma in wordTop5000List:
            score = score + 1
    return (score / length)


if __name__ == '__main__':
    sentence = 'hello world , i love the world very much'
    print(sentence, '单词常见程度', sentenceScore('sentence'))
