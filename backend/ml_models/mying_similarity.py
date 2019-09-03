#!/usr/bin/env python
# coding: utf-8

# In[10]:


import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from keras.preprocessing.sequence import pad_sequences
import gc
import pickle
import requests
import numpy as np
# bst_model_path = 'ckpt/lstm_225_144_0.29_0.23mynew.h5'
# model = load_model(bst_model_path)  # sotre model parameters in .h5 file
with open('data/tokenizer1.pickle', 'rb') as f:
    tokenizer = pickle.load(f)


def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.

    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        text = [w for w in text if not w in stop_words]

    text = " ".join(text)

    # Remove punctuation from text
    # text = "".join([c for c in text if c not in punctuation])

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    # text = re.sub(r"\0s", "0", text) # It doesn't make sense to me
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    return (text)

def process_sentence_pair(test_q1,test_q2):
    test_texts_1 = []
    test_texts_2 = []

    test_texts_1.append(text_to_wordlist(test_q1, remove_stopwords=False, stem_words=False))
    test_texts_2.append(text_to_wordlist(test_q2, remove_stopwords=False, stem_words=False))

    Max_Sequence_Length = 60

    # tokenizer.fit_on_texts(test_texts_1 + test_texts_2)
    test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
    test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)
    del test_texts_1
    del test_texts_2
    gc.collect()

    print('test_sequences are ready!')

    test_data_1 = pad_sequences(test_sequences_1, maxlen=Max_Sequence_Length)
    test_data_2 = pad_sequences(test_sequences_2, maxlen=Max_Sequence_Length)

    return test_data_1,test_data_2


# In[2]:


def my_similarity_serving(sent1,sent2):
    test_data_1, test_data_2 = process_sentence_pair(sent1, sent2)
    inputs = {
        "signature_name": 'my_similarity',
        "inputs": {
            "sent1": test_data_1.tolist() + test_data_2.tolist(),
            "sent2": test_data_2.tolist() + test_data_1.tolist()
        }
    }
    rs = requests.post(json=inputs, url='http://localhost:8510/v1/models/mymodels:predict')
    outputs = rs.json()['outputs']
    rs = np.mean(outputs)
    print(outputs)
    print(rs)

    return rs

# In[8]:


#获取metadata
# meta = requests.get('http://localhost:8505/v1/models/similarity/metadata').json()
# meta


# In[24]:
if __name__ == '__main__':


    #进行预测
    sent1 = 'hello, i love tea coffe dog cat'
    sent2 = 'hello,i love tea hello world'
    my_similarity_serving(sent1,sent2)


    # In[ ]:




