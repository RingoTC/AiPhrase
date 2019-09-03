import sys

sys.path.append('/home/fyyc/codes/deecamp/MinistAiCompose/zly')


# from xkcd import get_syn


import time

# First, you're going to need to import wordnet:
from nltk.corpus import wordnet

def get_syn(word):
    # Then, we're going to use the term "program" to find synsets like so:
    syns = wordnet.synsets(word)
    words = syns[:5]
    words = [w.lemmas()[0].name() for w in words]
    # Just the word:
    return list(set(words))

def get_syn_words(word):
    return get_syn(word)


if __name__ == '__main__':
    tic = time.time()
    sentence = 'hello i wanto to find som synonym.'
    toc = time.time()
    print(sentence)
    print(get_syn(sentence.split()[0]))
    print(toc - tic)
    while True:
        sentence = input('input>')
        print(get_syn(sentence))
