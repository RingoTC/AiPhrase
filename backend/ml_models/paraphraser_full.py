import sys

#sys.path.append('/data/lxd/paraphraser')
sys.path.append('/home/fyyc/codes/deecamp/MinistAiCompose/lxd/paraphraser')
from inference_ver2 import model_init, inference_full

paraphraser, model = model_init(0.5)  ## restore model


def paraphraser_full(sentence):
    return inference_full(model, sentence)


if __name__ == '__main__':

    source_sentence = 'I like apple'
    print(inference_full(model, source_sentence))
    print('next')
    print(inference_full(model, 'I like swimming'))
    print(inference_full(model,
                         'Speaking loudly in public places is considered impolite in both Chinese and Western countries.'))
