# # -*- coding: utf-8 -*-
# # @Time  : 2019/8/11 19:38
# # @Author : liuti
# # @Project : suc_bimpm
# # @FileName: predict_inte.py
# # @Software: PyCharm
# import torch
# from torch import nn
# from torchtext.vocab import GloVe, Vectors
# from torchtext import data
# from torch.autograd import Variable
#
# ltt_base_dir = '/home/fyyc/codes/deecamp/MinistAiCompose/BIMPM'
# import sys
#
# sys.path.append(ltt_base_dir)
# from model.BIMPM import BIMPM
# from model.utils import Quora
# import os
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # vectors = Vectors(name='glove.840B.300d.txt', cache=os.path.join(ltt_base_dir, '.vector_cache'))
# use_char_emb = False
# model_path = os.path.join(ltt_base_dir, "saved_models/BIBPM_Quora_0.83.pt")
#
#
# def load_model(args, data):
#     model = BIMPM(args, data)
#     model.load_state_dict(torch.load(model_path))
#     model.to(args.device)
#     model.eval()
#     return model
#
#
# import pickle
# # import argparse
# # parser = argparse.ArgumentParser()
# # parser.add_argument('--batch-size', default=64, type=int)
# # parser.add_argument('--char-dim', default=20, type=int)
# # parser.add_argument('--char-hidden-size', default=50, type=int)
# # parser.add_argument('--dropout', default=0.1, type=float)
# # parser.add_argument('--data-type', default='Quora', help='available: SNLI or Quora')
# # parser.add_argument('--epoch', default=10, type=int)
# # # parser.add_argument('--gpu', default=0, type=int)
# # parser.add_argument('--hidden-size', default=100, type=int)
# # parser.add_argument('--learning-rate', default=0.001, type=float)
# # parser.add_argument('--num-perspective', default=20, type=int)
# # parser.add_argument('--use-char-emb', default=False, action='store_true')
# # parser.add_argument('--word-dim', default=300, type=int)
# #
# # parser.add_argument('--model-path', default="/home/fyyc/codes/deecamp/MinistAiCompose/BIMPM/saved_models/BIBPM_Quora_0.83.pt")
# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # parser.add_argument('--device', default=device, type=int)
# #
# # args = parser.parse_args()
# # with open('ltt_args.pkl', 'wb') as f:
# #     pickle.dump(args, f)
# # raise Exception('stop here')
#
# with open('/home/fyyc/codes/deecamp/MinistAiCompose/AIEditorBackend/ltt_args.pkl', 'rb') as f:
#     args = pickle.load(f)
#
# quora_data = Quora(args)
# vocab = quora_data.TEXT.vocab
# setattr(args, 'char_vocab_size', len(quora_data.char_vocab))
# setattr(args, 'word_vocab_size', len(quora_data.TEXT.vocab))
# setattr(args, 'class_size', len(quora_data.LABEL.vocab))
# setattr(args, 'max_word_len', quora_data.max_word_len)
# model = load_model(args, quora_data)
#
#
# def bimpmPred(sentence1, sentence2):
#     # 用新加载的模型进行预测
#     vec_1 = [[vocab.stoi[word] for word in sentence1]]
#     vec_2 = [[vocab.stoi[word] for word in sentence2]]
#     vec_1 = torch.tensor(vec_1, dtype=torch.long).to(device)
#     vec_2 = torch.tensor(vec_2, dtype=torch.long).to(device)
#     kwargs = {'p': vec_1, 'h': vec_2}
#
#     with torch.no_grad():
#         pred = model(**kwargs)
#         pred_r = pred.cpu().numpy()[0][1]
#     return pred_r
#
#
# if __name__ == '__main__':
#     sentence1 = input("> ")
#     sentence2 = input("> ")
#     sim = bimpmPred(sentence1, sentence2)
#     print(sim)
