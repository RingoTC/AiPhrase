# import random
# import math
# from collections import OrderedDict
# from tqdm import tqdm
import torch
# from torch.nn import CrossEntropyLoss
# import torch.nn.functional as F
# from torchnlp.datasets import smt_dataset
from pytorch_transformers import (
    BertModel,
    BertTokenizer,
    BertForMaskedLM,
    # BertForTokenClassification,
    # AdamW,
    # WarmupLinearSchedule,
)

# from pytorch_pretrained_bert import BertModel, BertTokenizer

def bert_init():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    bert = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    return tokenizer,model,bert


def torch_synomy(tokenizer,model,bert,inputs):
    print("input sentence:the man loves the woman//then input index:3 to substitute word loves")
    # inputs = input("Source sentence:")
    # 第一种 去掉标点
    # inputs = inputs.replace('.','').replace(',',' ').replace('?','')
    # 第二种 记住标点
    inputs = inputs.replace(',', ' , ').replace('.', ' . ').replace('?', ' ? ').replace('!', ' ! ')
    text = "[CLS] " + inputs + " [SEP] " + inputs + " [SEP]"
    # 第三种 用原分词 但需要自己输入单词
    # tokenized_text = tokenizer.tokenize(text)
    print(text)
    tokenized_text = text.split()
    print(tokenized_text)
    length = (len(tokenized_text) - 3) // 2
    print(length)
    total_answer = []
    # to do
    for index_idx in range(1, length + 1):
        if tokenized_text[index_idx] not in [',', '.', '?', '!']:
            masked_idx = index_idx + 1 + length
            print(masked_idx)
            origin = tokenized_text[masked_idx]
            tokenized_text[masked_idx] = '[MASK]'
            tokens_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
            segments_ids = [0] * (length + 2) + [1] * (length + 1)
            tokens_tensor = torch.tensor([tokens_ids])
            segments_tensors = torch.tensor([segments_ids])
            # if torch.cuda.is_available():
            #     tokens_tensor = tokens_tensor.to('cuda')
            #     segments_tensors = segments_tensors.to('cuda')
            #     #model.to('cuda')
            with torch.no_grad():
                outputs = model(tokens_tensor, token_type_ids=segments_tensors)
                print(outputs)
                predictions = outputs[0]

            topk_score, topk_index = torch.topk(predictions[0, masked_idx], 5)
            topk_tokens = tokenizer.convert_ids_to_tokens(topk_index.tolist())
            print(f'Input: {tokenized_text}')
            print(f'Top5: {topk_tokens}')
            pos_answer = []
            for i in range(0, 5):
                sentence = ''
                tmp = tokenized_text[index_idx]
                tokenized_text[index_idx] = topk_tokens[i]
                j = 0
                for j in range(1, length):
                    sentence = sentence + tokenized_text[j] + ' '
                sentence = sentence + tokenized_text[j + 1]
                tokenized_text[index_idx] = tmp
                pos_answer.append(sentence)
            total_answer.append(pos_answer)
            tokenized_text[masked_idx] = origin
    return total_answer


if __name__ == '__main__':
    sentence = 'hello , i have a dream'
    tokenizer, model, bert = bert_init()
    print(sentence)
    print(torch_synomy(tokenizer, model, bert, sentence))
