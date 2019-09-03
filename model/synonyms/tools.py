from cube.api import Cube
import nltk
from nltk.corpus import wordnet
import joblib
from nltk.stem import WordNetLemmatizer
import random
import torch
from pytorch_transformers import (
    BertTokenizer,
    BertForMaskedLM,
)
import gensim
model_r = gensim.models.KeyedVectors.load_word2vec_format('/data/zly/GoogleNews-vectors-negative300.bin',limit=800000, binary=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

model.eval()
cube=Cube(verbose=True)
cube.load('en', tokenization=True, compound_word_expanding=False, tagging=True, lemmatization=True,parsing=False)
unigram_tagger=joblib.load("/data/zly/unigram_tagger.pkl")
lemmatizer = WordNetLemmatizer()

def convert(s):
    if (s=="NN" or  s=="NNS" or s=="NNP" or s=="NNPS"):
        return "NOUN"
    elif (s=="RB" or s=="RBR" or s=="RBS"):
        return"ADV"
    elif (s=="VB" or  s=="VBD" or s=="VBG" or s=="VBN" or s=="VBP" or s=="VBZ"):
        return "VERB"
    elif (s=="JJ" or s=="JJR" or s=="JJS"):
        return "ADJ"
    else:
        return ""
from functools import partial

def common_prefix(strings):

    if len(strings) == 1:
        return strings[0]

    prefix = strings[0]

    for string in strings[1:]:
        while string[:len(prefix)] != prefix and prefix:
            prefix = prefix[:len(prefix)-1]
        if not prefix:
            break

    return len(prefix)

def get_syn(string,ids):
   
    sentences=cube(string)
    
    tokenized_text1=['[CLS]']
    text=[]
    for sentence in sentences:
        for entry in sentence:
            text.append(entry.word)
    text.append('[SEP]')
    tokenized_text1+=text+text
    length = (len(tokenized_text1)-3)//2
    index=0
    cnt=0
    
    for sentence in sentences:
        for entry in sentence:
            cnt+=1
            index+=1
            
            if (not entry.word.replace("-"," ").replace("_"," ").isalpha()) :
                cnt-=1
                continue
            if (not cnt==ids):
                continue
            
            if (entry.upos=="NOUN" or entry.upos=="VERB" or entry.upos=="ADJ" or entry.upos=="ADV"):
                
                masked_idx = index + 1 + length
                tokenized_text=tokenized_text1.copy()
                tokenized_text[masked_idx] = '[MASK]'
                tokens_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
                segments_ids =[0]*(length+2)+[1]*(length+1)
                tokens_tensor = torch.tensor([tokens_ids])
                segments_tensors = torch.tensor([segments_ids])

                if torch.cuda.is_available():
                    tokens_tensor = tokens_tensor.to('cuda')
                    segments_tensors = segments_tensors.to('cuda')
                    model.to('cuda')
                with torch.no_grad():
                    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
                    predictions = outputs[0]

                topk_score, topk_index = torch.topk(predictions[0, masked_idx], 3)
                topk_tokens = tokenizer.convert_ids_to_tokens(topk_index.tolist())
                
                synonyms = []
                syns=set([])
                syns1=set([])
                
                lemma=""
                synsets=wordnet.synsets(entry.lemma)
                if (len(synsets)==0):
                    lemma=lemmatizer.lemmatize(entry.word)
                    synsets=wordnet.synsets(lemma)
                else:
                    lemma=entry.lemma
                cnt=0
                
                for syn in synsets:
                        for l in syn.lemmas():
                            synonyms.append(l.name())
                
                for word in synonyms+topk_tokens:
                    if (len(word)==1):
                        continue
                    tags=[tag[1] for tag in unigram_tagger.tag([word])]
                    if (word.lower()==entry.word or word.lower()==lemma):
                        continue
                    syn_word=word.replace("-"," ").replace("_"," ")
                        
                    if (not all(x.isalpha() or x.isspace() for x in syn_word)
                    or ((word in topk_tokens) and(not None in tags)and (not entry.upos in [convert(tag) for tag in tags]))
                    or ((word in synonyms) and(not None in tags)and (not entry.upos in [convert(tag) for tag in tags]))
                        or (common_prefix([syn_word,entry.word])>=4 and len(syn_word)-len(lemma)<=3)):
                        continue
                    if (word in synonyms):
                        syns1.add(syn_word)
                    else:
                        syns.add(syn_word)
                        
                
                list1=[]
                list2=[]
                for word in syns1:
                    if (not all(a in model_r.vocab for a in word.split())):
                        
                        list2.append(word.split(" "))
                    else:
                        
                        list1.append(word.split(" "))
              
                if (lemma in model_r.vocab):
                    list.sort(list1,key=partial(model_r.n_similarity,ws2=[lemma]),reverse=True)
                
                list3=[]
                list4=[]
                for word in syns:
                    if (not all(a in model_r.vocab for a in word.split())):
                        
                        list4.append(word.split(" "))
                    else:
                        
                        list3.append(word.split(" "))
                if (lemma in model_r.vocab):
                    list.sort(list3,key=partial(model_r.n_similarity,ws2=[lemma]),reverse=True)
                list4+=list2
                
                list3+=list1
                list5=[]
                for element in list3:
                    if (not lemma in model_r.vocab)or(model_r.n_similarity(element,[lemma])>0.2) or(not element in list1):
                        s=""
                        for parts in element:
                            s+=parts+" "
                        list5.append(s[:-1])
                
                list1=list5
                list5=[]
                for element in list4:
                    s=""
                    for parts in element:
                        s+=parts+" "
                    list5.append(s[:-1])
                if (len(list1)<5):
                    list1+=list5[0:min(len(list5),5-len(list1))]
                return list1[:5]
            else:
                return []
    return []

                
