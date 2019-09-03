import pickle
from pathlib import Path
import os
from tensorflow.contrib import predictor
import requests
import sys

AROOT_PATH = '/home/fyyc/codes/deecamp/MinistAiCompose/last_version/'
sys.path.append(AROOT_PATH + 'bert')

import run_classifier, tokenization

MODEL_PATH = '/data/saved_model'  # '/home/fyyc/PycharmProjects/model_similarity/saved_model'

MAX_SEQ_LENGTH = 200


def get_processor_tokenizer():
    class QQPProcessor(run_classifier.DataProcessor):
        """Processor for the Quora Question pair data set."""

        def get_train_examples(self, data_dir):
            """Reading train.tsv and converting to list of InputExample"""
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), 'train')

        def get_dev_examples(self, data_dir):
            """Reading dev.tsv and converting to list of InputExample"""
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), 'dev')

        def get_test_examples(self, data_dir):
            """Reading train.tsv and converting to list of InputExample"""
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test.tsv")), 'test')

        def get_predict_examples(self, sentence_pairs):
            """Given question pairs, conevrting to list of InputExample"""
            examples = []
            for (i, qpair) in enumerate(sentence_pairs):
                guid = "predict-%d" % (i)
                # converting questions to utf-8 and creating InputExamples
                text_a = tokenization.convert_to_unicode(qpair[0])
                text_b = tokenization.convert_to_unicode(qpair[1])
                # We will add label  as 0, because None is not supported in converting to features
                examples.append(
                    run_classifier.InputExample(guid=guid, text_a=text_a, text_b=text_b, label=0))
            return examples

        def _create_examples(self, lines, set_type):
            """Creates examples for the training, dev and test sets."""
            examples = []
            for (i, line) in enumerate(lines):
                guid = "%s-%d" % (set_type, i)
                if set_type == 'test':
                    # removing header and invalid data
                    if i == 0 or len(line) != 3:
                        print(guid, line)
                        continue
                    text_a = tokenization.convert_to_unicode(line[1])
                    text_b = tokenization.convert_to_unicode(line[2])
                    label = 0  # We will use zero for test as convert_example_to_features doesn't support None
                else:
                    # removing header and invalid data
                    if i == 0 or len(line) != 6:
                        continue
                    text_a = tokenization.convert_to_unicode(line[3])
                    text_b = tokenization.convert_to_unicode(line[4])
                    label = int(line[5])
                examples.append(
                    run_classifier.InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            return examples

        def get_labels(self):
            "return class labels"
            return [0, 1]

    BERT_MODEL = 'uncased_L-12_H-768_A-12'  # @param {type:"string"}
    BERT_PRETRAINED_DIR = AROOT_PATH + 'cloud-tpu-checkpoints/bert/' + BERT_MODEL
    processor = QQPProcessor()
    VOCAB_FILE = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')
    DO_LOWER_CASE = BERT_MODEL.startswith('uncased')

    tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=DO_LOWER_CASE)
    return processor, tokenizer


def load_model():
    export_dir = MODEL_PATH
    subdirs = [x for x in Path(export_dir).iterdir()
               if x.is_dir() and 'temp' not in str(x)]
    latest = str(sorted(subdirs)[-1])
    print(latest)
    predict_fn = predictor.from_saved_model(latest)
    processor, tokenizer = get_processor_tokenizer()
    return predict_fn, processor, tokenizer


# predict_fn, processor, tokenizer = load_model()
#
#
# def inferencePairsFromGraph(question1, question2):
#     print("inferencing..........")
#     MAX_SEQ_LENGTH = 200
#     sent_pairs = [(question1, question2)]
#     print("sentence1: " + question1)
#     print("sentence2: " + question2)
#
#     predict_examples = processor.get_predict_examples(sent_pairs)
#     label_list = processor.get_labels()
#     predict_features = run_classifier.convert_examples_to_features(predict_examples, label_list, MAX_SEQ_LENGTH,
#                                                                    tokenizer)
#     feature = predict_features[0]
#     feature = {'input_ids': [feature.input_ids],
#                'input_mask': [feature.input_mask],
#                'segment_ids': [feature.segment_ids],
#                'label_ids': [feature.label_id]}
#
#     result = predict_fn(feature)['probabilities'][0][1]
#     print("Prediction :", result)
#     return result
processor, tokenizer = get_processor_tokenizer()


def inferencePairsFromGraph(question1, question2):

    print("inferencing by request tensorflow serving")
    MAX_SEQ_LENGTH = 200
    sent_pairs = [(question1, question2)]
    print("sentence1: " + question1)
    print("sentence2: " + question2)

    predict_examples = processor.get_predict_examples(sent_pairs)
    label_list = processor.get_labels()
    predict_features = run_classifier.convert_examples_to_features(predict_examples, label_list, MAX_SEQ_LENGTH,
                                                                   tokenizer)
    feature = predict_features[0]
    features = {'input_ids': [feature.input_ids],
                'input_mask': [feature.input_mask],
                'segment_ids': [feature.segment_ids],
                'label_ids': [feature.label_id]}
    response = requests.post(json={'inputs': features}, url='http://localhost:8511/v1/models/bert_similarity:predict')
    result = response.json()['outputs'][0][1]
    print("Prediction :", result)
    return result


def inferencePairListFromGraph(question1s, question2s):
    print("inferencing by request tensorflow serving")
    MAX_SEQ_LENGTH = 200
    sent_pairs = list(zip(question1s, question2s))

    predict_examples = processor.get_predict_examples(sent_pairs)
    label_list = processor.get_labels()
    predict_features = run_classifier.convert_examples_to_features(predict_examples, label_list, MAX_SEQ_LENGTH,
                                                                   tokenizer)
    features = {'input_ids': [],
                'input_mask': [],
                'segment_ids': [],
                'label_ids': []}
    for feature in predict_features:
        features['input_ids'].append(feature.input_ids)
        features['input_mask'].append(feature.input_mask)
        features['segment_ids'].append(feature.segment_ids)
        features['label_ids'].append(feature.label_id)

    response = requests.post(json={'inputs': features}, url='http://localhost:8511/v1/models/bert_similarity:predict')
    result = response.json()['outputs']
    result = [r[1] for r in result]
    print("Prediction :", result)
    return result


if __name__ == '__main__':
    sent1 = 'Adapting to protocol v5.1 for kernel b9ec758f-28eb-4b4f-abbc-8fc52ee73100'
    sent2 = 'Adapting to protocol v5.1 for kernel b9ec758f-28eb-4b4f-abbc-8fc52ee73100'
    inferencePairsFromGraph(sent1, sent2)
    sents1 = ['dd s dss ds d', 'dssd dssd', 'sdds ds']
    sents2 = ['dd s dss ds d', 'dssd dssd', 'sdds ds']
    print(inferencePairListFromGraph(sents1, sents2))
