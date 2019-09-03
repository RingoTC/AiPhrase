#!/usr/bin/env python
# coding: utf-8

# # BERT FineTuning on Quora Questions Pairs 
# 
# ---
# 
# In this Colab Notebook, We will try to reproduce state of the art results on Quora Questions Pairs using BERT Model FineTuning. 
# 
# If you are not familiar with BERT, Please visit [The Illustrated BERT](http://jalammar.github.io/illustrated-bert/), [BERT Research Paper](https://arxiv.org/abs/1810.04805) and [BERT Github Repo](https://github.com/google-research/bert).
# 
# <table class="tfo-notebook-buttons" align="left" >
#  <td>
#     <a target="_blank" href="https://colab.research.google.com/drive/1dCbs4Th3hzJfWEe6KT-stIVDMqHZSA5V"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/drc10723/bert_quora_question_pairs"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View Copy on GitHub</a>
#   </td>
# </table>
# 
# \
# 
# \
# 
# This colab notebook supports both TPU and GPU runtype.

# ## Setting Up Environment 
# 
# 
# **USE_TPU :-** True, If you want to use TPU runtime. First change Colab Notebook runtype to TPU
# 
# **BERT_MODEL:-**  Choose BERT model
# 1.   **uncased_L-12_H-768_A-12**: uncased BERT base model
# 2.   **uncased_L-24_H-1024_A-16**: uncased BERT large model
# 3.   **cased_L-12_H-768_A-12:** cased BERT large model
# 
# **BUCKET:- ** Add bucket details, It is necessary to add bucket for TPU. For GPU runtype, If Bucket is empty, We will use disk.
# 
# For more details on how to setup TPU, Follow [Colab Notebook](https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb#scrollTo=RRu1aKO1D7-Z) 
# 
# 

# In[1]:


import os
import sys
import json
import datetime
import pprint
import tensorflow as tf

# Authenticate, so we can access storage bucket and TPU


# If you want to use TPU, first switch to tpu runtime in colab
USE_TPU = False #@param{type:"boolean"}

# We will use base uncased bert model, you can give try with large models
# For large model TPU is necessary
BERT_MODEL = 'uncased_L-12_H-768_A-12' #@param {type:"string"}

# BERT checkpoint bucket
BERT_PRETRAINED_DIR = './cloud-tpu-checkpoints/bert/' + BERT_MODEL
print('***** BERT pretrained directory: {} *****'.format(BERT_PRETRAINED_DIR))


# Bucket for saving checkpoints and outputs
BUCKET = '' #@param {type:"string"}
if BUCKET!="":
  OUTPUT_DIR = './{}/outputs'.format(BUCKET)
  tf.gfile.MakeDirs(OUTPUT_DIR)
elif USE_TPU:
  raise ValueError('Must specify an existing GCS bucket name for running on TPU')
else:
  OUTPUT_DIR = 'out_dir'
  #os.mkdir(OUTPUT_DIR)
print('***** Model output directory: {} *****'.format(OUTPUT_DIR))

if USE_TPU:
  # getting info on TPU runtime
  assert 'COLAB_TPU_ADDR' in os.environ, 'ERROR: Not connected to a TPU runtime; Change notebook runtype to TPU'
  TPU_ADDRESS = 'grpc://' + os.environ['COLAB_TPU_ADDR']
  print('TPU address is', TPU_ADDRESS)


# ## Clone BERT Repo and Download Quora Questions Pairs Dataset
# 

# In[2]:


# Clone BERT repo and add bert in system path
if not 'bert' in sys.path:
  sys.path += ['bert']
# Download QQP Task dataset present in GLUE Tasks.
TASK_DATA_DIR = 'glue_data/QQP'


# ## Model Configs and Hyper Parameters
# 

# In[3]:


import modeling
import optimization
import tokenization
import run_classifier

# Model Hyper Parameters
TRAIN_BATCH_SIZE = 32 # For GPU, reduce to 16
EVAL_BATCH_SIZE = 8
PREDICT_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 2.0
WARMUP_PROPORTION = 0.1
MAX_SEQ_LENGTH = 200 

# Model configs
SAVE_CHECKPOINTS_STEPS = 1000
ITERATIONS_PER_LOOP = 1000
NUM_TPU_CORES = 8
VOCAB_FILE = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')
CONFIG_FILE = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
INIT_CHECKPOINT = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')
DO_LOWER_CASE = BERT_MODEL.startswith('uncased')


# ## Read Questions Pairs
# 
# We will read data from TSV file and covert to list of InputExample. For `InputExample` and `DataProcessor` class defination refer to [run_classifier](https://github.com/google-research/bert/blob/master/run_classifier.py) file 

# In[4]:


class QQPProcessor(run_classifier.DataProcessor):
  """Processor for the Quora Question pair data set."""

  def get_train_examples(self, data_dir):
    """Reading train.tsv and converting to list of InputExample"""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir,"train.tsv")), 'train')

  def get_dev_examples(self, data_dir):
    """Reading dev.tsv and converting to list of InputExample"""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir,"dev.tsv")), 'dev')
  
  def get_test_examples(self, data_dir):
    """Reading train.tsv and converting to list of InputExample"""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir,"test.tsv")), 'test')
  
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
      if set_type=='test':
        # removing header and invalid data
        if i == 0 or len(line)!=3:
          print(guid, line)
          continue
        text_a = tokenization.convert_to_unicode(line[1])
        text_b = tokenization.convert_to_unicode(line[2])
        label = 0 # We will use zero for test as convert_example_to_features doesn't support None
      else:
        # removing header and invalid data
        if i == 0 or len(line)!=6:
          continue
        text_a = tokenization.convert_to_unicode(line[3])
        text_b = tokenization.convert_to_unicode(line[4])
        label = int(line[5])
      examples.append(
          run_classifier.InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_labels(self):
    "return class labels"
    return [0,1]


# ## Convert to Features
# 
# We will read examples and tokenize using Wordpiece based tokenization. Finally We will convert to `InputFeatures`.
# 
# BERT follows below tokenization procedure
# 1.   Instantiate an instance of tokenizer = tokenization.FullTokenizer
# 2.   Tokenize the raw text with tokens = tokenizer.tokenize(raw_text).
# 3.   Truncate to the maximum sequence length.
# 4.   Add the [CLS] and [SEP] tokens in the right place.
# 
# We need to create `segment_ids`, `input_mask` for `InputFeatures`. `segment_ids` will be `0` for question1 tokens and `1` for question2 tokens.
# 
# We will use following functions from [run_classifier](https://github.com/google-research/bert/blob/master/run_classifier.py) file for converting examples to features :-
# 
# 
# 1.   `convert_single_example` :- Converts a single `InputExample` into a single `InputFeatures`.
# 2.   `file_based_convert_examples_to_features` :- Convert a set of `InputExamples` to a TF_Record file.
# 
# For more details observe outputs for below cells
# 
# 

# In[5]:


# Instantiate an instance of QQPProcessor and tokenizer
processor = QQPProcessor()
label_list = processor.get_labels()
tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=DO_LOWER_CASE)


# In[6]:


# Converting training examples to features
print("################  Processing Training Data #####################")
TRAIN_TF_RECORD = os.path.join(OUTPUT_DIR, "train.tf_record")
train_examples = processor.get_train_examples(TASK_DATA_DIR)
num_train_examples = len(train_examples)
num_train_steps = int( num_train_examples / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)
#run_classifier.file_based_convert_examples_to_features(train_examples, label_list, MAX_SEQ_LENGTH, tokenizer, TRAIN_TF_RECORD)


# ## Creating Classification Model

# In[7]:


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  # Bert Model instant 
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # Getting output for last layer of BERT
  output_layer = model.get_pooled_output()
  
  # Number of outputs for last layer
  hidden_size = output_layer.shape[-1].value
  
  # We will use one layer on top of BERT pretrained for creating classification model
  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
    
    # Calcaulte prediction probabilites and loss
    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits, probabilities)


# ## Model Function Builder for Estimator
# 
# Based on mode, We will create optimizer for training, evaluation metrics for evalution and estimator spec

# In[8]:


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  
    """The `model_fn` for TPUEstimator."""

    # reading features input
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)
    
    # checking if training mode
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    
    # create simple classification model
    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)
    
    # getting variables for intialization and using pretrained init checkpoint
    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      # defining optimizar function
      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
      
      # Training estimator spec
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:
      # accuracy, loss, auc, F1, precision and recall metrics for evaluation
      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        accuracy = tf.metrics.accuracy(
            labels=label_ids, predictions=predictions, weights=is_real_example)
        f1_score = tf.contrib.metrics.f1_score(
            label_ids,
            predictions)
        auc = tf.metrics.auc(
            label_ids,
            predictions)
        recall = tf.metrics.recall(
            label_ids,
            predictions)
        precision = tf.metrics.precision(
            label_ids,
            predictions) 
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
            "f1_score": f1_score,
            "auc": auc,
            "precision": precision,
            "recall": recall
        }

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits, is_real_example])
      # estimator spec for evalaution
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      # estimator spec for predictions
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities},
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


# ## Creating TPUEstimator

# In[9]:


# Define TPU configs
if USE_TPU:
  tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(TPU_ADDRESS)
else:
  tpu_cluster_resolver = None
run_config = tf.contrib.tpu.RunConfig(
    cluster=tpu_cluster_resolver,
    model_dir=OUTPUT_DIR,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
    tpu_config=tf.contrib.tpu.TPUConfig(
        iterations_per_loop=ITERATIONS_PER_LOOP,
        num_shards=NUM_TPU_CORES,
        per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))


# In[16]:


# create model function for estimator using model function builder
model_fn = model_fn_builder(
    bert_config=modeling.BertConfig.from_json_file(CONFIG_FILE),
    num_labels=len(label_list),
    init_checkpoint=INIT_CHECKPOINT,
    learning_rate=LEARNING_RATE,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps,
    use_tpu=USE_TPU,
    use_one_hot_embeddings=True)


# In[17]:


# Defining TPU Estimator
estimator = tf.contrib.tpu.TPUEstimator(
    use_tpu=USE_TPU,
    model_fn=model_fn,
    config=run_config,
    train_batch_size=TRAIN_BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE,
    predict_batch_size=PREDICT_BATCH_SIZE)


# ## Finetune Training
# 
# 

# In[1]:


# Train the model.
print('QQP on BERT base model normally takes about 1 hour on TPU and 15-20 hours on GPU. Please wait...')
print('***** Started training at {} *****'.format(datetime.datetime.now()))
print('  Num examples = {}'.format(num_train_examples))
print('  Batch size = {}'.format(TRAIN_BATCH_SIZE))
tf.logging.info("  Num steps = %d", num_train_steps)
# we are using `file_based_input_fn_builder` for creating input function from TF_RECORD file
train_input_fn = run_classifier.file_based_input_fn_builder(TRAIN_TF_RECORD,
                                                            seq_length=MAX_SEQ_LENGTH,
                                                            is_training=True,
                                                            drop_remainder=True)
estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
print('***** Finished training at {} *****'.format(datetime.datetime.now()))


# ## Evalute FineTuned model
# First we will evalute on Train set and Then on Dev set

# In[ ]:


# eval the model on train set.
# print('***** Started Train Set evaluation at {} *****'.format(datetime.datetime.now()))
# print('  Num examples = {}'.format(num_train_examples))
# print('  Batch size = {}'.format(EVAL_BATCH_SIZE))
# # eval input function for train set
# train_eval_input_fn = run_classifier.file_based_input_fn_builder(TRAIN_TF_RECORD,
#                                                            seq_length=MAX_SEQ_LENGTH,
#                                                            is_training=False,
#                                                            drop_remainder=True)
# # evalute on train set
# result = estimator.evaluate(input_fn=train_eval_input_fn, 
#                             steps=int(num_train_examples/EVAL_BATCH_SIZE))
# print('***** Finished evaluation at {} *****'.format(datetime.datetime.now()))
# print("***** Eval results *****")
# for key in sorted(result.keys()):
#   print('  {} = {}'.format(key, str(result[key])))


# In[ ]:


# Converting eval examples to features
# print("################  Processing Dev Data #####################")
# EVAL_TF_RECORD = os.path.join(OUTPUT_DIR, "eval.tf_record")
# eval_examples = processor.get_dev_examples(TASK_DATA_DIR)
# num_eval_examples = len(eval_examples)
# run_classifier.file_based_convert_examples_to_features(eval_examples, label_list, MAX_SEQ_LENGTH, tokenizer, EVAL_TF_RECORD)
# #
# 
# # In[ ]:
# 
# 
# # Eval the model on Dev set.
# print('***** Started Dev Set evaluation at {} *****'.format(datetime.datetime.now()))
# print('  Num examples = {}'.format(num_eval_examples))
# print('  Batch size = {}'.format(EVAL_BATCH_SIZE))
#
# #eval input function for dev set
# eval_input_fn = run_classifier.file_based_input_fn_builder(EVAL_TF_RECORD,
#                                                            seq_length=MAX_SEQ_LENGTH,
#                                                            is_training=False,
#                                                            drop_remainder=True)
# # evalute on dev set
# result = estimator.evaluate(input_fn=eval_input_fn, steps=int(num_eval_examples/EVAL_BATCH_SIZE))
# print('***** Finished evaluation at {} *****'.format(datetime.datetime.now()))
# print("***** Eval results *****")
# for key in sorted(result.keys()):
#   print('  {} = {}'.format(key, str(result[key])))


# 
# ## Evaluation Results
# 
# 
# ---
# Evaluation results are on BERT base uncased model. For reproducing similar results, train for 3 epochs.
# 
# 
# 
# |**Metrics** | **Train Set** | **Dev Set** |
# |---|---|---|
# |**Loss**|0.150|0.497|
# |**Accuracy**|0.969|0.907|
# |**F1**|0.959|0.875|
# |**AUC**|0.969|0.902|
# |**Precision**|0.949|0.864|
# |**Recall**|0.969|0.886|
# 

# ## Predictions on Model
# 
# First We will predict on custom examples.
# 
# For test set, We will get predictions and save in file.

# In[ ]:


# examples sentences, feel free to change and try
sent_pairs = [("how can i improve my english?", "how can i become fluent in english?"), ("How can i recover old gmail account ?","How can i delete my old gmail account ?"),
             ("How can i recover old gmail account ?","How can i access my old gmail account ?"),
              ("How can I increase the speed of my internet connection while using a VPN?","How can Internet speed be increased by hacking through DNS?")]


# In[ ]:


print("*******  Predictions on Custom Data ********")
# create `InputExample` for custom examples
predict_examples = processor.get_predict_examples(sent_pairs)
num_predict_examples = len(predict_examples)

# For TPU, We will append `PaddingExample` for maintaining batch size
if USE_TPU:
  while(len(predict_examples)%EVAL_BATCH_SIZE!=0):
    predict_examples.append(run_classifier.PaddingInputExample())

# Converting to features 
predict_features = run_classifier.convert_examples_to_features(predict_examples, label_list, MAX_SEQ_LENGTH, tokenizer)

print('  Num examples = {}'.format(num_predict_examples))
print('  Batch size = {}'.format(PREDICT_BATCH_SIZE))

# Input function for prediction
predict_input_fn = run_classifier.input_fn_builder(predict_features,
                                                seq_length=MAX_SEQ_LENGTH,
                                                is_training=False,
                                                drop_remainder=False)
result = list(estimator.predict(input_fn=predict_input_fn))
print(result)
for ex_i in range(num_predict_examples):
  print("****** Example {} ******".format(ex_i))
  print("Question1 :", sent_pairs[ex_i][0])
  print("Question2 :", sent_pairs[ex_i][1])
  print("Prediction :", result[ex_i]['probabilities'][1])


# In[ ]:


# Converting test examples to features
# print("################  Processing Test Data #####################")
# TEST_TF_RECORD = os.path.join(OUTPUT_DIR, "test.tf_record")
# test_examples = processor.get_test_examples(TASK_DATA_DIR)
# num_test_examples = len(test_examples)
# run_classifier.file_based_convert_examples_to_features(test_examples, label_list, MAX_SEQ_LENGTH, tokenizer, TEST_TF_RECORD)
# 
# 
# # In[ ]:
# 
# 
# # Predictions on test set.
# print('***** Started Prediction at {} *****'.format(datetime.datetime.now()))
# print('  Num examples = {}'.format(num_test_examples))
# print('  Batch size = {}'.format(PREDICT_BATCH_SIZE))
# # predict input function for test set
# test_input_fn = run_classifier.file_based_input_fn_builder(TEST_TF_RECORD,
#                                                            seq_length=MAX_SEQ_LENGTH,
#                                                            is_training=False,
#                                                            drop_remainder=True)
# tf.logging.set_verbosity(tf.logging.ERROR)
# # predict on test set
# result = list(estimator.predict(input_fn=test_input_fn))
# print('***** Finished Prediction at {} *****'.format(datetime.datetime.now()))
# 
# # saving test predictions
# output_test_file = os.path.join(OUTPUT_DIR, "test_predictions.txt")
# with tf.gfile.GFile(output_test_file, "w") as writer:
#   for (example_i, predictions_i) in enumerate(result):
#     writer.write("%s , %s\n" % (test_examples[example_i].guid, str(predictions_i['probabilities'][1])))
# 
