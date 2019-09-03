import sys
AROOT_PATH = "/data/"
sys.path.append(AROOT_PATH+"react1/last_version/bert")
import run_classifier
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings
warnings.filterwarnings('ignore')

def inferencePairs(est,processor,tokenizer,question1,question2):


    print("inferencing..........")
    MAX_SEQ_LENGTH = 200
    sent_pairs = [(question1, question2)]
    print("sentence1: "+ question1)
    print("sentence2: "+ question2)

    predict_examples = processor.get_predict_examples(sent_pairs)
    label_list = processor.get_labels()
    predict_features = run_classifier.convert_examples_to_features(predict_examples, label_list, MAX_SEQ_LENGTH,
                                                                   tokenizer)
    predict_input_fn = run_classifier.input_fn_builder(predict_features,
                                                       seq_length=MAX_SEQ_LENGTH,
                                                       is_training=False,
                                                       drop_remainder=False)
    result = list(est.predict(input_fn=predict_input_fn))
    print("Prediction :", result[0]['probabilities'][1])
    return result[0]['probabilities'][1]
def inference():

    import os
    import sys
    if not 'bert' in sys.path:
      sys.path += ['bert']
    sys.path.append(AROOT_PATH+"react1/last_version/bert")
    TASK_DATA_DIR = AROOT_PATH+'react1/last_version/glue_data/QQP'
    import modeling
    import optimization
    import tokenization
    import run_classifier
    import  tensorflow as tf
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


    BERT_MODEL = 'uncased_L-12_H-768_A-12' #@param {type:"string"}
    BERT_PRETRAINED_DIR = AROOT_PATH+'react1/last_version/cloud-tpu-checkpoints/bert/' + BERT_MODEL
    LEARNING_RATE = 2e-5
    NUM_TRAIN_EPOCHS = 2.0
    WARMUP_PROPORTION = 0.1
    MAX_SEQ_LENGTH = 200

    processor = QQPProcessor()
    label_list = processor.get_labels()

    CONFIG_FILE = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
    INIT_CHECKPOINT = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')
    DO_LOWER_CASE = BERT_MODEL.startswith('uncased')
    OUTPUT_DIR = AROOT_PATH+'react1/last_version/out_dir'

    TRAIN_BATCH_SIZE = 32 # For GPU, reduce to 16
    EVAL_BATCH_SIZE = 8
    PREDICT_BATCH_SIZE = 8

    print("################  Processing Training Data #####################")
    TRAIN_TF_RECORD = os.path.join(OUTPUT_DIR, "train.tf_record")
    train_examples = processor.get_train_examples(TASK_DATA_DIR)
    num_train_examples = len(train_examples)
    num_train_steps = int( num_train_examples / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)
    num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

    model_fn = model_fn_builder(
        bert_config=modeling.BertConfig.from_json_file(CONFIG_FILE),
        num_labels=len(label_list),
        init_checkpoint=INIT_CHECKPOINT,
        learning_rate=LEARNING_RATE,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=False,
        use_one_hot_embeddings=True)


    SAVE_CHECKPOINTS_STEPS = 1000
    SAVE_CHECKPOINTS_STEPS = 1000
    ITERATIONS_PER_LOOP = 1000
    NUM_TPU_CORES = 8
    run_config = tf.contrib.tpu.RunConfig(
        cluster=None,
        model_dir=OUTPUT_DIR,
        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=ITERATIONS_PER_LOOP,
            num_shards=NUM_TPU_CORES,
            per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))



    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        predict_batch_size=PREDICT_BATCH_SIZE)







    MAX_SEQ_LENGTH = 200
    PREDICT_BATCH_SIZE = 8

    VOCAB_FILE = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')
    DO_LOWER_CASE = BERT_MODEL.startswith('uncased')

    tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=DO_LOWER_CASE)



    sent_pairs = [("how can i improve my english?", "how can i become fluent in english?"), ("How can i recover old gmail account ?","How can i delete my old gmail account ?"),
                 ("How can i recover old gmail account ?","How can i access my old gmail account ?"),
                  ("How can I increase the speed of my internet connection while using a VPN?","How can Internet speed be increased by hacking through DNS?")]

    print("*******  Predictions on Custom Data ********")
    # create `InputExample` for custom examples
    predict_examples = processor.get_predict_examples(sent_pairs)
    num_predict_examples = len(predict_examples)



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

    print("test finished")
    return estimator,processor,tokenizer

if __name__ == "__main__":
    inference()
