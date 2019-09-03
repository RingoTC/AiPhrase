import tensorflow as tf
from embeddings import load_sentence_embeddings
from preprocess_data import preprocess_batch
from lstm_model import lstm_model
import language_check
import sys
sys.path.append('/home/fyyc/codes/deecamp/MinistAiCompose/AIEditorBackend/ml_models')
from similarity import inferencePairListFromGraph
from torch_synomy_lxd import bert_init, torch_synomy

class Paraphraser(object):
    '''Heart of the paraphraser model.  This class loads the checkpoint
    into the Tensorflow runtime environme   nt and is responsible for inference.
    Greedy and sampling based approaches are supported
    '''

    def __init__(self, checkpoint, GPU_rate):
        """Constructor.  Load vocabulary index, start token, end token, unk id,
        mask_id.  Restore checkpoint.

        Args:
            checkpoint: A path to the checkpoint
        """
        self.word_to_id, self.idx_to_word, self.embedding, self.start_id, self.end_id, self.unk_id, self.mask_id = load_sentence_embeddings()
        self.checkpoint = checkpoint
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_rate)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.model = lstm_model(self.sess, 'infer', 300, self.embedding, self.start_id, self.end_id, self.mask_id)
        saver = tf.train.Saver()
        saver.restore(self.sess, checkpoint)

    def sample_paraphrase(self, sentence, sampling_temp=1.0, how_many=1):
        """Paraphrase by sampling a distribution

        Args:
            sentence (str): A sentence input that will be paraphrased by
                sampling from distribution.
            sampling_temp (int) : A number between 0 an 1

        Returns:
            str: a candidate paraphrase of the `sentence`
        """
        return self.infer(1, sentence, self.idx_to_word, sampling_temp, how_many)

    def greedy_paraphrase(self, sentence):
        """Paraphrase using greedy sampler

        Args:
            sentence : The source sentence to be paraphrased.

        Returns:
            str : a candidate paraphrase of the `sentence`
        """

        return self.infer(0, sentence, self.idx_to_word, 0., 1)

    def infer(self, decoder, source_sent, id_to_vocab, temp, how_many):
        """ Perform inferencing.  In other words, generate a paraphrase
        for the source sentence.

        Args:
            decoder : 0 for greedy, 1 for sampling
            source_sent : source sentence to generate a paraphrase for
            id_to_vocab : dict of vocabulary index to word
            end_id : the end token
            temp : the sampling temperature to use when `decoder` is 1

        Returns:
            str : for the generated paraphrase
        """

        seq_source_words, seq_source_ids = preprocess_batch([source_sent] * how_many)
        # print(seq_source_words)
        # print(seq_source_ids)
        seq_source_len = [len(seq_source) for seq_source in seq_source_ids]
        # print(seq_source_len)

        feed_dict = {
            self.model['seq_source_ids']: seq_source_ids,
            self.model['seq_source_lengths']: seq_source_len,
            self.model['decoder_technique']: decoder,
            self.model['sampling_temperature']: temp
        }

        feeds = [
            self.model['predictions']
            # model['final_sequence_lengths']
        ]

        predictions = self.sess.run(feeds, feed_dict)[0]
        # print(predictions)
        return self.translate(predictions, decoder, id_to_vocab, seq_source_words[0])

    def translate(self, predictions, decoder, id_to_vocab, seq_source_words):
        """ Translate the vocabulary ids in `predictions` to actual words
        that compose the paraphrase.

        Args:
            predictions : arrays of vocabulary ids
            decoder : 0 for greedy, 1 for sample, 2 for beam
            id_to_vocab : dict of vocabulary index to word

        Returns:
            str : the paraphrase
        """
        translated_predictions = []
        # np_end = np.where(translated_predictions == end_id)
        for sent_pred in predictions:
            translated = []
            for pred in sent_pred:
                word = 'UUNNKK'
                if pred == self.end_id:
                    break
                if pred == self.unk_id:
                    # Search for rare word
                    for seq_source_word in seq_source_words:
                        if seq_source_word not in self.word_to_id:
                            word = seq_source_word
                else:
                    word = id_to_vocab[pred]
                translated.append(word)
            translated_predictions.append(' '.join(translated))
        return translated_predictions


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='./model/model-335000',
                        help='Checkpoint path')
    args = parser.parse_args()
    paraphraser = Paraphraser(args.checkpoint)


def model_init(GPU_rate):
    # import argparse
    # parser = argparse.ArgumentParser()
    #parser.add_argument('--checkpoint', type=str,
    #                    default='/data/lxd/paraphraser/model/model-335000',
    #                    ## /data/Lxd/Lxd_paraphraser/logs/train-20190730-074926/model-335000'
    #                    help='Checkpoint path')
    # args = parser.parse_args()
    paraphraser = Paraphraser('/home/fyyc/codes/deecamp/MinistAiCompose/lxd/paraphraser/model/model-335000', GPU_rate)
    lan_tool = language_check.LanguageTool("en-US")
    tokenizer, model_bert, bert = bert_init()
    mod = {}
    mod['paraphrase']=paraphraser
    mod['lan_tool'] = lan_tool
    mod['bert'] = {'tokenizer': tokenizer, 'model': model_bert, 'bert': bert}
    return paraphraser,mod


def inference(model, source_sentence):
    paraphrases = model.sample_paraphrase(source_sentence, sampling_temp=0.75, how_many=10)
    return paraphrases

    '''while 1:
        source_sentence = input("Source: ")
        #p = paraphraser.greedy_paraphrase(source_sentence)
        #print(p)
        paraphrases = paraphraser.sample_paraphrase(source_sentence,  sampling_temp=0.75, how_many=10)
        for i, paraphrase in enumerate(paraphrases):
            print("Paraph #{}: {}".format(i, paraphrase))'''


def rule_sentence_check(sentences):
    if not sentences:
        print('rule_check_result:')
        print([])
        return []
    temp = []
    # drop continous double words
    print(sentences)
    for sentence in sentences:
        if sentence[0].isalpha():
            word_list = sentence.strip().split(' ')
            length = len(word_list)

            for idx, item in enumerate(word_list):
                if idx == 0:
                    if length-1>0:
                        if item == word_list[idx + 1]:
                            break
                    else:
                        temp.append(sentence)
                        break
                elif idx<length-1:
                    if item == word_list[idx - 1] or item == word_list[idx + 1]:
                        break
                else:
                    if item == word_list[idx - 1]:
                        break
                    else:
                        temp.append(sentence)
    # drop duplicates
    new = []
    for sentence in temp:
        if sentence not in new:
            new.append(sentence)
    print('rule_check_result:')
    print(new)
    return new


def drop_dup(sentences):
    new = []
    if not sentences:
        return []
    for sentence in sentences:
        if sentence not in new:
            new.append(sentence)
    return new

def grammar_sentence_check(model, sentences):
    if not sentences:
        print('grammar_check_result:')
        print([])
        return []
    output = []
    print('grammar_checking')
    for sentence in sentences:
        sentence_temp = sentence.capitalize().replace(' ,',',').replace(' ?','?').replace(' !','!').replace(' .','.')
        print(sentence_temp)
        matches = model.check(sentence_temp)
        error_no = len(matches)
        if error_no == 0:
            output.append(sentence)
        else:
            newSentence = language_check.correct(sentence_temp, matches)
            matches_new = model.check(newSentence)
            if len(matches_new)==0:
                output.append(newSentence)
    print('grammar_check_result:')
    print(output)
    return output


def similarity_sentence_check(sentences, source_sentence):
    if not sentences:
        print('similarity_check_result:')
        print([])
        return []
    sentence_list = []
    source_sentences = [source_sentence for i in range(len(sentences))]
    result_list = inferencePairListFromGraph(sentences,source_sentences)
    for idx,sentence in enumerate(sentences):
        if result_list[idx] > 0.58:
            sentence_list.append((sentence,result_list[idx]))
    sort_list = sorted(sentence_list,key=lambda x:x[1],reverse=True)
    output = []
    for item in sort_list:
        output.append(item[0])
    print('similarity_check_result:')
    print(output)
    return output


def sentence_prettify(sentences):
    output = []
    for sentence in sentences:
        sentence = sentence.capitalize().replace(' ,',',').replace(' ?','?').replace(' !','!').replace(' .','.')
        sentence = sentence.replace(" n't","n't").replace(" 'll","'ll").replace(" 's","'s").replace(" 're","'re").replace(" 'm","'m")
        sentence = sentence.replace(" 'd","'d")
        output.append(sentence)
    print('prettify_result:')
    print(output)
    return output


def full_check(lan_tool, sentences, source_sentence):
    paraphrases_rule_checked = rule_sentence_check(sentences)
    if len(paraphrases_rule_checked) > 0:
        paraphrases_grammar_checked = grammar_sentence_check(lan_tool, paraphrases_rule_checked)
        if len(paraphrases_grammar_checked) > 0:
            paraphrases_similarity_checked = similarity_sentence_check(paraphrases_grammar_checked, source_sentence)
            print(paraphrases_similarity_checked)
            return paraphrases_similarity_checked


def inference_full(model, source_sentence):
    paraphraser = model['paraphrase']

    bert_model = model['bert']
    tokenizer = bert_model['tokenizer']
    model_bert = bert_model['model']
    bert = bert_model['bert']
    lan_tool = model['lan_tool']
    paraphrases = paraphraser.sample_paraphrase(source_sentence, sampling_temp=0.75, how_many=10)

    paraphrases_checked = full_check(lan_tool, paraphrases, source_sentence)
    lenth_needed = 0
    final_result = []
    if paraphrases_checked:
        if len(paraphrases_checked) >= 5:
            return drop_dup(sentence_prettify(paraphrases_checked[:5]))
        else:
            lenth_needed = 5 - len(paraphrases_checked)
            final_result.extend(sentence_prettify(paraphrases_checked))
            # can not generate suitable sentence by paraphrasing network, using bert...
            paraphrases = torch_synomy(tokenizer, model_bert, bert, source_sentence)
            print('paraphrases')
            print(paraphrases)
            if paraphrases:
                if len(paraphrases)>=2:
                    paraphrases.pop()
            print('paraphrasespop')
            print(paraphrases)
            bert_senteces_tobe_check = []
            for word_change_sentence_list in paraphrases:
                print(word_change_sentence_list)
                bert_senteces_tobe_check.extend(word_change_sentence_list)
            print('bert_senteces_tobe_check')
            print(bert_senteces_tobe_check)
            paraphrases_checked = full_check(lan_tool, bert_senteces_tobe_check, source_sentence)
            if paraphrases_checked:
                if len(paraphrases_checked) <= lenth_needed:
                    final_result.extend(sentence_prettify(paraphrases_checked))
                    return drop_dup(final_result)
                elif len(paraphrases_checked) > lenth_needed:
                    final_result.extend(sentence_prettify(paraphrases_checked[:lenth_needed]))
                    return drop_dup(final_result)
            else:
                if final_result:
                    return drop_dup(final_result)
                else:
                    return sentence_prettify([source_sentence])
    else:
        lenth_needed = 5
        paraphrases = torch_synomy(tokenizer, model_bert, bert, source_sentence)
        if paraphrases:
            if len(paraphrases) >= 2:
                paraphrases.pop()
        bert_senteces_tobe_check = []
        for word_change_sentence_list in paraphrases:
            bert_senteces_tobe_check.extend(word_change_sentence_list)
        paraphrases_checked = full_check(lan_tool, bert_senteces_tobe_check, source_sentence)
        if paraphrases_checked:
            if len(paraphrases_checked) <= lenth_needed:
                final_result.extend(sentence_prettify(paraphrases_checked))
                return drop_dup(final_result)
            elif len(paraphrases_checked) > lenth_needed:
                final_result.extend(sentence_prettify(paraphrases_checked[:lenth_needed]))
                return drop_dup(final_result)
        else:
            if final_result:
                return drop_dup(final_result)
            else:
                return sentence_prettify([source_sentence])

if __name__ == '__main__':
    paraphraser,model = model_init(0.5)  ## restore model
    source_sentence = 'I like apple'
    print(inference_full(model, source_sentence))
    print('next')
    print(inference_full(model,'I like swimming'))
    print(inference_full(model,'Speaking loudly in public places is considered impolite in both Chinese and Western countries.'))
    # input:model,source_sentence(example):'I like apple'
    # output:a list contains 10 sentences
    pass

