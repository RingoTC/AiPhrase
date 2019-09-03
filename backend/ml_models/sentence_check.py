# -*- coding: UTF-8 -*-

import language_check

lan_tool = language_check.LanguageTool("en-US")


def sentence_correction(sentence):
    '''
    修改存在语法错误的英文句子，
    输入：句子
    输出：错误个数 + 错误位置 + 修改建议 + 改正后句子，
    {'error_no': number,
    'error_loc': [[x, y], [x, y]],
    'error_crec': [' ', ' '],
    'sen_crec': ' '}
    '''
    matches = lan_tool.check(sentence)
    error_no = len(matches)
    errors = []
    for i in range(error_no):
        word = sentence[matches[i].fromx: matches[i].fromx + matches[i].errorlength]
        word_replacements = matches[i].replacements
        if len(word_replacements) <= 3:
            word_replacements = word_replacements
        else:
            word_replacements = word_replacements[:3]
        word_msg = matches[i].msg
        errors.append([word, word_replacements, word_msg])
    return error_no, errors


#     error_loc = []
#     error_crec = []
#     for i in range(error_no):
#         error_loc.append([matches[i].fromx, matches[i].fromy,
#                           matches[i].errorlength])
#         error_crec.append([matches[i].msg, matches[i].replacements])
#     newSentence = language_check.correct(sentence, matches)
#     return {
#         'error_no': error_no,
#         'error_loc': error_loc,
#         'error_crec': error_crec,
#         'sen_crec': newSentence,
#     }

if __name__ == '__main__':
    sentence = 'Paraphrasing is extremely important, so it shoulds be learned . '
    print(sentence)
    print(sentence_correction(sentence))
