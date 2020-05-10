import os
import re
from opencc import OpenCC
import nltk
from nltk.tokenize import word_tokenize
import jieba
from zhon.hanzi import punctuation
jieba.enable_paddle()

cc = OpenCC('s2t')
jieba.set_dictionary('dict.txt.small')
en, cn = [], []
with open('cmn.txt', 'r') as f:
    for line in f:
        sentence = re.split('\t', line)
        sentence = list(filter(None, sentence))
        en_sentence = ''
        for word in word_tokenize(sentence[0]):
            en_sentence += word.lower() + ' '
        en.append(en_sentence)
        cn_sentence = ''
        #for word in list(jieba.cut(sentence[1], use_paddle=True)):
        for word in list(jieba.cut(sentence[1])):
            word = re.sub(r'[ \n\t\r]', '', word)
            if word == '':
                continue
            cn_sentence += cc.convert(word) + ' '
        cn.append(cn_sentence)

with open('en.txt', 'w') as f:
    for sentence in en:
        print (sentence, file=f)

with open('cn.txt', 'w') as f:
    for sentence in cn:
        print (sentence, file=f)

print ('Tokenizer Done!')
