import os
import re
import json
import random
from zhon.hanzi import punctuation

# English
word2int_en, int2word_en = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}, {0: '<PAD>', 1: '<BOS>', 2: '<EOS>', 3: '<UNK>'}
Index = 4
with open('en_vocab.txt', 'r') as f:
    for line in f:
        line = re.split(' ', line)
        line = list(filter(None, line))
        if int(line[1]) < 3:
            continue
        word2int_en[line[0]] = Index
        int2word_en[Index] = line[0]
        Index += 1

with open("../word2int_en.json", "w") as f:
    json.dump(word2int_en, f, ensure_ascii=False)

with open("../int2word_en.json", "w") as f:
    json.dump(int2word_en, f, ensure_ascii=False)


# Chinese
words = {}
with open('cn.txt', 'r') as f:
    for line in f:
        line = re.split('[ \n\t\r ]', line)
        line = list(filter(None, line))
        for word in line:
            words[word] = words.get(word, 0) + 1

words = sorted(words.items(), key=lambda d: d[1], reverse=True)
words = [word for word, count in words if count >= 3]
words = ['<PAD>', '<BOS>', '<EOS>', '<UNK>'] + words

word2int_cn, int2word_cn = {}, {}
for Index, word in enumerate(words):
    word2int_cn[word] = Index
    int2word_cn[Index] = word

with open("../word2int_cn.json", "w") as f:
    json.dump(word2int_cn, f, ensure_ascii=False)

with open("../int2word_cn.json", "w") as f:
    json.dump(int2word_cn, f, ensure_ascii=False)


en_sentences = []
with open('en_refine.txt', 'r') as f:
    for line in f.readlines():
        line = re.split('\n', line)
        line = list(filter(None, line))
        en_sentences.append(line[0])

cn_sentences = []
with open('cn.txt', 'r') as f:
    for line in f:
        line = re.split('\n', line)
        line = list(filter(None, line))
        cn_sentences.append(line[0])

sentences = []
for en_sentence, cn_sentence in zip(en_sentences, cn_sentences):
    tokens = re.split('[ \n\t\r  ]', en_sentence)
    tokens = list(filter(None, tokens))
    count = 0
    for token in tokens:
        Index = word2int_en.get(token, 3)
        if Index == 3:
            count += 1
    if count >= 3:
        continue

    tokens = re.split('[ \n\t\r  ]', cn_sentence)
    tokens = list(filter(None, tokens))
    count = 0
    for token in tokens:
        Index = word2int_cn.get(token, 3)
        if Index == 3:
            count += 1
    if count >= 3:
        continue

    sentences.append(en_sentence + '\t' + cn_sentence)


print (len(sentences))
sentences = list(set(sentences))
print (len(sentences))
random.seed(2020)
random.shuffle(sentences)

with open('../training.txt', 'w') as f:
    for sentence in sentences[:18000]:
        print (sentence, file=f)

with open('../validation.txt', 'w') as f:
    for sentence in sentences[18000:18500]:
        print (sentence, file=f)

with open('../testing.txt', 'w') as f:
    for sentence in sentences[18500:]:
        print (sentence, file=f)

print ('Build Dataset Down!')
