import os
import re
import string

import numpy as np

from nltk.corpus import stopwords

from torchtext.data.utils import get_tokenizer

tokenizer = get_tokenizer("basic_english")
stop_words = set(stopwords.words('english'))


emb_all = dict()


def parse_dataset(dataset_path):
    dataset = []
    labels = []
    for doc in os.listdir(dataset_path):
        print(doc)
        label = doc.replace('.txt', '')
        labels.append(label)

        doc_path = os.path.join(dataset_path, doc)
        count = 0
        with open(doc_path) as f:
            for line in f:
                line = line.lower()
                # line = line.replace(' is ', '')
                tab_idx = line.find('\t')
                # used to remove all small words and "see other word"
                #if len(line.split(' ')) < 4:
                #    continue
                line = re.sub('[^A-Za-z0-9]+', ' ', line)
                words_list = tokenizer(line)
                words_list = [w for w in words_list if w not in stop_words and w not in string.punctuation]
                dataset.append((words_list, label))
                count += 1
                if count == 2000:
                    break
    print('done parsing')
    return dataset, labels


def load_embeddings(filename):
    is_fasttext = '.vec' in filename
    with open(filename) as f:
        if is_fasttext:
            f.readline()
        for c, line in enumerate(f):
            line = line.split()
            word = line[0]
            if is_fasttext:
                if word in string.punctuation:  # personal addition. We want to ignore punctation
                    continue
                # if len(word) < 5:
                #    continue
            emb = np.array(line[1:], dtype=np.float)
            emb_all[word] = emb
            if is_fasttext:
                if c == 500000:
                    break
    return emb_all


def create_train_dataset(dataset, labels):
    train_data = []
    nones = 0
    tot_docs = 0
    for pair in dataset:
        tot_docs+=1
        emb_list = []
        words_list = pair[0]
        label = labels.index(pair[1])
        for word in words_list:
            w_emb = emb_all.get(word)
            if w_emb is not None:
                emb_list.append(w_emb)
        if not emb_list:
            nones += 1
            continue
        mean_emb = np.mean(np.array(emb_list), axis=0)
        train_data.append((mean_emb, label))
    print(tot_docs, nones)
    return train_data# , vocabulary, word2idx, idx2word


def generate_batches(data, batch_size):
    batches = []
    batch = []
    for i, pair in enumerate(data):
        emb = pair[0]
        label = pair[1]
        batch.append((emb, label))
        if (i+1) % batch_size == 0:
            batches.append(batch)
            batch = []
        if i + batch_size > len(data):
            print(i)
            break
    return batches
