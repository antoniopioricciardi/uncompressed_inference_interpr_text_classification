import features_contrib as feat_contr
import re
import numpy as np
from nltk.corpus import stopwords

from torchtext.data.utils import get_tokenizer

tokenizer = get_tokenizer("basic_english")
stop_words = set(stopwords.words('english'))


# TODO: Aggiungi altri metodi
class DimensionOperations:
    def __init__(self, emb_all, embeddings, idx2word, word2idx, EMBED_DIM, labels):
        self.emb_all = emb_all
        self.embeddings = embeddings
        self.idx2word = idx2word
        self.word2idx = word2idx
        self.EMBED_DIM = EMBED_DIM
        self.labels = labels

    def get_mean_embedding(self, text: str):
        text = text.lower()
        text = re.sub('[^A-Za-z0-9]+', ' ', text)
        split_words = tokenizer(text)

        words_list = []
        emb_list = []
        for word in split_words:
            emb = self.emb_all.get(word)
            if emb is not None:
                emb_list.append(emb)
                words_list.append(word)
            else:
                print('no embedding for', word)
        if not emb_list:
            return
        mean_emb = np.mean(np.array(emb_list), axis=0)

        return mean_emb, emb_list, words_list

    def get_important_dimensions_for_selected_class(self, mul_weights, class_idx):
        # mul_weights = torch.softmax(mul_weights, dim=-1)
        dimension_label_value_list = []
        for i in range(self.EMBED_DIM):
            dim_values = mul_weights[i]
            dimension_label_value_list.append((i, class_idx, dim_values[class_idx]))

        dimension_label_value_list = sorted(dimension_label_value_list, key=lambda x: x[2], reverse=True)
        top_emb_list = []
        for i in range(len(dimension_label_value_list[:5])):
            dim = dimension_label_value_list[i][0]
            label = self.labels[dimension_label_value_list[i][1]]
            value = dimension_label_value_list[i][2]
            # top_emb = sorted(enumerate(self.embeddings), key=lambda x: x[1][dim], reverse=True)[:10]
            top_emb = sorted(enumerate(self.embeddings), key=lambda x: x[1][dim], reverse=True)[:10]
            # sorted_emb_list = sorted(enumerate(self.embeddings), key=lambda x: x[1][dim], reverse=True)

            print(
                'dimension %d labelled as %s with score %f. Top words in this dimension:' % (dim, label, value.item()))

            # exit(31)
            top_emb = [(self.idx2word[emb_idx]) for emb_idx, emb in top_emb]
            # print(
            #     'dimension %d labelled as %s with score %f. Top words in this dimension:' % (dim, label, value.item()))
            print(top_emb)
            top_emb_list.append((top_emb, dim, value.item()))
        return top_emb_list

    def get_word_ranking_for_selected_class(self, emb_list, words_list, mul_weights, class_idx):
        label = self.labels[class_idx]

        # mul_weights has shape [num_in, num_out]
        dimension_label_value_list = []
        # get highest scored class for every dimension
        for i in range(self.EMBED_DIM):
            dim_values = mul_weights[i]
            label_ind = class_idx
            # save pairs (idx, pred_idx, value for that class in dimension i)
            dimension_label_value_list.append((i, label_ind, dim_values[label_ind]))

        dimension_label_value_list = sorted(dimension_label_value_list, key=lambda x: x[2], reverse=True)
        sums = {k: 0 for k in words_list}
        for i in range(len(dimension_label_value_list)):  # [:10])):
            dim = dimension_label_value_list[i][0]
            # label = self.labels[dimension_label_value_list[i][1]]
            value = dimension_label_value_list[i][2]
            top_emb = sorted(enumerate(self.embeddings), key=lambda x: x[1][dim], reverse=True)[:10]
            for idx, emb in enumerate(emb_list):
                sums[words_list[idx]] += emb[dim] * value.item()
        sums = {k: v for k, v in sorted(sums.items(), key=lambda x: x[1], reverse=True)}
        print(sums)
        return sums, label