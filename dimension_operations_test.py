import torch
import numpy as np

import re
from nltk.corpus import stopwords

from torchtext.data.utils import get_tokenizer

tokenizer = get_tokenizer("basic_english")
stop_words = set(stopwords.words('english'))


class Dimension_operations:
    def __init__(self, emb_all, embeddings, idx2word, word2idx, EMBED_DIM, labels):
        self.emb_all = emb_all
        self.embeddings = embeddings
        self.idx2word = idx2word
        self.word2idx = word2idx
        self.EMBED_DIM = EMBED_DIM
        self.labels = labels

    def emb_all_feature_importance(self, model, N_HIDDEN_LAYERS, text: str):
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

        model.eval()  # set model in eval mode

        mean_emb = np.mean(np.array(emb_list), axis=0)
        data = torch.tensor(mean_emb).float().to(model.device)

        preds = model.forward(data)
        pred = torch.argmax(preds)
        label = self.labels[pred]
        print('Prediction for words:', text, '-', label, ':', preds.cpu().detach()[pred])
        in_weights = model.hidden_layers[0].weight  # [500,1000]
        # the bias is added to every distributed element. If we want to compute ReLU, it needs to be divided by the
        # number of summed elements.
        bias_div = len(torch.t(in_weights))  # 1000
        # activated_weights = torch.mul(data, in_weights)  # [500,1000]
        # activated_weights_len = len(activated_weights)
        #
        # activated_weights = torch.t(activated_weights)  # [1000,500]

        mul_weights = torch.mul(data, in_weights)  # [500,1000]
        mul_weights = torch.t(mul_weights)  # [1000,500]

        # sum bias to the list of activated weights. For each one of the 1000 dimensions, we sum the 500 el bias
        # to the 500 el weights. In the end we sum bias vec 1000 times,
        # therefore each bias element needs to be divided by 1000
        mul_weights = mul_weights + model.hidden_layers[0].bias.div(bias_div)

        # sum each column in a 500-dim vector to compute activation function
        activated_weights = torch.relu(torch.sum(mul_weights, 0))
        for idx, val in enumerate(activated_weights):
            if val == 0.0:
                mul_weights[:, idx] = 0.0

        # sum bias to the list of activated weights.

        # mul_weights = mul_weights + model.hidden_layers[0].bias.div(activated_weights_len)

        # activated_emb_to_out_weights = torch.matmul(activated_emb_to_out_weights, out_weights)

        for i in range(1, N_HIDDEN_LAYERS):
            next_layer = torch.t(model.hidden_layers[i].weight)
            mul_weights = torch.matmul(mul_weights, next_layer)  # it will be [1000,4] in the end
            bias = model.hidden_layers[i].bias.div(bias_div)  # it will be length = 4 in the end
            mul_weights = mul_weights + bias

            activated_weights = torch.relu(torch.sum(mul_weights, 0))

            for idx, val in enumerate(activated_weights):
                if val == 0.0:
                    mul_weights[:, idx] = 0.0

        # mul_weights = torch.softmax(mul_weights, dim=-1)

        # mul_weights has shape [num_in, num_out]
        dimension_label_value_list = []
        # get highest scored class for every dimension
        for i in range(self.EMBED_DIM):
            dim_values = mul_weights[i]
            label_ind = dim_values.argmax()  # get the class for which dimension i has the highest contribution
            dimension_label_value_list.append((i, label_ind, dim_values[label_ind]))


        dimension_label_value_list = sorted(dimension_label_value_list, key=lambda x: x[2], reverse=True)
        sums = {k: 0 for k in words_list}
        for i in range(len(dimension_label_value_list[:10])):
            dim = dimension_label_value_list[i][0]
            label = self.labels[dimension_label_value_list[i][1]]
            value = dimension_label_value_list[i][2]
            # top_emb = sorted(enumerate(self.embeddings), key=lambda x: x[1][dim], reverse=True)[:10]
            top_emb = sorted(enumerate(self.embeddings), key=lambda x: x[1][dim], reverse=True)[:10]
            # sorted_emb_list = sorted(enumerate(self.embeddings), key=lambda x: x[1][dim], reverse=True)
            num_to_find = 5
            right_window = 0
            left_window = 0
            # for enum_idx, (idx, emb) in enumerate(sorted_emb_list):
            #     if word_idx == idx:
            #         left_window = enum_idx - num_to_find
            #         right_window = enum_idx + num_to_find
            #         if left_window < 0:
            #             right_window += -left_window
            #             left_window = 0
            #         if right_window > len(sorted_emb_list):
            #             left_window = left_window - (right_window - len(sorted_emb_list))
            #             right_window = len(sorted_emb_list)
            #         # print(enum_idx, left_window, right_window)

            # emb_list = sorted_emb_list[left_window: right_window]
            # emb_list = [(self.idx2word[emb_idx]) for emb_idx, emb in emb_list]
            # for emb in emb_list:
            #     if emb not in self.embeddings:
            #         emb_list.remove(emb)
            print(
                'dimension %d labelled as %s with score %f. Top words in this dimension:' % (
                dim, label, value.item()))

            # exit(31)
            top_emb = [(self.idx2word[emb_idx]) for emb_idx, emb in top_emb]
            # print(
            #     'dimension %d labelled as %s with score %f. Top words in this dimension:' % (dim, label, value.item()))
            print(top_emb)
            for idx, emb in enumerate(emb_list):
                sums[words_list[idx]] += emb[dim] * value.item()
                print(words_list[idx], ' - ', emb[dim])
        sums = {k: v for k, v in sorted(sums.items(), key=lambda x: x[1], reverse=True)}
        print(sums)
        print('-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-')
        preds = torch.softmax(preds, dim=-1)
        vals, preds_idx_list = torch.sort(preds, descending=True)
        for pred_idx in preds_idx_list[:10]:
            sums = {k: 0 for k in words_list}
            label = self.labels[pred_idx]
            print('top dimensions for label', label, 'with score', preds[pred_idx].cpu().detach())
            influent_weights = torch.t(mul_weights)[pred_idx]
            sorted_weights, dim_idx = torch.sort(influent_weights, descending=True)

            # dim_idx contain indices of the most important weights for a certain class label.
            # dim_idx = 0 corresponds to the 1st dimension of the embeddings

            # sort embeddings from highest to smallest
            # according to the highest valued sums over the top dim_idx dimensions for a certain label
            # x[1] because we are enumerating, therefore x[1] are actual embeddings
            top_emb = sorted(enumerate(self.embeddings), key=lambda x: x[1][dim_idx[:5].cpu().numpy()].sum(),
                             reverse=True)[:20]
            top_emb = [(self.idx2word[emb_idx]) for emb_idx, emb in top_emb]
            # pprint(top_emb)

            for dim in dim_idx.cpu()[:100].cpu():
                for idx, emb in enumerate(emb_list):
                    sums[words_list[idx]] += emb[dim]  # * influent_weights[dim].cpu().detach()
            # influent_weights, dim_idx = torch.sort(influent_weights)
            # low_emb = sorted(enumerate(embeddings), key=lambda x: x[1][dim_idx[:10].cpu().numpy()].sum())[:10]
            # low_emb = [(idx2word[emb_idx]) for emb_idx, emb in low_emb]
            # pprint(low_emb)
            sums = {k: v for k, v in sorted(sums.items(), key=lambda x: x[1], reverse=True)}
            print(sums, '-', sum(sums.values()))
        #
        # split_words = words.replace('(','').replace(')', '').replace('.','').replace(',','').split(' ')
        # words_list = []
        # emb_list = []
        # for word in split_words:
        #     emb = self.emb_all.get(word)
        #     if emb is not None:
        #         emb_list.append(emb)
        #         words_list.append(word)
        #     else:
        #         print('no embedding for', word)
        # if not emb_list:
        #     return
        #
        # model.eval()
        # mean_emb = np.mean(np.array(emb_list), axis=0)
        # data = torch.tensor(mean_emb).float().to(model.device)
        # preds = model.forward(data)
        # pred = torch.argmax(preds)
        # label = self.labels[pred]
        # print('Prediction for words:', words, '-', label, ':', preds.cpu().detach()[pred])
        # in_weights = model.hidden_layers[0].weight  # [500,1000]
        # # the bias is added to every distributed element. If we want to compute ReLU, it needs to be divided by the
        # # number of summed elements.
        # bias_div = len(torch.t(in_weights))
        # # activated_weights = torch.mul(data, in_weights)  # [500,1000]
        # # activated_weights_len = len(activated_weights)
        # #
        # # activated_weights = torch.t(activated_weights)  # [1000,500]
        #
        # mul_weights = torch.mul(data, in_weights)  # [500,1000]
        #
        # mul_weights = torch.t(mul_weights)  # [1000,500]
        #
        # # sum bias to the list of activated weights. For each one of the 1000 dimensions, we sum the 500 el bias
        # # to the 500 el weights. In the end we sum bias 1000 times, therefore each bias element needs to be divided by 1000
        # mul_weights = mul_weights + model.hidden_layers[0].bias.div(bias_div)
        #
        # activated_weights = torch.relu(torch.sum(mul_weights, 0))
        # for idx, val in enumerate(activated_weights):
        #     if val == 0.0:
        #         mul_weights[:, idx] = 0.0
        #
        # # sum bias to the list of activated weights.
        #
        # # mul_weights = mul_weights + model.hidden_layers[0].bias.div(activated_weights_len)
        #
        # # activated_emb_to_out_weights = torch.matmul(activated_emb_to_out_weights, out_weights)
        #
        # for i in range(1, N_HIDDEN_LAYERS):
        #     next_layer = torch.t(model.hidden_layers[i].weight)
        #     mul_weights = torch.matmul(mul_weights, next_layer)  # it will be [1000,4] in the end
        #     bias = model.hidden_layers[i].bias.div(bias_div)  # it will be length = 4 in the end
        #     mul_weights = mul_weights + bias
        #
        #     activated_weights = torch.relu(torch.sum(mul_weights, 0))
        #
        #     for idx, val in enumerate(activated_weights):
        #         if val == 0.0:
        #             mul_weights[:, idx] = 0.0
        #
        # # mul_weights = torch.softmax(mul_weights, dim=-1)
        # dimension_label_value_list = []
        # for i in range(self.EMBED_DIM):
        #     dim_values = mul_weights[i]
        #     label_ind = dim_values.argmax()
        #     dimension_label_value_list.append((i, label_ind, dim_values[label_ind]))
        #
        # dimension_label_value_list = sorted(dimension_label_value_list, key=lambda x: x[2], reverse=True)
        # sums = {k: 0 for k in words_list}
        # for i in range(len(dimension_label_value_list[:10])):
        #     dim = dimension_label_value_list[i][0]
        #     label = self.labels[dimension_label_value_list[i][1]]
        #     value = dimension_label_value_list[i][2]
        #     # top_emb = sorted(enumerate(self.embeddings), key=lambda x: x[1][dim], reverse=True)[:10]
        #     top_emb = sorted(enumerate(self.embeddings), key=lambda x: x[1][dim], reverse=True)[:10]
        #     # sorted_emb_list = sorted(enumerate(self.embeddings), key=lambda x: x[1][dim], reverse=True)
        #     num_to_find = 5
        #     right_window = 0
        #     left_window = 0
        #     # for enum_idx, (idx, emb) in enumerate(sorted_emb_list):
        #     #     if word_idx == idx:
        #     #         left_window = enum_idx - num_to_find
        #     #         right_window = enum_idx + num_to_find
        #     #         if left_window < 0:
        #     #             right_window += -left_window
        #     #             left_window = 0
        #     #         if right_window > len(sorted_emb_list):
        #     #             left_window = left_window - (right_window - len(sorted_emb_list))
        #     #             right_window = len(sorted_emb_list)
        #     #         # print(enum_idx, left_window, right_window)
        #
        #     #emb_list = sorted_emb_list[left_window: right_window]
        #     # emb_list = [(self.idx2word[emb_idx]) for emb_idx, emb in emb_list]
        #     # for emb in emb_list:
        #     #     if emb not in self.embeddings:
        #     #         emb_list.remove(emb)
        #     print(
        #         'dimension %d labelled as %s with score %f. Top words in this dimension:' % (dim, label, value.item()))
        #
        #     # exit(31)
        #     top_emb = [(self.idx2word[emb_idx]) for emb_idx, emb in top_emb]
        #     # print(
        #     #     'dimension %d labelled as %s with score %f. Top words in this dimension:' % (dim, label, value.item()))
        #     print(top_emb)
        #     for idx, emb in enumerate(emb_list):
        #         sums[words_list[idx]] += emb[dim] * value.item()
        #         print(words_list[idx], ' - ', emb[dim])
        # sums = {k: v for k, v in sorted(sums.items(), key=lambda x: x[1], reverse=True)}
        # print(sums)
        # print('-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-')
        # preds = torch.softmax(preds, dim=-1)
        # vals, preds_idx_list = torch.sort(preds, descending=True)
        # for pred_idx in preds_idx_list[:10]:
        #     sums = {k: 0 for k in words_list}
        #     label = self.labels[pred_idx]
        #     print('top dimensions for label', label, 'with score', preds[pred_idx].cpu().detach())
        #     influent_weights = torch.t(mul_weights)[pred_idx]
        #     sorted_weights, dim_idx = torch.sort(influent_weights, descending=True)
        #
        #     # dim_idx contain indices of the most important weights for a certain class label.
        #     # dim_idx = 0 corresponds to the 1st dimension of the embeddings
        #
        #     # sort embeddings from highest to smallest
        #     # according to the highest valued sums over the top dim_idx dimensions for a certain label
        #     # x[1] because we are enumerating, therefore x[1] are actual embeddings
        #     top_emb = sorted(enumerate(self.embeddings), key=lambda x: x[1][dim_idx[:5].cpu().numpy()].sum(),
        #                      reverse=True)[:20]
        #     top_emb = [(self.idx2word[emb_idx]) for emb_idx, emb in top_emb]
        #     # pprint(top_emb)
        #
        #     for dim in dim_idx.cpu()[:100].cpu():
        #         for idx, emb in enumerate(emb_list):
        #             sums[words_list[idx]] += emb[dim]# * influent_weights[dim].cpu().detach()
        #     # influent_weights, dim_idx = torch.sort(influent_weights)
        #     # low_emb = sorted(enumerate(embeddings), key=lambda x: x[1][dim_idx[:10].cpu().numpy()].sum())[:10]
        #     # low_emb = [(idx2word[emb_idx]) for emb_idx, emb in low_emb]
        #     # pprint(low_emb)
        #     sums = {k: v for k, v in sorted(sums.items(), key=lambda x: x[1], reverse=True)}
        #     print(sums, '-', sum(sums.values()))



    def emb_all_feature_importance_pred_class(self, model, N_HIDDEN_LAYERS, text: str):
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

        model.eval()  # set model in eval mode

        mean_emb = np.mean(np.array(emb_list), axis=0)
        data = torch.tensor(mean_emb).float().to(model.device)

        preds = model.forward(data)
        pred = torch.argmax(preds)
        label = self.labels[pred]
        print('Prediction for words:', text, '-', label, ':', preds.cpu().detach()[pred])
        in_weights = model.hidden_layers[0].weight  # [500,1000]
        # the bias is added to every distributed element. If we want to compute ReLU, it needs to be divided by the
        # number of summed elements.
        bias_div = len(torch.t(in_weights))  # 1000
        # activated_weights = torch.mul(data, in_weights)  # [500,1000]
        # activated_weights_len = len(activated_weights)
        #
        # activated_weights = torch.t(activated_weights)  # [1000,500]

        mul_weights = torch.mul(data, in_weights)  # [500,1000]
        mul_weights = torch.t(mul_weights)  # [1000,500]

        # sum bias to the list of activated weights. For each one of the 1000 dimensions, we sum the 500 el bias
        # to the 500 el weights. In the end we sum bias vec 1000 times,
        # therefore each bias element needs to be divided by 1000
        mul_weights = mul_weights + model.hidden_layers[0].bias.div(bias_div)

        # sum each column in a 500-dim vector to compute activation function
        activated_weights = torch.relu(torch.sum(mul_weights, 0))
        for idx, val in enumerate(activated_weights):
            if val == 0.0:
                mul_weights[:, idx] = 0.0

        # sum bias to the list of activated weights.

        # mul_weights = mul_weights + model.hidden_layers[0].bias.div(activated_weights_len)

        # activated_emb_to_out_weights = torch.matmul(activated_emb_to_out_weights, out_weights)

        for i in range(1, N_HIDDEN_LAYERS):
            next_layer = torch.t(model.hidden_layers[i].weight)
            mul_weights = torch.matmul(mul_weights, next_layer)  # it will be [1000,4] in the end
            bias = model.hidden_layers[i].bias.div(bias_div)  # it will be length = 4 in the end
            mul_weights = mul_weights + bias

            activated_weights = torch.relu(torch.sum(mul_weights, 0))

            for idx, val in enumerate(activated_weights):
                if val == 0.0:
                    mul_weights[:, idx] = 0.0

        # mul_weights = torch.softmax(mul_weights, dim=-1)

        # mul_weights has shape [num_in, num_out]
        dimension_label_value_list = []
        # get highest scored class for every dimension
        for i in range(self.EMBED_DIM):
            dim_values = mul_weights[i]
            label_ind = pred
            # save pairs (idx, pred_idx, value for that class in dimension i)
            dimension_label_value_list.append((i, label_ind, dim_values[label_ind]))


        dimension_label_value_list = sorted(dimension_label_value_list, key=lambda x: x[2], reverse=True)
        sums = {k: 0 for k in words_list}
        for i in range(len(dimension_label_value_list[:10])):
            dim = dimension_label_value_list[i][0]
            label = self.labels[dimension_label_value_list[i][1]]
            value = dimension_label_value_list[i][2]
            # top_emb = sorted(enumerate(self.embeddings), key=lambda x: x[1][dim], reverse=True)[:10]
            top_emb = sorted(enumerate(self.embeddings), key=lambda x: x[1][dim], reverse=True)[:10]
            # sorted_emb_list = sorted(enumerate(self.embeddings), key=lambda x: x[1][dim], reverse=True)
            num_to_find = 5
            right_window = 0
            left_window = 0
            # for enum_idx, (idx, emb) in enumerate(sorted_emb_list):
            #     if word_idx == idx:
            #         left_window = enum_idx - num_to_find
            #         right_window = enum_idx + num_to_find
            #         if left_window < 0:
            #             right_window += -left_window
            #             left_window = 0
            #         if right_window > len(sorted_emb_list):
            #             left_window = left_window - (right_window - len(sorted_emb_list))
            #             right_window = len(sorted_emb_list)
            #         # print(enum_idx, left_window, right_window)

            # emb_list = sorted_emb_list[left_window: right_window]
            # emb_list = [(self.idx2word[emb_idx]) for emb_idx, emb in emb_list]
            # for emb in emb_list:
            #     if emb not in self.embeddings:
            #         emb_list.remove(emb)
            print(
                'dimension %d labelled as %s with score %f. Top words in this dimension:' % (
                dim, label, value.item()))

            # exit(31)
            top_emb = [(self.idx2word[emb_idx]) for emb_idx, emb in top_emb]
            # print(
            #     'dimension %d labelled as %s with score %f. Top words in this dimension:' % (dim, label, value.item()))
            print(top_emb)
            for idx, emb in enumerate(emb_list):
                sums[words_list[idx]] += emb[dim] * value.item()
                print(words_list[idx], ' - ', emb[dim])
        sums = {k: v for k, v in sorted(sums.items(), key=lambda x: x[1], reverse=True)}
        print(sums)
        print('-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-')
        preds = torch.softmax(preds, dim=-1)
        vals, preds_idx_list = torch.sort(preds, descending=True)
        for pred_idx in preds_idx_list[:10]:
            sums = {k: 0 for k in words_list}
            label = self.labels[pred_idx]
            print('top dimensions for label', label, 'with score', preds[pred_idx].cpu().detach())
            influent_weights = torch.t(mul_weights)[pred_idx]
            sorted_weights, dim_idx = torch.sort(influent_weights, descending=True)

            # dim_idx contain indices of the most important weights for a certain class label.
            # dim_idx = 0 corresponds to the 1st dimension of the embeddings

            # sort embeddings from highest to smallest
            # according to the highest valued sums over the top dim_idx dimensions for a certain label
            # x[1] because we are enumerating, therefore x[1] are actual embeddings
            top_emb = sorted(enumerate(self.embeddings), key=lambda x: x[1][dim_idx[:5].cpu().numpy()].sum(),
                             reverse=True)[:20]
            top_emb = [(self.idx2word[emb_idx]) for emb_idx, emb in top_emb]
            # pprint(top_emb)

            for dim in dim_idx.cpu()[:100].cpu():
                for idx, emb in enumerate(emb_list):
                    sums[words_list[idx]] += emb[dim]  # * influent_weights[dim].cpu().detach()
            # influent_weights, dim_idx = torch.sort(influent_weights)
            # low_emb = sorted(enumerate(embeddings), key=lambda x: x[1][dim_idx[:10].cpu().numpy()].sum())[:10]
            # low_emb = [(idx2word[emb_idx]) for emb_idx, emb in low_emb]
            # pprint(low_emb)
            sums = {k: v for k, v in sorted(sums.items(), key=lambda x: x[1], reverse=True)}
            print(sums, '-', sum(sums.values()))
        #
        # split_words = words.replace('(','').replace(')', '').replace('.','').replace(',','').split(' ')
        # words_list = []
        # emb_list = []
        # for word in split_words:
        #     emb = self.emb_all.get(word)
        #     if emb is not None:
        #         emb_list.append(emb)
        #         words_list.append(word)
        #     else:
        #         print('no embedding for', word)
        # if not emb_list:
        #     return
        #
        # model.eval()
        # mean_emb = np.mean(np.array(emb_list), axis=0)
        # data = torch.tensor(mean_emb).float().to(model.device)
        # preds = model.forward(data)
        # pred = torch.argmax(preds)
        # label = self.labels[pred]
        # print('Prediction for words:', words, '-', label, ':', preds.cpu().detach()[pred])
        # in_weights = model.hidden_layers[0].weight  # [500,1000]
        # # the bias is added to every distributed element. If we want to compute ReLU, it needs to be divided by the
        # # number of summed elements.
        # bias_div = len(torch.t(in_weights))
        # # activated_weights = torch.mul(data, in_weights)  # [500,1000]
        # # activated_weights_len = len(activated_weights)
        # #
        # # activated_weights = torch.t(activated_weights)  # [1000,500]
        #
        # mul_weights = torch.mul(data, in_weights)  # [500,1000]
        #
        # mul_weights = torch.t(mul_weights)  # [1000,500]
        #
        # # sum bias to the list of activated weights. For each one of the 1000 dimensions, we sum the 500 el bias
        # # to the 500 el weights. In the end we sum bias 1000 times, therefore each bias element needs to be divided by 1000
        # mul_weights = mul_weights + model.hidden_layers[0].bias.div(bias_div)
        #
        # activated_weights = torch.relu(torch.sum(mul_weights, 0))
        # for idx, val in enumerate(activated_weights):
        #     if val == 0.0:
        #         mul_weights[:, idx] = 0.0
        #
        # # sum bias to the list of activated weights.
        #
        # # mul_weights = mul_weights + model.hidden_layers[0].bias.div(activated_weights_len)
        #
        # # activated_emb_to_out_weights = torch.matmul(activated_emb_to_out_weights, out_weights)
        #
        # for i in range(1, N_HIDDEN_LAYERS):
        #     next_layer = torch.t(model.hidden_layers[i].weight)
        #     mul_weights = torch.matmul(mul_weights, next_layer)  # it will be [1000,4] in the end
        #     bias = model.hidden_layers[i].bias.div(bias_div)  # it will be length = 4 in the end
        #     mul_weights = mul_weights + bias
        #
        #     activated_weights = torch.relu(torch.sum(mul_weights, 0))
        #
        #     for idx, val in enumerate(activated_weights):
        #         if val == 0.0:
        #             mul_weights[:, idx] = 0.0
        #
        # # mul_weights = torch.softmax(mul_weights, dim=-1)
        # dimension_label_value_list = []
        # for i in range(self.EMBED_DIM):
        #     dim_values = mul_weights[i]
        #     label_ind = dim_values.argmax()
        #     dimension_label_value_list.append((i, label_ind, dim_values[label_ind]))
        #
        # dimension_label_value_list = sorted(dimension_label_value_list, key=lambda x: x[2], reverse=True)
        # sums = {k: 0 for k in words_list}
        # for i in range(len(dimension_label_value_list[:10])):
        #     dim = dimension_label_value_list[i][0]
        #     label = self.labels[dimension_label_value_list[i][1]]
        #     value = dimension_label_value_list[i][2]
        #     # top_emb = sorted(enumerate(self.embeddings), key=lambda x: x[1][dim], reverse=True)[:10]
        #     top_emb = sorted(enumerate(self.embeddings), key=lambda x: x[1][dim], reverse=True)[:10]
        #     # sorted_emb_list = sorted(enumerate(self.embeddings), key=lambda x: x[1][dim], reverse=True)
        #     num_to_find = 5
        #     right_window = 0
        #     left_window = 0
        #     # for enum_idx, (idx, emb) in enumerate(sorted_emb_list):
        #     #     if word_idx == idx:
        #     #         left_window = enum_idx - num_to_find
        #     #         right_window = enum_idx + num_to_find
        #     #         if left_window < 0:
        #     #             right_window += -left_window
        #     #             left_window = 0
        #     #         if right_window > len(sorted_emb_list):
        #     #             left_window = left_window - (right_window - len(sorted_emb_list))
        #     #             right_window = len(sorted_emb_list)
        #     #         # print(enum_idx, left_window, right_window)
        #
        #     #emb_list = sorted_emb_list[left_window: right_window]
        #     # emb_list = [(self.idx2word[emb_idx]) for emb_idx, emb in emb_list]
        #     # for emb in emb_list:
        #     #     if emb not in self.embeddings:
        #     #         emb_list.remove(emb)
        #     print(
        #         'dimension %d labelled as %s with score %f. Top words in this dimension:' % (dim, label, value.item()))
        #
        #     # exit(31)
        #     top_emb = [(self.idx2word[emb_idx]) for emb_idx, emb in top_emb]
        #     # print(
        #     #     'dimension %d labelled as %s with score %f. Top words in this dimension:' % (dim, label, value.item()))
        #     print(top_emb)
        #     for idx, emb in enumerate(emb_list):
        #         sums[words_list[idx]] += emb[dim] * value.item()
        #         print(words_list[idx], ' - ', emb[dim])
        # sums = {k: v for k, v in sorted(sums.items(), key=lambda x: x[1], reverse=True)}
        # print(sums)
        # print('-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-')
        # preds = torch.softmax(preds, dim=-1)
        # vals, preds_idx_list = torch.sort(preds, descending=True)
        # for pred_idx in preds_idx_list[:10]:
        #     sums = {k: 0 for k in words_list}
        #     label = self.labels[pred_idx]
        #     print('top dimensions for label', label, 'with score', preds[pred_idx].cpu().detach())
        #     influent_weights = torch.t(mul_weights)[pred_idx]
        #     sorted_weights, dim_idx = torch.sort(influent_weights, descending=True)
        #
        #     # dim_idx contain indices of the most important weights for a certain class label.
        #     # dim_idx = 0 corresponds to the 1st dimension of the embeddings
        #
        #     # sort embeddings from highest to smallest
        #     # according to the highest valued sums over the top dim_idx dimensions for a certain label
        #     # x[1] because we are enumerating, therefore x[1] are actual embeddings
        #     top_emb = sorted(enumerate(self.embeddings), key=lambda x: x[1][dim_idx[:5].cpu().numpy()].sum(),
        #                      reverse=True)[:20]
        #     top_emb = [(self.idx2word[emb_idx]) for emb_idx, emb in top_emb]
        #     # pprint(top_emb)
        #
        #     for dim in dim_idx.cpu()[:100].cpu():
        #         for idx, emb in enumerate(emb_list):
        #             sums[words_list[idx]] += emb[dim]# * influent_weights[dim].cpu().detach()
        #     # influent_weights, dim_idx = torch.sort(influent_weights)
        #     # low_emb = sorted(enumerate(embeddings), key=lambda x: x[1][dim_idx[:10].cpu().numpy()].sum())[:10]
        #     # low_emb = [(idx2word[emb_idx]) for emb_idx, emb in low_emb]
        #     # pprint(low_emb)
        #     sums = {k: v for k, v in sorted(sums.items(), key=lambda x: x[1], reverse=True)}
        #     print(sums, '-', sum(sums.values()))


    def get_word_ranking(self, model, N_HIDDEN_LAYERS, text: str):
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

        model.eval()  # set model in eval mode

        mean_emb = np.mean(np.array(emb_list), axis=0)
        data = torch.tensor(mean_emb).float().to(model.device)

        preds = model.forward(data)
        pred = torch.argmax(preds)
        label = self.labels[pred]
        print('Prediction for text:', text, '\n', label, ':')  #, ':', preds.cpu().detach()[pred])
        in_weights = model.hidden_layers[0].weight  # [500,1000]
        # the bias is added to every distributed element. If we want to compute ReLU, it needs to be divided by the
        # number of summed elements.
        bias_div = len(torch.t(in_weights))  # 1000
        # activated_weights = torch.mul(data, in_weights)  # [500,1000]
        # activated_weights_len = len(activated_weights)
        #
        # activated_weights = torch.t(activated_weights)  # [1000,500]

        mul_weights = torch.mul(data, in_weights)  # [500,1000]
        mul_weights = torch.t(mul_weights)  # [1000,500]

        # sum bias to the list of activated weights. For each one of the 1000 dimensions, we sum the 500 el bias
        # to the 500 el weights. In the end we sum bias vec 1000 times,
        # therefore each bias element needs to be divided by 1000
        mul_weights = mul_weights + model.hidden_layers[0].bias.div(bias_div)

        # sum each column in a 500-dim vector to compute activation function
        activated_weights = torch.relu(torch.sum(mul_weights, 0))
        for idx, val in enumerate(activated_weights):
            if val == 0.0:
                mul_weights[:, idx] = 0.0

        # sum bias to the list of activated weights.

        # mul_weights = mul_weights + model.hidden_layers[0].bias.div(activated_weights_len)

        # activated_emb_to_out_weights = torch.matmul(activated_emb_to_out_weights, out_weights)

        for i in range(1, N_HIDDEN_LAYERS):
            next_layer = torch.t(model.hidden_layers[i].weight)
            mul_weights = torch.matmul(mul_weights, next_layer)  # it will be [1000,4] in the end
            bias = model.hidden_layers[i].bias.div(bias_div)  # it will be length = 4 in the end
            mul_weights = mul_weights + bias

            activated_weights = torch.relu(torch.sum(mul_weights, 0))

            for idx, val in enumerate(activated_weights):
                if val == 0.0:
                    mul_weights[:, idx] = 0.0

        # mul_weights = torch.softmax(mul_weights, dim=-1)

        # mul_weights has shape [num_in, num_out]
        dimension_label_value_list = []
        # get highest scored class for every dimension
        for i in range(self.EMBED_DIM):
            dim_values = mul_weights[i]
            label_ind = pred
            # save pairs (idx, pred_idx, value for that class in dimension i)
            dimension_label_value_list.append((i, label_ind, dim_values[label_ind]))

        dimension_label_value_list = sorted(dimension_label_value_list, key=lambda x: x[2], reverse=True)
        sums = {k: 0 for k in words_list}
        for i in range(len(dimension_label_value_list)):# [:10])):
            dim = dimension_label_value_list[i][0]
            # label = self.labels[dimension_label_value_list[i][1]]
            value = dimension_label_value_list[i][2]
            top_emb = sorted(enumerate(self.embeddings), key=lambda x: x[1][dim], reverse=True)[:10]
            for idx, emb in enumerate(emb_list):
                sums[words_list[idx]] += emb[dim] * value.item()
        sums = {k: v for k, v in sorted(sums.items(), key=lambda x: x[1], reverse=True)}
        print(sums)
        return sums, label

    def get_important_dimensions_for_predicted_class(self, model, N_HIDDEN_LAYERS, text: str):
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

        model.eval()  # set model in eval mode

        mean_emb = np.mean(np.array(emb_list), axis=0)
        data = torch.tensor(mean_emb).float().to(model.device)

        preds = model.forward(data)
        pred = torch.argmax(preds)
        label = self.labels[pred]
        print('Prediction for text:', text, '\n', label, ':')  #, ':', preds.cpu().detach()[pred])
        in_weights = model.hidden_layers[0].weight  # [500,1000]
        # the bias is added to every distributed element. If we want to compute ReLU, it needs to be divided by the
        # number of summed elements.
        bias_div = len(torch.t(in_weights))  # 1000
        # activated_weights = torch.mul(data, in_weights)  # [500,1000]
        # activated_weights_len = len(activated_weights)
        #
        # activated_weights = torch.t(activated_weights)  # [1000,500]

        mul_weights = torch.mul(data, in_weights)  # [500,1000]
        mul_weights = torch.t(mul_weights)  # [1000,500]

        # sum bias to the list of activated weights. For each one of the 1000 dimensions, we sum the 500 el bias
        # to the 500 el weights. In the end we sum bias vec 1000 times,
        # therefore each bias element needs to be divided by 1000
        mul_weights = mul_weights + model.hidden_layers[0].bias.div(bias_div)

        # sum each column in a 500-dim vector to compute activation function
        activated_weights = torch.relu(torch.sum(mul_weights, 0))
        for idx, val in enumerate(activated_weights):
            if val == 0.0:
                mul_weights[:, idx] = 0.0

        # sum bias to the list of activated weights.

        # mul_weights = mul_weights + model.hidden_layers[0].bias.div(activated_weights_len)

        # activated_emb_to_out_weights = torch.matmul(activated_emb_to_out_weights, out_weights)

        for i in range(1, N_HIDDEN_LAYERS):
            next_layer = torch.t(model.hidden_layers[i].weight)
            mul_weights = torch.matmul(mul_weights, next_layer)  # it will be [1000,4] in the end
            bias = model.hidden_layers[i].bias.div(bias_div)  # it will be length = 4 in the end
            mul_weights = mul_weights + bias

            activated_weights = torch.relu(torch.sum(mul_weights, 0))

            for idx, val in enumerate(activated_weights):
                if val == 0.0:
                    mul_weights[:, idx] = 0.0


        # mul_weights = torch.softmax(mul_weights, dim=-1)
        dimension_label_value_list = []
        for i in range(self.EMBED_DIM):
            dim_values = mul_weights[i]
            label_ind = pred  # we get input dimension scores for the predicted class
            # label_ind = dim_values.argmax()  # we get the class label for which each input dim contributes the most
            dimension_label_value_list.append((i, label_ind, dim_values[label_ind]))

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
        return top_emb_list, label
