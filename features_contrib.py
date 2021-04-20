import torch


class FeaturesContributions:
    def __init__(self, model, n_hidden_layers):
        """

        :param model: trained model, needed to retrieve its trained parameters. It takes a n_x dimensional input vector and
        returns a n_y dimensional output vector
        :param n_hidden_layers: we need this to iterate the parameters over all layers
        """
        self.model = model
        self.n_hidden_layers = n_hidden_layers

    def get_features_contribution_matrix(self,  x):
        """
        Method for obtaining features contribution scores.
        :param x: n_x dimensional input data
        :return: a n_x X n_y matrix of scores
        """

        ''' #### THE ACTIVATION FUNCTIONS NEED TO BE MODIFIED ACCORDING TO THE FORWARD METHOD #### '''

        self.model.eval()  # set model in eval mode

        data = torch.tensor(x).float().to(self.model.device)

        # preds = model.forward(data)
        # pred = torch.argmax(preds)
        # label = self.labels[pred]
        in_weights = self.model.hidden_layers[0].weight  # [n_h1,n_x]
        # the bias is added to every distributed element. If we want to compute ReLU, it needs to be divided by the
        # number of summed elements.
        bias_div = len(torch.t(in_weights))  # n_x
        # activated_weights = torch.mul(data, in_weights)  # [500,1000]
        # activated_weights_len = len(activated_weights)
        #
        # activated_weights = torch.t(activated_weights)  # [1000,500]

        mul_weights = torch.mul(data, in_weights)  # [n_h1,n_x]
        mul_weights = torch.t(mul_weights)  # [n_x,n_h1]

        # sum bias to the list of activated weights. For each one of the 1000 dimensions, we sum the 500 el bias
        # to the 500 el weights. In the end we sum bias vec 1000 times,
        # therefore each bias element needs to be divided by 1000
        mul_weights = mul_weights + self.model.hidden_layers[0].bias.div(bias_div)

        # sum each column in a n_h1-dim vector to compute activation function
        activated_weights = torch.relu(torch.sum(mul_weights, 0))
        for idx, val in enumerate(activated_weights):
            if val == 0.0:
                mul_weights[:, idx] = 0.0

        # sum bias to the list of activated weights.

        # mul_weights = mul_weights + model.hidden_layers[0].bias.div(activated_weights_len)

        # activated_emb_to_out_weights = torch.matmul(activated_emb_to_out_weights, out_weights)

        for i in range(1, self.n_hidden_layers):
            next_layer = torch.t(self.model.hidden_layers[i].weight)
            mul_weights = torch.matmul(mul_weights, next_layer)  # [n_x,n_hi] - however it will be [1000,n_out] in the end
            bias = self.model.hidden_layers[i].bias.div(bias_div)  # n_hi - however it will be length = n_out in the end
            mul_weights = mul_weights + bias

            activated_weights = torch.relu(torch.sum(mul_weights, 0))

            for idx, val in enumerate(activated_weights):
                if val == 0.0:
                    mul_weights[:, idx] = 0.0

        return mul_weights

# Throwable
# def other():
#     # mul_weights = torch.softmax(mul_weights, dim=-1)
#     dimension_label_value_list = []
#     for i in range(self.EMBED_DIM):
#         dim_values = mul_weights[i]
#         label_ind = pred   # we get input dimension scores for the predicted class
#         # label_ind = dim_values.argmax()  # we get the class label for which each input dim contributes the most
#         dimension_label_value_list.append((i, label_ind, dim_values[label_ind]))
#
#     dimension_label_value_list = sorted(dimension_label_value_list, key=lambda x: x[2], reverse=True)
#     top_emb_list = []
#     for i in range(len(dimension_label_value_list[:5])):
#         dim = dimension_label_value_list[i][0]
#         label = self.labels[dimension_label_value_list[i][1]]
#         value = dimension_label_value_list[i][2]
#         # top_emb = sorted(enumerate(self.embeddings), key=lambda x: x[1][dim], reverse=True)[:10]
#         top_emb = sorted(enumerate(self.embeddings), key=lambda x: x[1][dim], reverse=True)[:10]
#         # sorted_emb_list = sorted(enumerate(self.embeddings), key=lambda x: x[1][dim], reverse=True)
#
#         print(
#             'dimension %d labelled as %s with score %f. Top words in this dimension:' % (dim, label, value.item()))
#
#         # exit(31)
#         top_emb = [(self.idx2word[emb_idx]) for emb_idx, emb in top_emb]
#         # print(
#         #     'dimension %d labelled as %s with score %f. Top words in this dimension:' % (dim, label, value.item()))
#         print(top_emb)
#         top_emb_list.append((top_emb, dim, value.item()))
#     return top_emb_list, label
