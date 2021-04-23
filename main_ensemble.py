import random
import time

import torch

from trainer import Trainer

from nlp_test_parser import *
from nn_classifier import NNClassifier, DeepLinear, DoubleLinear, SingleLinear
from dimension_operations_test import Dimension_operations
from features_contrib import FeaturesContributions
from text_features_contrib import DimensionOperations

# need to run only first time, then comment code.
# import nltk
# nltk.download('stopwords')

''' DEFINING PATHS '''
raw_dataset_path = 'datasets/glossboot_dataset'
dest_path = 'test_data'
EMB_PATH = 'embeddings/SPINE_word2vec.txt'
#EMB_PATH = 'embeddings/glove_original_15k_300d_train.txt'

MODEL_PATH = 'models/model.pth'

if not os.path.exists('models'):
    os.mkdir('models')

''' DEFINING MODEL PARAMETERS '''
BATCH_SIZE = 64
EMB_DIM = 1000
N_HIDDEN_LAYERS = 2
N_EPOCHS = 120

#### LOAD AND PARSE EMBEDDINGS AND DATA ####

# READ EMBEDDINGS
emb_all = load_embeddings(EMB_PATH)
print(len(emb_all))
print('done reading embeddings')

embeddings = emb_all.values()
idx2word = dict()
word2idx = dict()
for idx, word in enumerate(emb_all.keys()):
    idx2word[idx] = word
    word2idx[word] = idx

# PARSE RAW DATA AND CREATE DATASET
dataset, labels = parse_dataset(raw_dataset_path)

words_l = dataset[0][0]

preproc_dataset = create_train_dataset(dataset, labels)
print('training dataset length:', len(preproc_dataset))

random.shuffle(preproc_dataset)  # randomly shuffle train dataset


#### TRAINING AND TESTING CODE ####
# If we're testing, just load the model; otherwise train and save model parameters.
TESTING = True
train_dataset = generate_batches(preproc_dataset, BATCH_SIZE)
test_dataset = []
for i in range(4):  # test set contains 4 batches
    test_dataset.append(train_dataset.pop())

############################ MODEL ####################
model = DeepLinear(EMB_DIM, len(labels), N_HIDDEN_LAYERS, 0.01, None)

if TESTING:
    # model2 is trained with leave-out test set, 83% acc
    model.load_state_dict(torch.load(MODEL_PATH))
else:
    trainer = Trainer(model, train_dataset, test_dataset, n_epochs=N_EPOCHS, model_path=MODEL_PATH)
    model = trainer.train()

model.eval()
feat_contribs = FeaturesContributions(model, N_HIDDEN_LAYERS)


sentence = 'i played some rock with my guitar, then played some pink floyd'
dim_op = DimensionOperations(emb_all, embeddings, idx2word, word2idx, EMB_DIM, labels)
mean_emb, emb_list, words_list = dim_op.get_mean_embedding(sentence)

preds = {lab: 0 for lab in labels}
preds_scores = []  # contains a list of scores for every class, for every word
uncompressed_inf_list = []
for idx, emb in enumerate(emb_list):
    mul_weights = feat_contribs.get_features_contribution_matrix(emb)
    uncompressed_inf_list.append(mul_weights)

    data = torch.tensor(emb).float().to(model.device)
    pred = model.forward(data)
    preds_scores.append(pred)
    pred = torch.argmax(pred)
    preds[labels[pred]] += 1

print(preds)
print(uncompressed_inf_list[0])
maxkey = max(preds, key=preds.get)
print(maxkey)
exit(3)



mul_weights = feat_contribs.get_features_contribution_matrix(mean_emb)

data = torch.tensor(mean_emb).float().to(model.device)

preds = model.forward(data)
pred = torch.argmax(preds)
label = labels[pred]
print('Prediction for text:', sentence, '\n', label)  # , ':', preds.cpu().detach()[pred])

dim_op.get_important_dimensions_for_selected_class(mul_weights, pred)

print('\n-----------------------------------------\n')
dim_op.get_word_ranking_for_selected_class(emb_list, words_list, mul_weights, pred)