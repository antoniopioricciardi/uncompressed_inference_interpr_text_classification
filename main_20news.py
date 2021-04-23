import random
import time
import torch

from pprint import pprint
from trainer import Trainer
from nlp_test_parser import *
from nn_classifier import NNClassifier, DeepLinear, DoubleLinear, SingleLinear
from dimension_operations_test import Dimension_operations
from features_contrib import FeaturesContributions
from text_features_contrib import DimensionOperations

from sklearn.datasets import fetch_20newsgroups
from sklearn import metrics


def parse_dataset_20news(dataset_list, labels_list):
    dataset = []
    labels = []
    for doc_idx, doc in enumerate(dataset_list):
        label = labels_list[doc_idx]
        labels.append(label)
        doc = doc.lower()
        # count = 0
        doc = re.sub('[^A-Za-z0-9]+', ' ', doc)
        words_list = tokenizer(doc)
        words_list = [w for w in words_list if w not in stop_words and w not in string.punctuation]
        dataset.append((words_list, label))
    print('done parsing')
    return dataset, labels

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

newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

# pprint(newsgroups_train.data[0])
# pprint(len(newsgroups_train.data))
# pprint(newsgroups_train.target_names)

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

labels = list(set(newsgroups_train.target))
# PARSE RAW DATA AND CREATE DATASET
train_data, _ = parse_dataset_20news(newsgroups_train.data, newsgroups_train.target)
test_data, _ = parse_dataset_20news(newsgroups_test.data, newsgroups_test.target)
words_l = train_data[0][0]

preproc_dataset_train = create_train_dataset(train_data, labels)
preproc_dataset_test = create_train_dataset(test_data, labels)

#### TRAINING AND TESTING CODE ####
# If we're testing, just load the model; otherwise train and save model parameters.
TESTING = True
train_dataset = generate_batches(preproc_dataset_train, BATCH_SIZE)
test_dataset = generate_batches(preproc_dataset_test, BATCH_SIZE)

############################ MODEL ####################
model = DeepLinear(EMB_DIM, len(labels), N_HIDDEN_LAYERS, 0.1, None)

if TESTING:
    # model2 is trained with leave-out test set, 83% acc
    model.load_state_dict(torch.load(MODEL_PATH))
else:
    trainer = Trainer(model, train_dataset, test_dataset, n_epochs=N_EPOCHS, model_path=MODEL_PATH)
    model = trainer.train()


model.eval()
feat_contribs = FeaturesContributions(model, N_HIDDEN_LAYERS)


sentence = 'i played some rock with my guitar, then played some pink floyd'
sentence = newsgroups_test.data[42]
dim_op = DimensionOperations(emb_all, embeddings, idx2word, word2idx, EMB_DIM, labels)
mean_emb, emb_list, words_list = dim_op.get_mean_embedding(sentence)

mul_weights = feat_contribs.get_features_contribution_matrix(mean_emb)

data = torch.tensor(mean_emb).float().to(model.device)

preds = model.forward(data)
pred = torch.argmax(preds)
label = labels[pred]
print('Prediction for text:', sentence, '\n', newsgroups_train.target_names[label])  # , ':', preds.cpu().detach()[pred])
print('ground truth:', newsgroups_train.target_names[newsgroups_test.target[42]])
print(preds)

dim_op.get_important_dimensions_for_selected_class(mul_weights, pred)

print('\n-----------------------------------------\n')
dim_op.get_word_ranking_for_selected_class(emb_list, words_list, mul_weights, pred)