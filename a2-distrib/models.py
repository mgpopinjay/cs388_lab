# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *
from statistics import mean
import matplotlib.pyplot as plt  # for plotting
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F





class SentimentClassifier(object):  # TODO: change from object to nn.Module?
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class NeuralSentimentClassifier(SentimentClassifier, nn.Module):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """

    def __init__(self, word_embeddings, input_size, hidden_layer_size=4, num_classes=2):  # TODO: 4 ok? based on the exercise 8b example
        """
        Constructs the computation graph by instantiating the various layers and initializing weights.

        :param input_size: size of input (integer)
        :param hidden_layer_size: size of hidden layer(integer)
        :param num_classes: size of output (integer), which should be the number of classes
        """
        super(NeuralSentimentClassifier, self).__init__()  # TODO: exactly why this is needed?

        self.word_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(word_embeddings.vectors))
        self.word_embeddings.weight.requires_grad = False

        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.num_classes = num_classes

        # Linear & Nonlinear Transformations
        self.V = nn.Linear(self.input_size, self.hidden_layer_size)
        self.g = nn.Tanh() # option 1
        # self.g = nn.ReLU()  # option 2
        self.W = nn.Linear(self.hidden_layer_size, self.num_classes)


        # Tranforming to probablities
        self.log_softmax = nn.LogSoftmax(dim=0)


    def initialize_weights(self, with_zeros=False):  # TODO: figure out output
        """
        Initializes weights.
        :return: TODO
        """

        # initialize weights with zeros
        if with_zeros:
            nn.init.zeros_(self.V.weight)
            nn.init.zeros_(self.W.weight)

        # Xavier Glorot weight initialization
        else:
            nn.init.xavier_uniform_(self.V.weight)
            nn.init.xavier_uniform_(self.W.weight)

    def forward(self, x):
        """
        Runs the neural network on the given data and returns log probabilities of the various classes.

        :param x: a [input_size]-sized tensor of input data
        :return: an [num_classes]-sized tensor of log probabilities. (In general your network can be set up to return either log
        probabilities or a tuple of (loss, log probability) if you want to pass in y to this function as well
        """

        # print("\nx:", x.size())
        # print("V:", self.V(x).size())
        # print("g:", self.g(self.V(x)).size())
        # print("W:", self.W(self.g(self.V(x))).size())
        # print("W:", self.W(self.g(self.V(x))))


        return self.log_softmax(self.W(self.g(self.V(x))))
        # return self.W(self.g(self.V(x)))

    def indexing(self, sentence: List[str]):
        sentence_idx = []
        for word in sentence:
            index = max(1, self.indexer.index_of(word))
            sentence_idx.append(index)

        # print("\nsentence_idx", sentence_idx)
        return sentence_idx


    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """

        # argmax returns the indices of the maximum value of all elements in the input tensor.
        return torch.argmax(self.forward(ex_words))  # TODO: correct??



def batch_sentences(train_exs, batch_size):
    """
    Creates batches of sentences, with each batch containing batch_size number of sentences.
    """
    batched_sentences = []
    for i in range(0, len(train_exs), batch_size):
        batched_sentences.append(train_exs[i: i + batch_size])

    # print("batched_sentences:", batched_sentences)
    # print("batched_sentences size:", len(batched_sentences))
    return batched_sentences


def embed_sentences(train_exs, batch_size, num_classes, all_embeddings):
    """

    """
    batched_sentences = batch_sentences(train_exs, batch_size)  # each batch has batch_size number of sentences
    # print("batched_sentences:", batched_sentences)

    # print("batched_sentences:", len(batched_sentences))
    batched_embedding_averages = []
    batched_labels = []

    for batch_indx in range(len(batched_sentences)):

        embeddings = []
        labels = []

        for sentence in batched_sentences[batch_indx]:
            sentence_label = sentence.label

            # get embedding for each word in the sentence
            word_embeddings = [all_embeddings.get_embedding(word) for word in sentence.words]

            # compute sentence embedding average (add the embedding for each word and divide by number of words in sentence)
            sentence_embedding_average = np.mean(word_embeddings)

            embeddings.append(sentence_embedding_average)
            labels.append(sentence_label)

        batched_embedding_averages.append(embeddings)
        batched_labels.append(labels)

        # print("batched_embedding_averages:", batched_embedding_averages)
        # print("batched_labels:", batched_labels)

    return torch.from_numpy(np.asarray(batched_embedding_averages)).float(), torch.from_numpy(
        np.asarray(batched_labels))


def train_classifier(train_exs, initial_learning_rate, all_embeddings, num_classes, batch_size=20, epochs=10,
                     print_results=True):  # 5-10 epochs, lr = 3e-4. scheduler is overkill

    batched_embeddings, batched_labels = embed_sentences(train_exs, batch_size, num_classes, all_embeddings)

    print("batched_embeddings:", batched_embeddings)
    print("batched_labels:", batched_labels)

    max_len = -1
    for ex in train_exs:
        if len(ex.words) > max_len:
            max_len = len(ex.words)

    # construct neural net model
    input_size = batch_size
    hidden_layer_size = 256
    pad_size = max_len

    ffnn = NeuralSentimentClassifier(input_size, hidden_layer_size,
                                     num_classes=2)  # TODO: what should num_classes be?

    # initialize weights
    ffnn.initialize_weights(with_zeros=False)

    # construct an optimizer object
    optimizer = torch.optim.Adam(ffnn.parameters(), lr=initial_learning_rate)

    batch_x = []
    batch_y = []

    for epoch in range(epochs):
        random.shuffle(train_exs)
        total_loss = 0.0
        for ex in train_exs:
            # print("\nex:", ex)
            if len(batch_x) < batch_size:
                sentence_idx = ffnn.indexing(ex.words)  # index words locations in embedding
                # print('\nsentence_idx:', sentence_idx)
                padded_sentence_idx = [0] * pad_size
                padded_sentence_idx[:len(sentence_idx)] = sentence_idx
                # print('\npadded_sentence_idx:', padded_sentence_idx)
                # print('padded_sentence_idx:', padded_sentence_idx.__len__())

                label = ex.label
                # print('\nlabel:', label)

                batch_x.append(padded_sentence_idx)
                batch_y.append(label)
                # print('\nbatch_x:', batch_x)
                # print('\nbatch_y:', batch_y)
                # print('\nbatch_x:', batch_x.__len__())
                # print('\nbatch_y:', batch_y.__len__())

            else:
                target = torch.tensor(batch_y)
                ffnn.model.zero_grad()
                probability = ffnn.model.forward(torch.tensor(batch_x))
                print('\nprobability:', probability)
                print('\ntarget:', target)

                loss = ffnn.model.loss_function(probability, target)
                total_loss += loss
                loss.backward()
                optimizer.step()
                batch_x = []
                batch_y = []
        print("Total loss on epoch %i: %f" % (epoch, total_loss))
    return ffnn

    # for epoch in range(0, epochs):
    #
    #     total_loss = 0.0
    #     #        data_indices = [i for i in range(data_sentiment_labels.size)]
    #     batch_indices = [i for i in range(len(batched_embeddings))]
    #
    #     # shuffle data
    #     random.shuffle(batch_indices)
    #     for batch_index in batch_indices:
    #         embeddings = batched_embeddings[batch_index]
    #         labels = batched_labels[batch_index]
    #
    #         # zero out the gradients from the FFNN object
    #         ffnn.zero_grad()
    #
    #         print("\nembeddings", embeddings.size())
    #
    #         log_probs = ffnn.forward(embeddings)
    #         target = labels.long()  # TODO: how to get target?
    #
    #         # loss calculation option (has issues): NLL
    #         # loss_func = nn.NLLLoss()
    #         loss_func = nn.CrossEntropyLoss()
    #
    #
    #         print('\nCalculating Loss...')
    #         print('\nlog_prob_size', log_probs.size())
    #         print('\log_prob', log_probs)
    #         print('\ntarget_size', target.size())
    #         print('target', target)
    #
    #         loss = loss_func(log_probs, target)
    #         print('\nloss', loss)
    #
    #         # compute gradients
    #         loss.backward()
    #
    #         # update the model parameters based on the computed gradients
    #         optimizer.step()
    #
    #     if print_results:
    #         print(f"Loss on epoch {epoch}: {loss}")
    #
    # return ffnn


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """

    initial_learning_rate = 3e-4
    epochs = 10
    DAN_classifier = train_classifier(train_exs, initial_learning_rate, word_embeddings, epochs, print_results=False)

    return DAN_classifier



#######  REFERENCE Below ######




'''###############################################------ Unit Tests ------################################################'''


# NN = NeuralSentimentClassifier(300, 100, num_classes = 2)
#
# NN.initialize_weights(with_zeros = False) # Xavier Glorot
#
#
# train_exs = read_sentiment_examples("data/train.txt")
# initial_learning_rate = 3e-4
# word_embeddings = read_word_embeddings("data/glove.6B.300d-relativized.txt")
# model = train_classifier(train_exs, initial_learning_rate, word_embeddings, num_classes=2, epochs=10,
#                          print_results=False)

# ex_words = ["It 's", "a", "lovely", "film", "with", "lovely", "performances", "by", "Buy", "and", "Accorsi"]
# model.forward(ex_words)
