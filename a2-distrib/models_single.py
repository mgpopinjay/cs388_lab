# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *
import matplotlib.pyplot as plt  # for plotting


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

    def __init__(self, input_size, hidden_layer_size=4, num_classes=2):  # TODO: 4 ok? based on the exercise 8b example
        """
        Constructs the computation graph by instantiating the various layers and initializing weights.

        :param input_size: size of input (integer)
        :param hidden_layer_size: size of hidden layer(integer)
        :param num_classes: size of output (integer), which should be the number of classes
        """
        super(NeuralSentimentClassifier, self).__init__()  # TODO: exactly why this is needed?

        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.num_classes = num_classes

        # linear transformation
        self.V = nn.Linear(input_size, hidden_layer_size)

        # nonlinearity
        # self.g = nn.Tanh() # option 1
        self.g = nn.ReLU()  # option 2

        # linear transformation
        self.W = nn.Linear(hidden_layer_size, num_classes)

        # tranforming to probablities
        self.log_softmax = nn.LogSoftmax(dim=0)

        # TODO: nn.Embedding layer initialized appropriately


    def initialize_weights(self, with_zeros=False):  # TODO: figure out output
        """
        Initializes weights.
        :return: TODO
        """

        # initialize weights with zeros
        if with_zeros:
            nn.init.zeros_(self.V.weight)
            nn.init.zeros_(self.W.weight)

        # according to a formula due to Xavier Glorot
        else:
            nn.init.xavier_uniform_(self.V.weight)
            nn.init.xavier_uniform_(self.W.weight)

    def embed_sentence(self, all_embeddings, sentence: List[str]):
        # get embedding for each word in the sentence
        word_embeddings = [all_embeddings.get_embedding(word) for word in sentence]

        # compute sentence embedding average (add the embedding for each word and divide by number of words in sentence)
        sentence_embedding_average = np.mean(word_embeddings)

        # return torch.from_numpy(np.asarray(sentence_embedding_average))

        return np.asarray(sentence_embedding_average)

    def forward(self, x):
        """
        Runs the neural network on the given data and returns log probabilities of the various classes.

        :param x: a [input_size]-sized tensor of input data
        :return: an [num_classes]-sized tensor of log probabilities. (In general your network can be set up to return either log
        probabilities or a tuple of (loss, log probability) if you want to pass in y to this function as well
        """

        all_embeddings = read_word_embeddings(x)

        all_embeddings = read_word_embeddings("data/glove.6B.300d-relativized.txt")

        x = torch.from_numpy(np.asarray(self.embed_sentence(all_embeddings, x)))

        return self.log_softmax(self.W(self.g(self.V(x))))


    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        #        return torch.argmax(self.forward(torch.stack(ex_words).to(device)))

        return torch.argmax(self.forward(ex_words))  # TODO: correct??


def find_max_len(train_exs):
    """

    """
    max_len = 0
    for sentence in train_exs:
        if len(sentence.words) > max_len:
            max_len = len(sentence.words)
    return max_len


def add_padding(numpy_array, length):
    """
    Adds right padding to a numpy array with zeros to achieve size = length (e.g. [val, ..., val] -> [val, ..., val, 0, ..., 0]).
    """
    return np.append(numpy_array, np.zeros(length - numpy_array.shape[0]))  # TODO: pad with 0??


def train_classifier(train_exs, initial_learning_rate, all_embeddings, num_classes, epochs=10,
                     print_results=True):  # 5-10 epochs, lr = 3e-4. scheduler is overkill

    data_sentiment_labels = np.array([sentence.label for sentence in train_exs])

    input_size = all_embeddings.get_embedding_length()  # TODO: correct?

    # construct neural net model
    hidden_layer_size = 4
    ffnn = NeuralSentimentClassifier(input_size, hidden_layer_size,
                                     num_classes)  # TODO: 4 ok? based on the exercise 8b example

    # initialize weights
    ffnn.initialize_weights(with_zeros=True)

    # construct an optimizer object
    optimizer = torch.optim.Adam(ffnn.parameters(), lr=initial_learning_rate)  # TODO: has a schedueler?

    iters = []  # save the iteration counts here for plotting
    losses = []  # save the avg loss here for plotting

    num_examples = len(train_exs)
    for epoch in range(0, epochs):

        total_loss = 0.0
        data_indices = [i for i in range(data_sentiment_labels.size)]

        # shuffle data
        random.shuffle(data_indices)
        for index in data_indices:
            sentence = train_exs[index]
            #            sentence = padded_data[index] #old
            sentence_label = data_sentiment_labels[index]

            # get embedding for each word in the sentence
            word_embeddings = np.asarray([all_embeddings.get_embedding(word) for word in sentence.words])

            # compute sentence embedding average (add the embedding for each word and divide by number of words in sentence)
            sentence_embedding_average = np.mean(np.asarray(word_embeddings), axis=0)
            sentence_embedding_average = torch.from_numpy(
                sentence_embedding_average).float()


            # Build one-hot representation of sentence_label. Instead of the label 0 or 1, y_onehot is either [0, 1] or [1, 0]. This
            # way we can take the dot product directly with a probability vector to get class probabilities.
            label_onehot = torch.zeros(num_classes)
            label_onehot.scatter_(0, torch.from_numpy(np.asarray(sentence_label, dtype=np.long)), 1)

            # zero out the gradients from the FFNN object
            ffnn.zero_grad()
            log_probs = ffnn.forward(sentence_embedding_average)

            # calculate loss
            loss = (torch.neg(log_probs)).dot(label_onehot)
            total_loss += loss  # should be around 4000 over 7000 examples

            # compute gradients
            loss.backward()

            # update the model parameters based on the computed gradients
            optimizer.step()

            iters.append(int(epoch))
            losses.append(int(total_loss))

        if print_results:
            print(f"Total loss on epoch {epoch}: {total_loss}")


    return ffnn


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


'''###############################################------ Unit Tests ------################################################'''


NN = NeuralSentimentClassifier(300, 4, num_classes = 2)
NN.initialize_weights(with_zeros = False) # Xavier Glorot

train_exs = read_sentiment_examples("data/train.txt")
initial_learning_rate = 3e-4
word_embeddings = read_word_embeddings("data/glove.6B.300d-relativized.txt")
model = train_classifier(train_exs, initial_learning_rate, word_embeddings, num_classes = 2, epochs = 10, print_results = True)
ex_words = ["It 's", "a", "lovely", "film", "with", "lovely", "performances", "by", "Buy", "and", "Accorsi"]
word_tensor = torch.from_numpy(np.asarray([word_embeddings.get_embedding(word) for word in ex_words]))
print("word_embeddings:", word_tensor)
print(model.forward(ex_words))

