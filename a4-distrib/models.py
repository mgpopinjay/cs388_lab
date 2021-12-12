# models.py

import numpy as np
import collections
from torch import optim, nn
import torch
import random

#####################
# MODELS FOR PART 1 #
#####################

class ConsonantVowelClassifier(object):
    def predict(self, context):
        """
        :param context:
        :return: 1 if vowel, 0 if consonant
        """
        raise Exception("Only implemented in subclasses")


class FrequencyBasedClassifier(ConsonantVowelClassifier):
    """
    Classifier based on the last letter before the space. If it has occurred with more consonants than vowels,
    classify as consonant, otherwise as vowel.
    """
    def __init__(self, consonant_counts, vowel_counts):
        self.consonant_counts = consonant_counts
        self.vowel_counts = vowel_counts

    def predict(self, context):
        # Look two back to find the letter before the space
        if self.consonant_counts[context[-1]] > self.vowel_counts[context[-1]]:
            return 0
        else:
            return 1


class RNN(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, num_layers, num_classes, network="RNN"):
    # def __init__(self, vocab_size, input_size, hidden_size, num_classes):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = nn.Embedding(vocab_size, input_size)  # [27, 32]
        self.W = nn.Linear(hidden_size, num_classes)
        self.network = network

        ### Initialization how does this matters?
        nn.init.xavier_uniform_(self.W.weight)

        if self.network == "GRU":
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        if self.network == "LSTM":
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        else:
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)  # batch_fisrt=True: (batch_size, sequence_length, input_size)

        ### Choose RNN or its variants
        # self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)  # batch_fisrt=True: (batch_size, sequence_length, input_size)
        # self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)


    def forward(self, x):
        # Set initial hidden states & LSTM cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        embeddings = self.embeddings(x)  # embeddings size: [batch_size, 20, 32]; x size: [250, 20]


        if self.network == "LSTM":   # Need to unpack the tensor tuple
            # Forward propogate RNN
            output, h_n = self.rnn(embeddings, (h0, c0))
            # Decode last hidden state
            h_n = h_n[0].squeeze()
        else:
            # Forward propogate RNN
            output, h_n = self.rnn(embeddings, h0)      # ouput = (batch_size, sequence_length, 1*hidden_size) if NOT bidirectional
                                                        # h_n = final hidden state for each element in batch
            # Decode last hidden state
            h_n = h_n.squeeze()  # squeezed h_n size: [batch_size, hidden_size]

        return self.W(h_n)



class RNNClassifier(ConsonantVowelClassifier):
    def __init__(self, vocab_indexer, network):
        self.indexer = vocab_indexer
        self.vocab_size = len(self.indexer)
        self.network = network
        self.model = RNN(self.vocab_size, 32, 32, 1, 2, self.network)

    def indexing(self, context):
        context_idx = [self.indexer.index_of(char) for char in context]
        return context_idx

    def predict(self, context):
        probability = self.model(torch.tensor([self.indexing(context)]))
        return torch.argmax(probability)


def train_frequency_based_classifier(cons_exs, vowel_exs):
    consonant_counts = collections.Counter()
    vowel_counts = collections.Counter()
    for ex in cons_exs:
        consonant_counts[ex[-1]] += 1
    for ex in vowel_exs:
        vowel_counts[ex[-1]] += 1
    return FrequencyBasedClassifier(consonant_counts, vowel_counts)


def combine_exs(vocab_index, cons_exs, cons_label, vowel_exs, vowel_label):
    """
      :param vocab_index: an Indexer of the character vocabulary (27 characters)
      :param cons_exs: list of strings followed by consonants
      :param cons_label: 0 for consonants
      :param vowel_exs: list of strings followed by vowels
      :param vowel_label: 1 for vowels
      :return: combined list of training examples
      """

    train_exs = []
    all_exs = cons_exs + vowel_exs
    labels = len(cons_exs)*[cons_label] + len(vowel_exs)*[vowel_label]
    for ex_idx in range(len(all_exs)):
        char_idx = [vocab_index.index_of(char) for char in all_exs[ex_idx]]
        label = labels[ex_idx]
        train_exs.append([char_idx, label])
    return train_exs


def train_rnn_classifier(args, train_cons_exs, train_vowel_exs, dev_cons_exs, dev_vowel_exs, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_cons_exs: list of strings followed by consonants
    :param train_vowel_exs: list of strings followed by vowels
    :param dev_cons_exs: list of strings followed by consonants
    :param dev_vowel_exs: list of strings followed by vowels
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNClassifier instance trained on the given data
    """

    ### choose "GRU", "LSTM" or default to normal "RNN"
    network = "RNN"
    clf = RNNClassifier(vocab_index, network)

    epochs = 20
    batch_size = 250
    learning_rate = 0.01
    train_x, train_y = [], []

    optimizer = optim.Adam(clf.model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()

    # Combine and label data set using helper function combine_exs
    train_exs = combine_exs(vocab_index, train_cons_exs, 0, train_vowel_exs, 1)

    print("Training on:", network)

    for epoch in range(epochs):
        epoch_loss = 0.0
        random.shuffle(train_exs)
        for ex in train_exs:
            if len(train_x) < batch_size:
                char_idx, label = ex[0], ex[1]
                train_x.append(char_idx)
                train_y.append(label)
            else:
                clf.model.zero_grad()
                target = torch.tensor(train_y)
                characters = torch.tensor(train_x)
                probability = clf.model(characters)
                loss = loss_function(probability, target)
                epoch_loss += loss

                loss.backward()
                optimizer.step()
                train_x, train_y = [], []
        print("Total loss on epoch %i: %f" % (epoch, epoch_loss))
    print("Completed training on:", network)

    return clf



#####################
# MODELS FOR PART 2 #
#####################


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e
        :param context: a single character to score
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class RNNLM(nn.Module):

    def __init__(self, vocab_size, input_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embeddings = nn.Embedding(vocab_size, input_size)
        self.W = nn.Linear(hidden_size, vocab_size)
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)

        ### Initialization how does this matters?
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        embeddings = self.embeddings(x)  # embeddings size: [batch_size, 20, 32]; x size: [250, 20]

        # Forward propogate RNN
        output, h_n = self.rnn(embeddings, h0)      # ouput = (batch_size, sequence_length, 1*hidden_size) if NOT bidirectional
                                                    # h_n = final hidden state for each element in batch
        # Decode last hidden state
        h_n = h_n.squeeze()  # squeezed h_n size: [batch_size, hidden_size]

        return self.W(output), self.W(h_n)


class RNNLanguageModel(LanguageModel):
    def __init__(self, vocab_indexer):
        self.indexer = vocab_indexer
        self.vocab_size = len(self.indexer)
        self.model = RNNLM(self.vocab_size, 32, 32, 1)
        self.softmax = nn.LogSoftmax()

    def get_next_char_log_probs(self, context):
        context_indexed = [self.indexer.index_of(char) for char in context]
        _, h_n = self.model(torch.tensor([context_indexed]))
        log_prob = self.softmax(h_n.squeeze())

        return log_prob.detach().numpy()

    def get_log_prob_sequence(self, next_chars, context):
        seq = context + next_chars
        seq_idx = [self.indexer.index_of(char) for char in seq]
        # print('seq_indexed', seq_idx)

        output, _ = self.model(torch.tensor([seq_idx]))
        output = output.squeeze()

        total_log_prob = 0
        for i, next_char in enumerate(next_chars):
            hidden_state = output[len(context)-1+i, :]
            next_char_idx = self.indexer.index_of(next_char)
            log_prob_dist = self.softmax(hidden_state)
            log_prob = log_prob_dist.squeeze()[next_char_idx]

            total_log_prob += log_prob
        # print("tota_log_prob:", tota_log_prob)
        return total_log_prob.item()


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev texts as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNLanguageModel instance trained on the given data
    """

    clf = RNNLanguageModel(vocab_index)

    epochs = 20
    chunk_size = 25
    batch_size = 250
    learning_rate = 0.01
    train_exs = []
    train_x, train_y = [], []
    text_chunks = []

    optimizer = optim.Adam(clf.model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()


    for i in range(0, len(train_text), chunk_size):
        text_chunks.append(list(train_text)[i:i+chunk_size])
    # print("chunked_text:", text_chunk)

    for chunk in text_chunks:
        # Pad last chunk's tail if shorter than chunk size
        if chunk_size % len(chunk):
            chunk = chunk + list(' ')*(chunk_size % len(chunk))
            print("chunk padded:", chunk)

        # Output as input sequence shifted by one. Make input_seq from padding sliced output_seq
        output_seq = [vocab_index.index_of(char) for char in chunk]
        input_seq = [vocab_index.index_of(' ')] + output_seq[:-1]
        train_exs.append([input_seq, output_seq])

    for epoch in range(epochs):
        epoch_loss = 0.0
        random.shuffle(train_exs)

        for ex in train_exs:
            if len(train_x) < batch_size:
                char_idx, label = ex[0], ex[1]
                train_x.append(char_idx)
                train_y.append(label)
            else:
                clf.model.zero_grad()
                target = torch.tensor(train_y).view(batch_size*chunk_size)

                characters = torch.tensor(train_x)
                probability, hidden_n = clf.model(characters)
                probability = probability.view(batch_size*chunk_size, len(vocab_index))

                loss = loss_function(probability, target)
                epoch_loss += loss

                loss.backward()
                optimizer.step()
                train_x, train_y = [], []
        print("Total loss on epoch %i: %f" % (epoch, epoch_loss))

    return clf

