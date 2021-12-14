# models.py

import torch
import torch.nn as nn
import numpy as np
import random
from sentiment_data import *
import torch.nn.functional as F
from torch import optim


class SentimentClassifier(object):
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


class FFNet(nn.Module):
    def __init__(self, input_ch=300, num_classes=2):
        """
        Constructs the computation graph by instantiating the various layers and initializing weights.
        :param input_ch: size of input (integer)
        :param hidden_layers: size of hidden layer(integer)
        :param num_classes: size of output (integer), which should be the number of classes
        """
        super(FFNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_ch, 256),
            nn.Linear(256, 128),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):

        return  F.softmax(self.net(x))


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, network, word_embeddings, embedding_dim):
        self.model = network
        self.embedding = word_embeddings
        self.embedding_dim = embedding_dim

    def embed_sentence(self, sentence: List[str]):
        # print("\nsent:", sentence)
        emb_sum = np.zeros(self.embedding_dim)

        for token in sentence:
            # extract a (300,) embedding
            token_embedding = self.embedding.get_embedding(token)
            # print("\ntoken_embedding:", token_embedding)
            emb_sum += np.array(token_embedding)
            # print("\nemb_sum:", emb_sum)
        embeddings_avg = emb_sum / len(sentence)

        return embeddings_avg

    def predict(self, ex_words: List[str]) -> int:

        embeddings_avg = self.embed_sentence(ex_words)
        train_x_batch = torch.unsqueeze(torch.from_numpy(embeddings_avg).float(), 0)

        # get sentiment probability
        probability = self.model.forward(train_x_batch)[0]
        prediction = torch.argmax(probability)

        return prediction


    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        return [self.predict(ex_words) for ex_words in all_ex_words]


def embed_sentences(sentiment_exs, word_embeddings, embedding_dim=300):
    """
    Unpacks sentiment objects and embeds each sentence using input word embeddings
    """
    batch_size = len(sentiment_exs)
    embedded_sent, labels = [], []

    for batch in range(batch_size):
        emb_dim = np.zeros(embedding_dim)
        sentence_len = len(sentiment_exs[batch][0].words)  # unpack sentiment object from batch list

        for token_num in range(sentence_len):
            token = sentiment_exs[batch][0].words[token_num]
            token_embedding = word_embeddings.get_embedding(token)
            emb_dim += np.array(token_embedding)

        token_avg = emb_dim / batch_size
        embedded_sent.append(token_avg)

        label = sentiment_exs[batch][0].label
        labels.append(label)

    return np.array(embedded_sent), np.array(labels)


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


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """

    epochs = 50
    batch_size = 32
    learning_rate = 0.0001
    embedding_dim = 300  # adjust according to dataset
    num_classes = 2
    # drop_out = .2

    model = FFNet(embedding_dim, num_classes)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=1, factor=0.5)

    model.train()

    for epoch in range(epochs):
        # shuffle at each epoch
        random.shuffle(train_exs)

        # batch after shuffle
        batch_exs = batch_sentences(train_exs, batch_size)
        running_loss = 0.0

        for i in range(batch_size):

            # embed batch by batch
            batch_emb_x, batch_emb_y = embed_sentences(batch_exs, word_embeddings, embedding_dim)

            train_x_batch = torch.from_numpy(batch_emb_x).float()
            # train_x_batch = torch.from_numpy(batch_emb_x).long()

            train_y_batch = torch.from_numpy(batch_emb_y)

            optimizer.zero_grad()
            pred_batch = model(train_x_batch)

            batch_loss = loss_func(pred_batch, train_y_batch)
            running_loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()

        epoch_loss = running_loss/batch_size
        lr_scheduler.step(epoch_loss)

        print("Loss on epoch %i: %f" % (epoch, epoch_loss))
        # print(f"loss: {epoch_loss}")

    model.eval()

    return NeuralSentimentClassifier(model, word_embeddings, embedding_dim)
