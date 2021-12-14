# models.py

from sentiment_data import *
from utils import *
from collections import Counter
import string
import random
import numpy as np
import nltk
from nltk.corpus import stopwords


class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """

        # raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor): # 1.29 TODO: make sure works with data structure (SentimentExample object)
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = Indexer
        

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extracts unigram bag-of-words features from a sentence reflecting feature counts. 
        The sentence preprocessing involves: lower casing, punctutation removal
        :param sentence: words in the example to featurize.
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        :return: A feature vector.
        """        
        global features

        # print('sentence:', sentence)

        # stop_words = set(stopwords.words('english'))
        punctuations = string.punctuation
        # stop_words.add(set(punctuations))
        sentence = [word.lower() for word in sentence if word not in set(punctuations) | stop_words]
        features = Counter(sentence)
        return features
    
        
    def feature_vector_size(self) -> int:
        """
        Get the size of the feature vector.
        :return: vector size.
        """
        return len(features)
        
    # data of data.txt is: [<class '__main__.SentimentExample'>, <class '__main__.SentimentExample'>, ...]
    # (data[0].words)        

class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = Indexer
        

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extracts bigram bag-of-words features from a sentence reflecting feature counts. 
        The sentence preprocessing involves: lower casing, punctutation removal
        :param sentence: words in the example to featurize.
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        :return: A feature vector.
        """        
        global features
        
        punctuations = string.punctuation
        sentence = [(sentence[i - 1].lower(), sentence[i].lower()) for i in range(1, len(sentence)) \
                    if sentence[i-1] not in punctuations and sentence[i] not in punctuations]
        features = Counter(sentence)
        return features
    
        
    def feature_vector_size(self) -> int:
        """
        Get the size of the feature vector.
        :return: vector size.
        """
        return len(features)


class BetterFeatureExtractor(FeatureExtractor):
    def __init__(self, indexer: Indexer):
        self.indexer = Indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        """
        Alternative to TF-IDF approach (commented out).  Adding nltk stopwords for removal from feature vector.
        """
        global features

        # Using hard-coded stopword list from nltk due to import issues.
        # stop_words = set(stopwords.words('english'))
        punctuations = string.punctuation
        sentence = [word.lower() for word in sentence if word not in set(punctuations) | stop_words]
        features = Counter(sentence)
        return features


    def feature_vector_size(self) -> int:
        """
        Get the size of the feature vector.
        :return: vector size.
        """
        return len(features)

        
# class BetterFeatureExtractor(FeatureExtractor):
#     """
#     Better feature extractor...try whatever you can think of!
#     """
#     def __init__(self, indexer: Indexer):
#         self.indexer = Indexer
#         self.doc_count = 0
#         # self.features = Counter()
#         self.corpus_freq = Counter()   #number of docs containing the word
#         self.tf_idf = Counter()
#
#     def extract_features(self, sentence: List[str], doc_length=1, count= 1, add_to_indexer: bool = False) -> Counter:
#         """
#         TF-IDF
#         corpus_freq accumulates
#         """
#         self.doc_count += count
#
#         global features
#
#         punctuations = string.punctuation
#         sentence = [word.lower() for word in sentence if word not in punctuations]
#
#         features = Counter(sentence)
#
#         length = len(sentence)
#         # print("pre-normalized:", features)
#         for key in features.keys():
#             features[key] = features[key] / length
#             # print("normalized:", features)
#
#             # Accumulate the number of docs across the corpus containing the word
#             if key in sentence:
#                 self.corpus_freq[key] += 1
#
#         # At last doc/sentence of the corpus, calculate TF-IDF
#         if self.doc_count == doc_length:
#             # TF = normalized word count across sentences = features
#             tf = features
#
#             # TF * log( total number of docs / number of docs containing the term)
#             for key in tf.keys():
#                 self.tf_idf[key] = float(tf[key] * (np.log( self.doc_count / self.corpus_freq[key])))
#
#             print("tf_idf:", self.tf_idf)
#
#         return self.tf_idf



class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    
    # def __init__(self, feat_extractor: FeatureExtractor, train_exs: List[SentimentExample]):
    def __init__(self, train_exs: List[SentimentExample], feat_extractor: FeatureExtractor):
        self.feat_extractor = feat_extractor
        self.train_exs = train_exs
        
        self.corpus_vocab = Counter()
        self.weight_vector = Counter()
        
    def create_corpus_vocab(self) -> Counter: #TODO: ??cache??
        """
        Creates the corpus vocabulary by aggregating sparse vectors of each sample (i.e. sentence) 
            in the training dataset.
        :return: A Counter of all the words in the corpus and their frequencies.
        """         
        
        data_size = len(self.train_exs)

        # for each example in training data
        for i in range(data_size):

            # compute sparse feature vector 
            sentence_feature_vector = self.feat_extractor.extract_features(self.train_exs[i].words) # ??? why is the self in extract_features(self,..) necessary here???

            # update corpus 
            self.corpus_vocab += sentence_feature_vector

        # print("train_exs:", self.train_exs)
        return self.corpus_vocab

                  
    
    def initialize_weight_vector(self) -> Counter: # should be the same size as the corpus vocab
        """
        Initializes the weight vector (same size as the corpus vocabulary) with zeroes.
        :return: A Counter of words with zeroes as weights.
        """           

        for word in list(self.corpus_vocab): 
            self.weight_vector.update({word: 0})
        
        return self.weight_vector
    
    
    def dot_product(self, corpus_weight_Counter: Counter, sentence_feature_Counter: Counter) -> float: 
        """
        Computes the dot product of the weight vector and the feature vector for one sentence. 
        :param corpus_weight_Counter: vector of all corpus words and their current weights
        :param sentence_feature_Counter: vector of sentence features and feature frequencies
        :return: dot product value
        """   

        result = 0
        for word in list(sentence_feature_Counter):
            result += corpus_weight_Counter[word]
        return result

#    corpus = Counter({'the': 1, 'rock': 0, 'is': 1, 'destined': 0, 'to': -1, 'be': 0, '21st': 0, 'century': 0, "'s": 0, 'new': 0})
#    feature = Counter({'the': 205, 'rock': 8,  'destined': 3,  'be': 88, '21st': 2})
#    print(dot_product (corpus, feature))    
    
    
    def predicted_label(self, sentence_feature_Counter: Counter) -> bool: #?? needs self??
        """
        Predicts a sentiment label for a single sample (i.e. sentence).
        :param sentence_feature_Counter: Counter of sentence features and feature frequencies
        :return: label 
        """

        if self.dot_product(self.weight_vector, sentence_feature_Counter) > 0:
            return 1
        else:
            return 0 


    # def update_weights(self, sentence_feature_Counter: Counter, operation = "add", learning_rate = 1):
    def update_weights(self, sentence_feature_Counter: Counter, learning_rate=1, operation="add"):

        """
        Updates weight vector using feature vector values.
        :param X: 
        :return:  
        """         

        for word in list(sentence_feature_Counter):
            if operation == "add":
                self.weight_vector[word] += (learning_rate * 1)
            else:
                self.weight_vector[word] -= (learning_rate * 1)
        

    def train_classifier(self, epochs = 10, rate= "progressive", print_results = True): # ??? how many iterations??
        """
        XXX
        :param X: 
        :return:  
        """    
        # ?? need to compute loss anywhere??
        data_size = len(self.train_exs) 
        accuracy = np.zeros([data_size]) 
        
        # create corpus vocabulary 
        corpus_counter = self.create_corpus_vocab()    

        # initialize weight vector
        weight_vector = self.initialize_weight_vector() 


        for i in range(epochs):

            if rate == "constant":
                learning_rate = (1 // (i + 1))
            else:
                learning_rate = 1

            # shuffle data
            random.shuffle(self.train_exs)
            doc_length = len(self.train_exs)

            for j in range(len(self.train_exs)): 
                
                # create feature vector for current sample (i.e. sentence)
                feature_vector = self.feat_extractor.extract_features(self.train_exs[j].words, doc_length)


                # predict label
                current_label = self.predicted_label(feature_vector)

                # make updates to weight vector based on predicted label
                # pred 0, true label = 1
                if current_label < self.train_exs[j].label:
                    self.update_weights(feature_vector, learning_rate, operation = "add")

                # pred 1, true label = 0
                elif current_label > self.train_exs[j].label:
                    self.update_weights(feature_vector, learning_rate, operation = "subtract")

                else:
                    accuracy[j] = 1

            #
            # if print_results:
            #     print(f'Epoch: {i} --> Accuracy: {round(np.mean(accuracy), 2) * 100}%')

        return self.feat_extractor


    def predict(self, sentence: List[str]) -> int: # TODO: what needs to be done here?? how is it different?
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """

        sentence_feature = self.feat_extractor.extract_features(sentence)

        if self.dot_product(self.weight_vector, sentence_feature) > 0:
            return 1
        else:
            return 0


          
class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, train_exs: List[SentimentExample], feat_extractor: FeatureExtractor):
        self.feat_extractor = feat_extractor
        self.train_exs = train_exs
        self.corpus_vocab = Counter()
        self.weight_vector = Counter()
            
        
    def create_corpus_vocab(self) -> Counter: #TODO: ??cache??
        """
        Creates the corpus vocabulary by aggregating sparse vectors of each sample (i.e. sentence) 
            in the training dataset.
        :return: A Counter of all the words in the corpus and their frequencies.
        """         
        
        data_size = len(self.train_exs)

        # for each example in training data
        for i in range(data_size):
            
            # compute sparse feature vector 
            sentence_feature_vector = self.feat_extractor.extract_features(self.train_exs[i].words) # ??? why is the self in extract_features(self,..) necessary here???
            
            # update corpus 
            self.corpus_vocab += sentence_feature_vector
            
        return self.corpus_vocab

    
    def initialize_weight_vector(self) -> Counter: # should be the same size as the corpus vocab
        """
        Initializes the weight vector (same size as the corpus vocabulary) with zeroes.
        :return: A Counter of words with zeroes as weights.
        """           

        for word in list(self.corpus_vocab): 
            self.weight_vector.update({word: 0})
        
        return self.weight_vector
 
    
    def dot_product(self, corpus_weight_Counter: Counter, sentence_feature_Counter: Counter) -> float: 
        """
        Computes the dot product of the weight vector and the feature vector for one sentence. 
        :param corpus_weight_Counter: vector of all corpus words and their current weights
        :param sentence_feature_Counter: vector of sentence features and feature frequencies
        :return: dot product value
        """   

        result = 0
        for word in list(sentence_feature_Counter):
            result += corpus_weight_Counter[word]
        return result

    
    def sigmoid(self, weight_feature_dot_product: int) -> float: # TODO: input is float or int?
        """
        Maps the dot product of weight vector and feature vector to a probability (of positive class).
        :param weight_feature_dot_product: 
        :return: 
        """      
        
        positive_class_probability = 1 / (1 + np.exp(-weight_feature_dot_product))      
        return positive_class_probability    
        

    def predicted_label(self, weight_feature_dot_product: float) -> bool: #?? needs self??
        """
        Predicts a sentiment label for a single sample (i.e. sentence).
        :param sentence_feature_Counter: Counter of sentence features and feature frequencies
        :return: label 
        """     
        
        if self.sigmoid(weight_feature_dot_product) > 0.5: # ?? self stuff correct??
            return 1
        else:
            return 0 
   
     
    def update_weights(self, sentence_feature_Counter: Counter, positive_class_probability: float, learning_rate = 1, operation="add"):
        """
        Updates weight vector using feature vector values.
        :param X: 
        :return:  
        """         

        # For Positive Label

        for word in list(sentence_feature_Counter):
            if operation == "subtract":
                self.weight_vector[word] -= (learning_rate * 1) * (positive_class_probability)
            else:
                self.weight_vector[word] += (learning_rate * 1) * (1- positive_class_probability)

        return self.weight_vector



    def train_classifier(self, epochs = 30, rate = "constant", print_results = True): # ??? how many iterations??
        """
        XXX
        :param X: 
        :return:  
        """    
        # ?? need to compute loss anywhere??
        data_size = len(self.train_exs) 
        accuracy = np.zeros([data_size]) 
        
        # create corpus vocabulary 
        corpus_counter = self.create_corpus_vocab()    

        # initialize weight vector
        weight_vector = self.initialize_weight_vector()
        doc_length = len(self.train_exs)

        for i in range(epochs):

            if rate == "progressive":
                learning_rate = (1 // (i + 1))
            else:
                learning_rate = 0.05

            # shuffle data
            random.shuffle(self.train_exs)

            for j in range(len(self.train_exs)): 
                
                # create feature vector for current sample (i.e. sentence)
                feature_vector = self.feat_extractor.extract_features(self.train_exs[j].words, doc_length)
                weight_feature_dot_product = self.dot_product(weight_vector, feature_vector)
                positive_class_probability = self.sigmoid(weight_feature_dot_product)

                # make updates to weight vector based on predicted label

                if self.train_exs[j].label == 0:
                    self.update_weights(feature_vector, positive_class_probability, learning_rate, operation = "subtract")
                else:
                    self.update_weights(feature_vector, positive_class_probability, learning_rate, operation = "add")

                # predict label
                current_label = self.predicted_label(weight_feature_dot_product)

                if current_label == self.train_exs[j].label:
                    accuracy[j] = 1
                else:
                    accuracy[j] = 0

            if print_results:
                print(f'Epoch: {i} --> Training Accuracy: {round(np.mean(accuracy), 2) * 100}%')

        return self.feat_extractor


    def predict(self, sentence: List[str]) -> int:  # TODO: what needs to be done here?? how is it different?
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """

        feature_vector = self.feat_extractor.extract_features(sentence)
        weight_feature_dot_product = self.dot_product(self.weight_vector, feature_vector)

        if self.sigmoid(weight_feature_dot_product) > 0.5: # ?? self stuff correct??
            return 1
        else:
            return 0

    # weight vector (initially zero)
    # feature vector (per sentence)
    # go through the sentences
        # compute sentence feature vector
        # compute dot product of sentence feature vector and weight vector
        # pass result of dot prodcut to sigmoid to get probability of positive class
        # update weights
        
        
        # if probability of positive class > 0.5: TODO: finalize architecture after this line
            # predicted label = 1
                # no update needed ???
        # else:
            # predicted label = 0

        
    # Making a Logistic Regression Classifier Model:
        
        #1 convert corpus into features 
        #2 initialize weight vector
        
        #3 run epochs
        #4 compute label 
        #5 compute loss 
        #6 update weight vector
            # repeat 3-6


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """

    model = PerceptronClassifier(train_exs, feat_extractor)
    model.train_classifier()
    return model

def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """

    model = LogisticRegressionClassifier(train_exs, feat_extractor)
    model.train_classifier()
    return model

def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model



stop_words = {'a',
 'about',
 'above',
 'after',
 'again',
 'against',
 'ain',
 'all',
 'am',
 'an',
 'and',
 'any',
 'are',
 'aren',
 "aren't",
 'as',
 'at',
 'be',
 'because',
 'been',
 'before',
 'being',
 'below',
 'between',
 'both',
 'but',
 'by',
 'can',
 'couldn',
 "couldn't",
 'd',
 'did',
 'didn',
 "didn't",
 'do',
 'does',
 'doesn',
 "doesn't",
 'doing',
 'don',
 "don't",
 'down',
 'during',
 'each',
 'few',
 'for',
 'from',
 'further',
 'had',
 'hadn',
 "hadn't",
 'has',
 'hasn',
 "hasn't",
 'have',
 'haven',
 "haven't",
 'having',
 'he',
 'her',
 'here',
 'hers',
 'herself',
 'him',
 'himself',
 'his',
 'how',
 'i',
 'if',
 'in',
 'into',
 'is',
 'isn',
 "isn't",
 'it',
 "it's",
 'its',
 'itself',
 'just',
 'll',
 'm',
 'ma',
 'me',
 'mightn',
 "mightn't",
 'more',
 'most',
 'mustn',
 "mustn't",
 'my',
 'myself',
 'needn',
 "needn't",
 'no',
 'nor',
 'not',
 'now',
 'o',
 'of',
 'off',
 'on',
 'once',
 'only',
 'or',
 'other',
 'our',
 'ours',
 'ourselves',
 'out',
 'over',
 'own',
 're',
 's',
 'same',
 'shan',
 "shan't",
 'she',
 "she's",
 'should',
 "should've",
 'shouldn',
 "shouldn't",
 'so',
 'some',
 'such',
 't',
 'than',
 'that',
 "that'll",
 'the',
 'their',
 'theirs',
 'them',
 'themselves',
 'then',
 'there',
 'these',
 'they',
 'this',
 'those',
 'through',
 'to',
 'too',
 'under',
 'until',
 'up',
 've',
 'very',
 'was',
 'wasn',
 "wasn't",
 'we',
 'were',
 'weren',
 "weren't",
 'what',
 'when',
 'where',
 'which',
 'while',
 'who',
 'whom',
 'why',
 'will',
 'with',
 'won',
 "won't",
 'wouldn',
 "wouldn't",
 'y',
 'you',
 "you'd",
 "you'll",
 "you're",
 "you've",
 'your',
 'yours',
 'yourself',
 'yourselves'}



'''###############################################------ Unit Tests ------################################################'''
 
   
'''............ 1. UnigramFeatureExtractor ............'''    

''' 1a. create a feature extractor object '''
#unigram_extractor = UnigramFeatureExtractor(Indexer) 
#print(unigram_extractor) 

''' 1b. extract sample (i.e. sentence) features using UnigramFeatureExtractor '''
#sentence = ["I", "am", "here", ".", "I", "made", "this", ",", "Class"]
#print(unigram_extractor.extract_features(sentence)) 

''' 1c. get size of sample feature vector (number of words in Counter) '''
#print(unigram_extractor.feature_vector_size())

''' 1d. TF-IDF tester '''
# sentence = ["I", "am", "here", ".", "I", "made", "this", ",", "Class"]
# doc_count = 1
#
# dev_exs = read_sentiment_examples("data/dev.txt")
# doc_count = len(dev_exs)
#
# better = BetterFeatureExtractor(Indexer)
# extract = [ better.extract_features(sentence.words, doc_count, count=1) for sentence in dev_exs ]




'''............ 1.2. BigramFeatureExtractor ............'''

''' 1.2.a. create a feature extractor object '''
# bigram_extractor = BigramFeatureExtractor(Indexer)
# print(bigram_extractor)

''' 1.2.b. extract sample (i.e. sentence) features using BigramFeatureExtractor '''
# sentence = ["I", "am", "here", ".", "I", "made", "this", ",", "Class"]
# print(bigram_extractor.extract_features(sentence))

''' 1.2.c. get size of sample feature vector (number of words in Counter) '''
# print(bigram_extractor.feature_vector_size())


   
'''............ 2. PerceptronClassifier ............'''

''' 2a. create a PerceptronClassifier object with UnigramFeatureExtractor and train.txt data '''
# train_exs = read_sentiment_examples("data/train.txt")
# perceptron = PerceptronClassifier(train_exs, UnigramFeatureExtractor)
# perceptron = PerceptronClassifier(train_exs, BigramFeatureExtractor)
# print(type(perceptron))

# print(perceptron.feat_extractor)

''' 2b. print first sentence (list of tokens) of dataset '''
# print((perceptron.train_exs)[0].words) # prints first sentence of the dataset

''' 2c. get initial corpus vocabulary and weight_vector (empty collections.Counter objects) '''
# print(perceptron.corpus_vocab)
# print(perceptron.weight_vector)

''' 2d. create corpus vocabulary of train.txt data '''
# corpus_counter = perceptron.create_corpus_vocab()
# print(corpus_counter) # prints corpus vocabulary and frequencies

''' 2e. create initial weight vector with zeroes '''
# NOTE: must use create_corpus_vocab to create corpus vocab prior to using this line
# perceptron.initialize_weight_vector()
#print(perceptron.weight_vector) # prints weight vector with all weights set to 0

''' 2f. train PerceptronClassifier with UnigramFeatureExtractor and train.txt data '''
# perceptron.train_classifier(epochs = 3)

''' 2g. get weight vector after training '''
# print(perceptron.weight_vector) # ??? should it have values > 1 or < -1??

''' 2h. create a PerceptronClassifier object with UnigramFeatureExtractor and dev.txt data '''
# dev_exs = read_sentiment_examples("data/dev.txt")
# dev_perceptron = PerceptronClassifier(UnigramFeatureExtractor, dev_exs)
# dev_perceptron = PerceptronClassifier(BigramFeatureExtractor, dev_exs)
# dev_perceptron.train_classifier(epochs = 3, rate = "constant")
# print(dev_perceptron.weight_vector) # ?? achieves 100% accuracy with 10 epochs, may be overfitting??? reduce epochs?

''' 2i. test the PerceptronClassifier (dev.txt data) with different learning rates ''' # TODO: figure out
# dev_exs = read_sentiment_examples("data/dev.txt")
# dev_perceptron = PerceptronClassifier(dev_exs, BigramFeatureExtractor)
# print("\nConstant learning rate = 1  w/ Bigram")
# dev_perceptron.train_classifier(epochs = 3, rate = "progressive")
# print("\nProgressive learning rate = 1/i  w/ Bigram")
# dev_perceptron.train_classifier(epochs = 3, rate = "constant")


''' 2j. Test DocTest Path  '''
# dev_exs = read_sentiment_examples("data/dev.txt")
#
# ### DocTest path: train_model -> train_perceptron --> outputs a classifier object
# perceptron_model = train_perceptron(dev_exs, BigramFeatureExtractor)
# print(type(perceptron_model))
#
# ### How DocTest feeds in sentences to .predict method --> list of booleans:
# prediction = [perceptron_model.predict(ex.words) for ex in dev_exs]
# print("\nPrediction labels:", prediction)



'''............ 3. LogisticRegressionClassifier ............'''

''' 3a. create a LogisticRegressionClassifier object with UnigramFeatureExtractor and train.txt data '''
#train_exs = read_sentiment_examples("data/train.txt")
#LR = LogisticRegressionClassifier(UnigramFeatureExtractor, train_exs)     
#print(type(LR))  


''' 3b. train LogisticRegressionClassifier with UnigramFeatureExtractor and train.txt data '''
#LR.train_classifier(epochs = 5)

''' 3c. create a LogisticRegressionClassifier object with UnigramFeatureExtractor and dev.txt data '''
#dev_exs = read_sentiment_examples("data/dev.txt")
#dev_LR = LogisticRegressionClassifier(UnigramFeatureExtractor, dev_exs)   
#dev_LR.train_classifier(epochs = 5)
#print(dev_LR.weight_vector) # ?? achieves 100% accuracy with 10 epochs, may be overfitting??? reduce epochs?

''' 3d. create a LogisticRegressionClassifier object with BigramFeatureExtractor and train.txt data '''
#train_exs = read_sentiment_examples("data/train.txt")
#LR = LogisticRegressionClassifier(BigramFeatureExtractor, train_exs)     
#print(type(LR)) 

''' 3e. train LogisticRegressionClassifier with BigramFeatureExtractor and train.txt data '''
#LR.train_classifier(epochs = 5) 
#print(len(LR.weight_vector))
#print(LR.weight_vector)

''' 3f. create a LogisticRegressionClassifier object with BigramFeatureExtractor and dev.txt data '''
#dev_exs = read_sentiment_examples("data/dev.txt")
#dev_LR = LogisticRegressionClassifier(BigramFeatureExtractor, dev_exs)   
#dev_LR.train_classifier(epochs = 5)
#print(dev_LR.weight_vector) # ?? achieves 100% accuracy with 10 epochs, may be overfitting??? reduce epochs?
