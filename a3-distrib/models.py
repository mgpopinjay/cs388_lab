# models.py

from optimizers import *
from nerdata import *
from utils import *

import random
import time

from collections import Counter
from typing import List

import numpy as np


class ProbabilisticSequenceScorer(object):
    """
    Scoring function for sequence models based on conditional probabilities.
    Scores are provided for three potentials in the model: initial scores (applied to the first tag),
    emissions, and transitions. Note that CRFs typically don't use potentials of the first type.

    Attributes:
        tag_indexer: Indexer mapping BIO tags to indices. Useful for dynamic programming
        word_indexer: Indexer mapping words to indices in the emission probabilities matrix
        init_log_probs: [num_tags]-length array containing initial sequence log probabilities
        transition_log_probs: [num_tags, num_tags] matrix containing transition log probabilities (prev, curr)
        emission_log_probs: [num_tags, num_words] matrix containing emission log probabilities (tag, word)
    """
    def __init__(self, tag_indexer: Indexer, word_indexer: Indexer, init_log_probs: np.ndarray, transition_log_probs: np.ndarray, emission_log_probs: np.ndarray):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs

    def score_init(self, sentence_tokens: List[Token], tag_idx: int):
        return self.init_log_probs[tag_idx]

    def score_transition(self, sentence_tokens: List[Token], prev_tag_idx: int, curr_tag_idx: int):
        return self.transition_log_probs[prev_tag_idx, curr_tag_idx]

    def score_emission(self, sentence_tokens: List[Token], tag_idx: int, word_posn: int):
        word = sentence_tokens[word_posn].word
        word_idx = self.word_indexer.index_of(word) if self.word_indexer.contains(word) else self.word_indexer.index_of("UNK")
        return self.emission_log_probs[tag_idx, word_idx]


class HmmNerModel(object):
    """
    HMM NER model for predicting tags

    Attributes:
        tag_indexer: Indexer mapping BIO tags to indices. Useful for dynamic programming
        word_indexer: Indexer mapping words to indices in the emission probabilities matrix
        init_log_probs: [num_tags]-length array containing initial sequence log probabilities
        transition_log_probs: [num_tags, num_tags] matrix containing transition log probabilities (prev, curr)
        emission_log_probs: [num_tags, num_words] matrix containing emission log probabilities (tag, word)
    """
    def __init__(self, tag_indexer: Indexer, word_indexer: Indexer, init_log_probs, transition_log_probs, emission_log_probs):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs
        self.scorer = ProbabilisticSequenceScorer(tag_indexer, word_indexer, init_log_probs, transition_log_probs, emission_log_probs)


    def decode(self, sentence_tokens: List[Token]) -> LabeledSentence:
        """
        See BadNerModel for an example implementation
        :param sentence_tokens: List of the tokens in the sentence to tag
        :return: The LabeledSentence consisting of predictions over the sentence
        """

        pred_tags = []
        tags = self.tag_indexer

        # Initialize N x T path probability matrix.  N = n_tags, T = n_tokens.
        n_tags = len(tags)
        n_tokens = len(sentence_tokens)  # length of observations

        viterbi = np.zeros((n_tags, n_tokens))
        backpointer = np.zeros((n_tags, n_tokens))

        for s in range(N):
            viterbi[s, 0] = self.init_log_probs[s] + self.scorer.score_emission(sentence_tokens, s, 0)  # Use Prob. Seq Scorer
            # print('\nViterbi at', s, ':', viterbi[s])

        # Recursively compute viterbi matrix
        for word_idx in range(1, n_tokens):
            for curr_tag_idx in range(n_tags):
                max_vals = np.zeros(n_tags)

                for prev_tag_idx in range(n_tags):
                    trans_prob = self.transition_log_probs[prev_tag_idx, curr_tag_idx]  # prior prob.
                    emis_prob = self.scorer.score_emission(sentence_tokens, curr_tag_idx, word_idx)  # likelihood prob.
                    max_vals[prev_tag_idx] = viterbi[prev_tag_idx, word_idx - 1] + trans_prob + emis_prob

                    # print("viterbi", viterbi)
                    # print("max_vals", max_vals)

                    viterbi[curr_tag_idx, word_idx] = np.max(max_vals)
                    backpointer[curr_tag_idx, word_idx] = np.argmax(max_vals)

        pred_tags.append(self.tag_indexer.get_object(np.argmax(viterbi[:, n_tokens - 1])))

        for word_idx in range(1, n_tokens):
            pred_tags.append(self.tag_indexer.get_object(backpointer[self.tag_indexer.index_of(pred_tags[-1]), n_tokens - word_idx]))

        pred_tags.reverse()
        chunks = chunks_from_bio_tag_seq(pred_tags)

        return LabeledSentence(sentence_tokens, chunks)


def train_hmm_model(sentences: List[LabeledSentence], silent: bool=False) -> HmmNerModel:
    """
    Uses maximum-likelihood estimation to read an HMM off of a corpus of sentences.
    Any word that only appears once in the corpus is replaced with UNK. A small amount
    of additive smoothing is applied.
    :param sentences: training corpus of LabeledSentence objects
    :return: trained HmmNerModel
    """
    # Index words and tags. We do this in advance so we know how big our
    # matrices need to be.
    tag_indexer = Indexer()
    word_indexer = Indexer()
    word_indexer.add_and_get_index("UNK")
    word_counter = Counter()
    for sentence in sentences:
        for token in sentence.tokens:
            word_counter[token.word] += 1.0
    for sentence in sentences:
        for token in sentence.tokens:
            # If the word occurs fewer than two times, don't index it -- we'll treat it as UNK
            get_word_index(word_indexer, word_counter, token.word)
        for tag in sentence.get_bio_tags():
            tag_indexer.add_and_get_index(tag)

    # Count occurrences of initial tags, transitions, and emissions
    # Apply additive smoothing to avoid log(0) / infinities / etc.
    init_counts = np.ones((len(tag_indexer)), dtype=float) * 0.001
    transition_counts = np.ones((len(tag_indexer),len(tag_indexer)), dtype=float) * 0.001
    emission_counts = np.ones((len(tag_indexer),len(word_indexer)), dtype=float) * 0.001
    for sentence in sentences:
        bio_tags = sentence.get_bio_tags()
        for i in range(0, len(sentence)):
            tag_idx = tag_indexer.add_and_get_index(bio_tags[i])
            word_idx = get_word_index(word_indexer, word_counter, sentence.tokens[i].word)
            emission_counts[tag_idx][word_idx] += 1.0
            if i == 0:
                init_counts[tag_idx] += 1.0
            else:
                transition_counts[tag_indexer.add_and_get_index(bio_tags[i-1])][tag_idx] += 1.0

    # Turn counts into probabilities for initial tags, transitions, and emissions. All
    # probabilities are stored as log probabilities
    if not silent:
        print("\ninit_counts\n", repr(init_counts))

    init_counts = np.log(init_counts / init_counts.sum())

    # transitions are stored as count[prev state][next state], so we sum over the second axis
    # and normalize by that to get the right conditional probabilities
    transition_counts = np.log(transition_counts / transition_counts.sum(axis=1)[:, np.newaxis])

    # similar to transitions
    emission_counts = np.log(emission_counts / emission_counts.sum(axis=1)[:, np.newaxis])

    if not silent:
        print("\nTag indexer: \n %s" % tag_indexer)
        print("\nInitial state log probabilities: \n %s" % init_counts)
        print("\nTransition log probabilities: \n  %s" % transition_counts)
        print("\nEmission log probs too big to print...")
        print("\nEmission log probs for India: \n %s" % emission_counts[:,word_indexer.add_and_get_index("India")])
        print("\nEmission log probs for Phil: \n %s" % emission_counts[:,word_indexer.add_and_get_index("Phil")])
        print("\nnote that these distributions don't normalize because it's p(word|tag) that normalizes, not p(tag|word)")
    return HmmNerModel(tag_indexer, word_indexer, init_counts, transition_counts, emission_counts)


def get_word_index(word_indexer: Indexer, word_counter: Counter, word: str) -> int:
    """
    Retrieves a word's index based on its count. If the word occurs only once, treat it as an "UNK" token
    At test time, unknown words will be replaced by UNKs.
    :param word_indexer: Indexer mapping words to indices for HMM featurization
    :param word_counter: Counter containing word counts of training set
    :param word: string word
    :return: int of the word index
    """
    if word_counter[word] < 1.5:
        return word_indexer.add_and_get_index("UNK")
    else:
        return word_indexer.add_and_get_index(word)


##################
# CRF code follows

class FeatureBasedSequenceScorer(object):
    """
    Feature-based sequence scoring model. Note that this scorer is instantiated *for every example*: it contains
    the feature cache used for that example.
    """
    def __init__(self, tag_indexer, feature_weights, feat_cache):
        self.tag_indexer = tag_indexer
        self.feature_weights = feature_weights
        self.feat_cache = feat_cache

    def score_init(self, sentence, tag_idx):
        if isI(self.tag_indexer.get_object(tag_idx)):
            return -1000
        else:
            return 0

    def score_transition(self, sentence_tokens, prev_tag_idx, curr_tag_idx):
        prev_tag = self.tag_indexer.get_object(prev_tag_idx)
        curr_tag = self.tag_indexer.get_object(curr_tag_idx)
        if (isO(prev_tag) and isI(curr_tag))\
                or (isB(prev_tag) and isI(curr_tag) and get_tag_label(prev_tag) != get_tag_label(curr_tag)) \
                or (isI(prev_tag) and isI(curr_tag) and get_tag_label(prev_tag) != get_tag_label(curr_tag)):
            return -1000
        else:
            return 0

    def score_emission(self, sentence_tokens, tag_idx, word_posn):
        feats = self.feat_cache[word_posn][tag_idx]
        return self.feature_weights.score(feats)
        # return score_indexed_features(feats, self.feature_weights)

class CrfNerModel(object):
    def __init__(self, tag_indexer, feature_indexer, feature_weights):
        self.tag_indexer = tag_indexer
        self.feature_indexer = feature_indexer
        self.feature_weights = feature_weights

    def decode(self, sentence_tokens: List[Token]) -> LabeledSentence:
        """
        See BadNerModel for an example implementation
        :param sentence_tokens: List of the tokens in the sentence to tag
        :return: The LabeledSentence consisting of predictions over the sentence
        """

        pred_tags = []
        tag_indexer = self.tag_indexer

        # Initialize N x T path probability matrix.  N = n_tags, T = n_tokens.
        n_tags = len(self.tag_indexer)
        n_tokens = len(sentence_tokens)  # length of observations

        viterbi = np.zeros((n_tags, n_tokens))
        backpointer = np.zeros((n_tags, n_tokens))

        # initialize 3-d feature cache indexed by tag index and word index
        feat_cache = [[[] for tag_idx in range(n_tags)] for word_idx in range(n_tokens)]

        for word_idx in range(n_tokens):
            for tag_idx in range(n_tags):
                feat_cache[word_idx][tag_idx] = extract_emission_features(
                    sentence_tokens,
                    word_idx,
                    tag_indexer.get_object(tag_idx),
                    self.feature_indexer,
                    add_to_indexer=False)

        # FeatureBasedScoreer takes: tag_indexer, feature_weights, feat_cache
        feat_scorer = FeatureBasedSequenceScorer(tag_indexer, self.feature_weights, feat_cache)

        for tag_idx in range(n_tags):
            viterbi[tag_idx, 0] = feat_scorer.score_emission(sentence_tokens, tag_idx, 0)  # Use Prob. Seq Scorer
            # print('\nCRF Viterbi at', tag_idx, ':', viterbi[tag_idx])

        # Recursively compute viterbi matrix
        for word_idx in range(n_tokens):
            for curr_tag_idx in range(n_tags):
                max_vals = np.zeros(n_tags)  # N = tag counts
                emis_score = feat_scorer.score_emission(sentence_tokens, curr_tag_idx, word_idx)  # score_emission takes: sentence_tokens, tag_idx, word_posn

                if word_idx == 0:
                    trans_score = feat_scorer.score_init(sentence_tokens, curr_tag_idx)
                else:
                    for prev_tag_idx in range(n_tags):
                        trans_score_temp = feat_scorer.score_transition(sentence_tokens, prev_tag_idx, curr_tag_idx)
                        max_vals[prev_tag_idx] = viterbi[prev_tag_idx, word_idx -1] + trans_score_temp
                    backpointer[curr_tag_idx, word_idx] = np.argmax(max_vals)
                    trans_score = np.max(max_vals)
                viterbi[curr_tag_idx, word_idx] = trans_score + emis_score

        # back track to collect tag indexes
        pred_tags.append(tag_indexer.get_object(np.argmax(viterbi[:, n_tokens - 1])))

        for t in range(1, n_tokens):
            pred_tags.append(tag_indexer.get_object(backpointer[tag_indexer.index_of(pred_tags[-1]), n_tokens - t]))

        pred_tags.reverse()
        chunks = chunks_from_bio_tag_seq(pred_tags)

        return LabeledSentence(sentence_tokens, chunks)


    def decode_beam(self, sentence_tokens: List[Token]) -> LabeledSentence:
        """
        See BadNerModel for an example implementation
        :param sentence_tokens: List of the tokens in the sentence to tag
        :return: The LabeledSentence consisting of predictions over the sentence
        """

        beam_size = 5
        beam_list = []
        tag_indexer = self.tag_indexer

        n_tags = len(self.tag_indexer)
        n_tokens = len(sentence_tokens)  # length of observations

        # Initialize N x T backpointer matrix.  N = n_tags, T = n_tokens.
        backpointer = np.zeros((n_tags, n_tokens), int)

        # initialize 3-d feature cache indexed by tag index and word index
        feat_cache = [[[] for tag_idx in range(n_tags)] for word_idx in range(n_tokens)]
        # feature_cache = [[[] for k in range(0, len(self.tag_indexer))] for j in range(0, len(sentence_tokens))]

        for word_idx in range(n_tokens):
            for tag_idx in range(n_tags):
                feat_cache[word_idx][tag_idx] = extract_emission_features(sentence_tokens, word_idx, tag_indexer.get_object(tag_idx), self.feature_indexer, add_to_indexer=False)

        # FeatureBasedScorer takes: tag_indexer, feature_weights, feat_cache
        feat_scorer = FeatureBasedSequenceScorer(tag_indexer, self.feature_weights, feat_cache)

        # tracker = np.zeros((n_tokens, n_tags), int)  // replaced by Backpointer

        for word_idx in range(n_tokens):
            beam = Beam(beam_size)
            for curr_tag_idx in range(n_tags):
                emission_score = feat_scorer.score_emission(sentence_tokens, curr_tag_idx, word_idx)

                if word_idx == 0:
                    transition_score = feat_scorer.score_init(sentence_tokens, word_idx)
                else:
                    beam_target = beam_list[word_idx - 1]
                    prev_tags = [prev_tag_idx for prev_tag_idx, _ in beam_target.get_elts_and_scores()]
                    # print('\nprev_tags:', prev_tags)

                    trans_score_temp = [feat_scorer.score_transition(sentence_tokens, prev_tag_idx, curr_tag_idx) + score for prev_tag_idx,score in beam_target.get_elts_and_scores()]

                    trans_score_temp_idx = trans_score_temp.index(max(trans_score_temp))

                    prev_tag = prev_tags[trans_score_temp_idx]
                    # print('\nprev_tag:', prev_tag)
                    backpointer[curr_tag_idx][word_idx] = prev_tag
                    transition_score = trans_score_temp[trans_score_temp_idx]



                beam.add(curr_tag_idx, emission_score + transition_score)
                # print("\nbeam:", beam)
            beam_list.append(beam)
            # print('beam_list:', beam_list)


        bio_tags = []
        tag_idx = 0

        for word_idx in reversed(range(n_tokens)):
            bio_tags.append(tag_indexer.ints_to_objs[tag_idx])
            tag_idx = backpointer[tag_idx][word_idx]

        return LabeledSentence(sentence_tokens, chunks_from_bio_tag_seq(bio_tags[::-1]))


def train_crf_model(sentences: List[LabeledSentence], silent: bool=False) -> CrfNerModel:
    """
    Trains a CRF NER model on the given corpus of sentences.
    :param sentences: The training data
    :param silent: True to suppress output, false to print certain debugging outputs
    :return: The CrfNerModel, which is primarily a wrapper around the tag + feature indexers as well as weights
    """

### GIVEN TEMPLATE - No Modification Needed ###
    tag_indexer = Indexer()
    for sentence in sentences:
        for tag in sentence.get_bio_tags():
            tag_indexer.add_and_get_index(tag)
    if not silent:
        print("Extracting features")
    feature_indexer = Indexer()

    # 4-d list indexed by sentence index, word index, tag index, feature index
    feature_cache = [[[[] for k in range(0, len(tag_indexer))] for j in range(0, len(sentences[i]))] for i in range(0, len(sentences))]

    for sentence_idx in range(0, len(sentences)):
        if sentence_idx % 100 == 0 and not silent:
            print("Ex %i/%i" % (sentence_idx, len(sentences)))
        for word_idx in range(0, len(sentences[sentence_idx])):
            for tag_idx in range(0, len(tag_indexer)):
                feature_cache[sentence_idx][word_idx][tag_idx] = extract_emission_features(sentences[sentence_idx].tokens, word_idx, tag_indexer.get_object(tag_idx), feature_indexer, add_to_indexer=True)
    if not silent:
        print("Training")

    weight_vector = UnregularizedAdagradTrainer(np.zeros((len(feature_indexer))), eta=1.0)
    num_epochs = 1
    random.seed(0)

    for epoch in range(0, num_epochs):
        epoch_start = time.time()

        if not silent:
            print("Epoch %i" % epoch)
        sent_indices = [i for i in range(0, len(sentences))]
        random.shuffle(sent_indices)
        total_obj = 0.0

        for counter, i in enumerate(sent_indices):
            if counter % 100 == 0 and not silent:
                print("Ex %i/%i" % (counter, len(sentences)))
            scorer = FeatureBasedSequenceScorer(tag_indexer, weight_vector, feature_cache[i])
            (gold_log_prob, gradient) = compute_gradient(sentences[i], tag_indexer, scorer, feature_indexer)
            total_obj += gold_log_prob
            weight_vector.apply_gradient_update(gradient, 1)
        if not silent:
            print("Objective for epoch: %.2f in time %.2f" % (total_obj, time.time() - epoch_start))

    return CrfNerModel(tag_indexer, feature_indexer, weight_vector)


def extract_emission_features(sentence_tokens: List[Token], word_index: int, tag: str, feature_indexer: Indexer, add_to_indexer: bool):
    """
    Extracts emission features for tagging the word at word_index with tag.
    :param sentence_tokens: sentence to extract over
    :param word_index: word index to consider
    :param tag: the tag that we're featurizing for
    :param feature_indexer: Indexer over features
    :param add_to_indexer: boolean variable indicating whether we should be expanding the indexer or not. This should
    be True at train time (since we want to learn weights for all features) and False at test time (to avoid creating
    any features we don't have weights for).
    :return: an ndarray
    """
    feats = []
    curr_word = sentence_tokens[word_index].word
    # Lexical and POS features on this word, the previous, and the next (Word-1, Word0, Word1)
    for idx_offset in range(-1, 2):
        if word_index + idx_offset < 0:
            active_word = "<s>"
        elif word_index + idx_offset >= len(sentence_tokens):
            active_word = "</s>"
        else:
            active_word = sentence_tokens[word_index + idx_offset].word
        if word_index + idx_offset < 0:
            active_pos = "<S>"
        elif word_index + idx_offset >= len(sentence_tokens):
            active_pos = "</S>"
        else:
            active_pos = sentence_tokens[word_index + idx_offset].pos
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Word" + repr(idx_offset) + "=" + active_word)
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Pos" + repr(idx_offset) + "=" + active_pos)
    # Character n-grams of the current word
    max_ngram_size = 3
    for ngram_size in range(1, max_ngram_size+1):
        start_ngram = curr_word[0:min(ngram_size, len(curr_word))]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":StartNgram=" + start_ngram)
        end_ngram = curr_word[max(0, len(curr_word) - ngram_size):]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":EndNgram=" + end_ngram)
    # Look at a few word shape features
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":IsCap=" + repr(curr_word[0].isupper()))
    # Compute word shape
    new_word = []
    for i in range(0, len(curr_word)):
        if curr_word[i].isupper():
            new_word += "X"
        elif curr_word[i].islower():
            new_word += "x"
        elif curr_word[i].isdigit():
            new_word += "0"
        else:
            new_word += "?"
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":WordShape=" + repr(new_word))
    return np.asarray(feats, dtype=int)


def compute_gradient(sentence: LabeledSentence, tag_indexer: Indexer, scorer: FeatureBasedSequenceScorer, feature_indexer: Indexer) -> (float, Counter):
    """
    Computes the gradient of the given example (sentence). The bulk of this code will be computing marginals via
    forward-backward: you should first compute these marginals, then accumulate the gradient based on the log
    probabilities.
    :param sentence: The LabeledSentence of the current example
    :param tag_indexer: The Indexer of the tags
    :param scorer: FeatureBasedSequenceScorer is a scoring model that wraps the weight vector and which also contains a
    feat_cache field that will be useful when computing the gradient.
    :param feature_indexer: The Indexer of the features
    :return: A tuple of two items. The first is the log probability of the correct sequence, which corresponds to the
    training objective. This value is only needed for printing, so technically you do not *need* to return it, but it
    will probably be useful to compute for debugging purposes.
    The second value is a Counter containing the gradient -- this is a sparse map from indices (features)
    to weights (gradient values).
    """

    # compute marginals
    # accumulate gradient based on log probabilities
    # return the log prob of correct sequence (for printing) and the gradient Counter (sparse map from indices to weight)


# ### WIP ###

    n_tokens = len(sentence)
    n_tags = len(tag_indexer)

    forward = np.zeros((n_tokens, n_tags))
    backward = np.zeros((n_tokens, n_tags))
    emission_mat = np.zeros((n_tokens, n_tags))
    transition_mat = np.zeros((n_tags, n_tags))
    marginal_probs = np.zeros((n_tokens, n_tags))


    # Forward
    for tag_idx in range(n_tags):
        # base case for the forward part
        forward[0, tag_idx] = scorer.score_emission(sentence.tokens, tag_idx, 0)
        backward[len(sentence.tokens) - 1, tag_idx] = 0

    for word_idx in range(1, n_tokens):
        for curr_tag_idx in range(n_tags):
            for prev_tag_idx in range(n_tags):
                emission = scorer.score_emission(sentence.tokens, curr_tag_idx, word_idx)
                tran = scorer.score_transition(sentence.tokens, prev_tag_idx, curr_tag_idx)

                curr_score = forward[word_idx - 1, prev_tag_idx] + emission + tran

                if prev_tag_idx > 0:
                    forward[word_idx, curr_tag_idx] = np.logaddexp(forward[word_idx, curr_tag_idx], curr_score)
                else:
                    forward[word_idx, curr_tag_idx] = curr_score
            emission_mat[word_idx, curr_tag_idx] = emission


    # Backward
    for token in reversed(range(n_tokens - 1)):
        for tag in range(n_tags):
            for next_tag in range(n_tags):
                emission = emission_mat[token + 1, next_tag]
                transition = transition_mat[tag, next_tag]
                curr_score = backward[token + 1, next_tag] + emission + transition

                if next_tag > 0:
                    backward[token, tag] = np.logaddexp(backward[token, tag], curr_score)
                else:
                    backward[token, tag] = curr_score

    # Merge forward and backward part
    normalizer = np.zeros(n_tokens)

    for word_idx in range(n_tokens):
        # Norm
        normalizer[word_idx] = forward[word_idx, 0] + backward[word_idx, 0]
        for tag_idx in range(1, n_tags):
            curr_score = forward[word_idx, tag_idx] + backward[word_idx, tag_idx]
            normalizer[word_idx] = np.logaddexp(normalizer[word_idx], curr_score)

        # Marginal
        norm = normalizer[word_idx]
        for tag_idx in range(n_tags):
            curr_score = forward[word_idx, tag_idx] + backward[word_idx, tag_idx]
            marginal_probs[word_idx, tag_idx] = curr_score - norm

    gold_tags = []
    bio_tags = bio_tags_from_chunks(sentence.chunks, n_tokens)
    for tag in bio_tags:
        gold_tags.append(tag_indexer.index_of(tag))

    gradient = Counter()
    for t in range(n_tokens):
        gradient.update(scorer.feat_cache[t][gold_tags[t]])

        for y in range(n_tags):
            for feat in scorer.feat_cache[t][y]:
                gradient[feat] -= np.exp(marginal_probs[t, y])

    gold_emissions = np.sum([emission_mat[t, y] for t, y in enumerate(gold_tags)])
    gold_transitions = np.sum([transition_mat[y, y_next] for y, y_next in zip(gold_tags[:-1], gold_tags[1:])])

    gold_log_prob = gold_emissions + gold_transitions - normalizer[0]

    return gold_log_prob, gradient


