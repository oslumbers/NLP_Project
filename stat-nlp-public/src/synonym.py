import glob
import importlib
import logging
import nltk
import numpy as np
import os
import scipy.stats
import sys
import time
import torch
import train_eval

from keras.preprocessing.sequence import pad_sequences
from tqdm import trange, tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# from transformers import *

nltk.download("wordnet")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize

# Global Variables
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
IMDB = os.path.join(DATA_DIR, 'aclImdb')
YELP = os.path.join(DATA_DIR, 'yelp')
r_state = 0


class SynonymAttacks:
    def __init__(self, model_path, tokenizer, max_seq):
        self.tokenizer = tokenizer
        self.max_seq = max_seq
        self.model = torch.load(model_path)

    @staticmethod
    def returnFirstSynonym(word):
        """
        Return the first synonym of a word with (approx) matching pos from wordnet 
        """
        pos = nltk.pos_tag(word)[0][1]
        targets = ["NN", "NNS", "JJ", "JJS", "JJR", "RB", "RBR",
                   "RBS", "VB", "VBD", "VBG", "VBN", "VBP"]

        if pos not in targets:
            return False
        else:
            # match to the word net equivalent pos
            wn_pos = pos[0].lower()

            for s in wn.synsets(word):
                name = s.name()
                if name.find(".{}.".format(wn_pos)):
                    synonym = name.split(".")[0]
                    if word not in synonym:
                        return synonym 
            else:
                return False      
    
    @staticmethod
    def returnRandomSynonym(word):
        """
        Return random synonym with (approx) matching pos from wordnet 
        """
        pos = nltk.pos_tag(word)[0][1]
        targets = ["NN", "NNS", "JJ", "JJS", "JJR", "RB", "RBR",
                   "RBS", "VB", "VBD", "VBG", "VBN", "VBP"]

        if pos not in targets:
            return False
        else:
            # match to the word net equivalent pos
            wn_pos = pos[0].lower()
            wn_synonyms = wn.synsets(word)
            np.random.shuffle(wn_synonyms)
            for s in wn_synonyms:
                name = s.name()
                if name.find(".{}.".format(wn_pos)):
                    synonym = name.split(".")[0]
                    if word not in synonym:
                        return synonym 
                    else:
                        continue
            else:
                return False      


    def createBlankedReviews(self, sentences, method="blank"):
        """
        Yields blanked reviews, with the original review, and the word that was
        switched or blanked

        sentences: numpy array of text samples to loop over
        method: one of "blank", or "synonym"
        """
        max_seq = self.max_seq
        
        for review in sentences:        
            encoding = np.array(self.tokenizer.encode(review, max_length=max_seq))
            base_review = self.tokenizer.decode(encoding)
            blanked_reviews = []
            blanked_words = []
            for j, word in enumerate(encoding):
                # Don't bother for words we will truncate
                syn = self.returnFirstSynonym(self.tokenizer.decode([word]))

                if j > max_seq-1:
                    continue
                if syn:
                    mask = np.ones(len(encoding), dtype=np.bool)
                    mask[j] = 0

                    if method == "blank":
                        blanked_reviews.append(self.tokenizer.decode(encoding[mask]))
                    elif method == "synonym":
                        syn_rev = np.insert(encoding[mask], j, self.tokenizer.encode(syn)[1])
                        blanked_reviews.append(self.tokenizer.decode(syn_rev))

                    blanked_words.append(self.tokenizer.decode(encoding[~mask]))

            yield base_review, blanked_reviews, blanked_words

    def generateHotWords(self, train_data, train_labels, train_no=None, method="blank", max_batch=256):
        """
        Yields changes in classifier by word, based on blanking words in each review

        train_data: array of sentences
        train_labels: array of ints
        train_no: int, limit number to train on. default do whole array
        method: str, one of "blank", or "synonym". Either blanks words, or replaces them with a synonym
        max_batch: max batch to evaluate on
        """
        if not train_no:
            train_no = len(train_data)
        model = self.model
        tokenizer = self.tokenizer
        labels = train_labels
        blank_review_iterator = self.createBlankedReviews(train_data[0:train_no], method=method)

        for i, blanks in enumerate(blank_review_iterator):
            # Unpack the blanked reviews and words
            base_review, blanked_reviews, blanked_words = blanks
            blanked_words = np.array(blanked_words)

            # Set up the evaluation data
            all_reviews = np.array([base_review, *blanked_reviews])
            true_labels = np.full(len(all_reviews), labels[i])
            evaluation_data, _ = train_eval.ReviewDataset.setUpData(all_reviews, true_labels, tokenizer)

            # Run model to get softmax outputs
            batch = min([len(all_reviews), max_batch])
            _, return_pred_labels, sm, _, _ = train_eval.evaluate(model, evaluation_data, batch_size=batch, return_pred_labels=True)
            return_pred_labels = np.array(return_pred_labels)[0,:]

            # Keep changes soft max
            gross_change = np.sum(sm[0][1:, :] - sm[0][0, :], axis=1)

             # misclassify_words = np.where(return_pred_labels[1:] != return_pred_labels[0])
             # hot_words.append(blanked_words[misclassify_words])

            if i % 50 == 0:
                print("{}: Reviews complete: {}".format(time.ctime(), i))

            yield blanked_words, gross_change

    def createSynonymReview(self, sentence, label, replacements, hot_word_distribution, method="random", synonym_method="random"):
        """
        Returns an adverserial version of a review

        sentence: str, Target sentence to change
        label: int, Ground truth label
        replacements: int, How many words to switch out
        hot_word_distribution: dictionary of word: score (where a positive score will make a review more positive
                                                    and vice versa)
        method: str, random or hot_word (method for sampling for word replacement, either random or weighted)
        synonym_method: str, first or random (method for sampling wordnet synonym)
        """
        max_seq = self.max_seq

        encoding = np.array(self.tokenizer.encode(sentence, max_length=max_seq))[1:-1]
        no_words = len(encoding)
        importance = np.full(no_words, np.nan)

        for i, enc in enumerate(encoding):
            try:
                word = self.tokenizer.decode([enc]) 
                importance[i] = hot_word_distribution[word]
            except KeyError:
                # We don't have a score for this word: leave as nan
                pass
        
        # if it's negative label, switch sign
        if label == 0:
            importance = -importance
        
        # calc softmax
        exp = np.exp(importance)
        probabilities = exp / np.nansum(exp)

        # mask words we don't have scores for
        mask = np.isfinite(importance)        
        replaceable = np.sum(mask)
        idx = np.arange(no_words)[mask]
        
        # randomly sample words, weighted by their score
        if method == "hot_word":
            idx_to_replace = np.random.choice(idx, p=probabilities[mask], size=min(replaceable, replacements), replace=False)
        elif method == "random":
            idx_to_replace = np.random.choice(idx, size=min(replaceable, replacements), replace=False)
        
        words_to_replace = encoding[idx_to_replace]
        
        # loop over target words, replacing them with a random synonym from wordnet
        for i, w in zip(idx_to_replace, words_to_replace):
            if synonym_method == "random":
                syn = self.returnRandomSynonym(self.tokenizer.decode([w]).lower())
            elif synonym_method == "first":
                syn = self.returnFirstSynonym(self.tokenizer.decode([w]).lower())
           
            if syn:
                if "_" in syn:
                    # handle wordnet multi words replacements
                    syn = syn.split("_")
                
                # How our synonym is encoded
                replacement_encoding = self.tokenizer.encode(syn)[1:-1]

                # replaceAndInsert switches out the old encoding for the new one at each point in the array
                encoding = replaceAndInsert(encoding, w, replacement_encoding)

        return self.tokenizer.decode(encoding)            

    def generateSynonymReviews(self, target_data, target_label, replacements, hot_word_distribution, method):
        """
        Returns an adverserial version of a dataset

        target_data: Array of sentences: base data
        label: array of labels
        replacements: How many words to switch out in each instance
        hot_word_distribution: dictionary of word: score (where a positive score will make a review more positive
                                                    and vice versa)
        method: hot_word or random
        """
        adversarial_reviews = np.empty_like(target_data)
        
        for i, data in enumerate(zip(target_data, target_label)):
            review, label = data
            adversarial_reviews[i] = self.createSynonymReview(review, label, replacements,
                                                              hot_word_distribution, method)

        return adversarial_reviews, target_label

    def generateAttentions(self, train_data, train_labels, max_batch=256):
        """
        Return word attentions for each review
        """
        self.model.config.output_attentions = True
        self.evaluation_data, _ = train_eval.ReviewDataset.setUpData(train_data, train_labels, self.tokenizer)
        _, _, _, attentions, _ = train_eval.evaluate(self.model, self.evaluation_data, batch_size=max_batch, 
                                                    return_attentions=True)
        self.model.config.output_attentions = False
        return attentions



# ==================================================================================
# Some helper functionss
# ==================================================================================

def replaceAndInsert(target_array, target_value, replacement):
    """
    Recursive helper function to delete and insert in a numpy array
    """
    rs = np.where(target_array == target_value)
    if len(rs[0]) == 0:
        return target_array
    else:
        r = rs[0][0]
        target_array = np.delete(target_array, r)
        target_array = np.insert(target_array, r, replacement)
        return replaceAndInsert(target_array, target_value, replacement)