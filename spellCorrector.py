import pickle
import nltk
import string
import sys
from nltk.corpus.reader import wordnet
from nltk.corpus import words
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import *


class SpellCorrector:
    """
        Class that checks the spelling of the words in a sentence/paragraph and
        suggests correction
    """

    def __init__(self):
        """
        Constructor
        """
        self.dictionaryWords = [w.lower() for w in words.words()]
        self.trigrams = self.loadNgramModelFromFile(3)
        self.bigrams = self.loadNgramModelFromFile(2)
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.numberOfGrams = 3
        self.maxDistance = 4

    def check(self, x):
        """

        checks the spelling of the given sentence/paragraph and provides possible replacements for misspelled words
        :param x: string (paragraph/sentence) to be checked
        :return: list of tuples (p,q,r) where p denotes the position of the word that is misspelled (punctuations
        count as words, the first word is at location 0), q denotes the actual word that is misspelled, r represents a
        list of possible correct words that should replace the error word. Each list have at most 5 words.
        the words are ranked from the most likely to the least likely correction.

        """

        word_tokens = ['.']+nltk.word_tokenize(x.lower())
        tagged_words = nltk.pos_tag(word_tokens)
        lemmatized_words = [self.lemmatizer.lemmatize(word, self.get_wordnet_pos(tag)) for word, tag in tagged_words]
        answer = []
        updated_word_tokens = lemmatized_words
        for index, word in enumerate(lemmatized_words):
            if self.hasSpellingError(word):

                """
                currently using both bigrams and trigrams
                """
                key = tuple(updated_word_tokens[index-(self.numberOfGrams-1):index])
                possibleTriGramWords = []
                closeTriGramWordsWithDistance = []
                if key in self.trigrams:
                    possibleTriGramWords = self.trigrams[key]
                    closeTriGramWordsWithDistance = self.findNearestWords(possibleTriGramWords, word,
                                                                          tagged_words[index][1])

                key = tuple([updated_word_tokens[index - 1]])
                possibleBigramWords = self.bigrams[key]
                closeBiGramWordswithDistance = self.findNearestWords(possibleBigramWords, word, tagged_words[index][1])
                #check ngrams, edit distance and recommend words
                close_words = self.merge_words(closeTriGramWordsWithDistance, closeBiGramWordswithDistance)
                # print(close_words)
                if len(close_words) > 5:
                    replacements = [term for term in close_words[:5]]
                else:
                    replacements = [term for term in close_words]
                updated_word_tokens[index] = replacements[0]

                answer.append((index-1, word, replacements))
        print(answer)
        return answer

    def merge_words(self, trigrams, bigrams):
        """Merges the words suggested by trigrams and bigrams based on edit distance"""
        final_words = []
        for idx in range(self.maxDistance):
            dist = idx + 1
            trigram_for_dist = [term[0] for term in trigrams if term[1] == dist]
            final_words.extend(trigram_for_dist)
            bigram_for_dist = [term[0] for term in bigrams if term[1] == dist and term[0] not in final_words]
            final_words.extend(bigram_for_dist)
        return final_words

    """
    detects spelling error in the word
    """
    def hasSpellingError(self, word):
        if word in self.dictionaryWords:
            return False
        elif word in string.punctuation:
            return False
        elif self.stemmer.stem(word) + 'e' in self.dictionaryWords:
            return False
        else:
            return True

    """
    ranks the given wordlist based on the distances
    """
    def findNearestWords(self, wordlist, actualWord, tag):
        distance = []
        for term in wordlist:
            if term[0][0] not in string.punctuation:
                wordTowordDistance = nltk.edit_distance(actualWord, term[0].lower())
                lemma = self.lemmatizer.lemmatize(term[0].lower(), self.get_wordnet_pos(tag))
                lemmaTowordDistance = nltk.edit_distance(actualWord, lemma)
                if wordTowordDistance <= self.maxDistance:
                    distance.append([term[0].lower(), wordTowordDistance])
                elif lemmaTowordDistance <= self.maxDistance:
                    distance.append([lemma, lemmaTowordDistance])

        distance.sort(key=lambda x: x[1])
        return distance

    """
    loads the trained ngram model from the file
    """
    def loadNgramModelFromFile(self, n):
        try:
            with open(str(n)+'grams.txt', 'rb') as handle:
                return pickle.loads(handle.read())
        except FileNotFoundError:
            print("Error: Couldn't find the file for N-gram model. Please make sure to run nGramGenerate before this")
            sys.exit()

    @staticmethod
    def get_wordnet_pos(tag):
        """Takes the tag given by the default nltk tagger and converts it to wordnet tag
        :param tag: default tag
        :return: wordnet tag
        """
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN     # default pos in lemmatization is noun


spellcorrect = SpellCorrector()
spellcorrect.check("I was aple to sleep tonight. The water aill is thirty dollars. The "
                   "European Southern Observatory will release the first glimpse of a collapsed "
                   "btar in the center of our galaxy. Tomorrow is a brand ewnd day. The road lpeds to nowprae. John "
                   "kicks the uall to the brick uall.")
# spellcorrect.check("John kicks the uall to the brick uall. I play baln")
# spellcorrect.check("I word tonight in the office")
