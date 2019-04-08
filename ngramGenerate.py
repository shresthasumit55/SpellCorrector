import nltk
from nltk.corpus import state_union
from nltk.corpus import brown
from nltk.corpus import reuters
from nltk.corpus import gutenberg
from collections import Counter
from nltk import ngrams
import pickle


"""
This class generates the NGram Model and populates the file with probabilities of a word given its history ( for given
value of number of grams)
"""
class NGramModel:

    def __init__(self, corpus, n, maxword):
        if n < 1 or maxword < 1:
            raise Exception("Silly noodle, negatives aren't fun")

        #Special Word
        self.SPECIALWORD = "??"

        #corpus
        self.corpus = corpus

        #Number of Grams
        self.numberGrams = n

        #Number of words to consider
        self.maxNumberGrams = maxword

        #List of word in in corpus
        self.listOfTotalWords = self.corpus.words()
        self.listOfSecondCorpusWords = reuters.words()
        self.listofThirdCorpusWords = state_union.words()
        self.listofThirdCorpusWords = gutenberg.words()

        #Dictionary of Words
        self.dictionaryOfWords = Counter(self.listOfTotalWords)
        self.dictionaryOfWordsSecondCorpus = Counter(self.listOfSecondCorpusWords)

        #Dictionary of maxword most common
        self.listOfCommonWords = self.dictionaryOfWords.most_common(maxword)

        # Make sure our special word is special
        flag = True
        while flag:
            if self.SPECIALWORD in self.dictionaryOfWords:
                self.SPECIALWORD = self.SPECIALWORD + "?"
            else:
                flag = False


        #We have to use a dictionary here because a set cannot
        #house elements of length one, which, for example, a period can
        #be one of our most common words. Therefore we made a dictionary
        #with literally a dummy value
        self.dictOfCommonWords = dict()
        for word in self.listOfCommonWords:
            self.dictOfCommonWords[word.__getitem__(0)] = "dummy"


        #The new listing of words after we replace all the
        #undesired words
        self.newListingOfWords = ['.'] + self.listOfTotalWords + self.listOfSecondCorpusWords + self.listofThirdCorpusWords

        """
        for word in self.listOfTotalWords:
            
            if word in self.dictOfCommonWords:
                self.newListingOfWords.append(word)
            else:
                self.newListingOfWords.append(word)
                #self.newListingOfWords.append(self.SPECIALWORD)
        """

        #Length of newListingOfWords
        self.lengthOfNewListingOfWords = len(self.newListingOfWords)

        #The ngrams of the the new listing of words
        self.ngramsOfNewList = ngrams(self.newListingOfWords, self.numberGrams)

        #Dictionary of a number of Occurances for a particular gram
        self.numberOccurancesOfGrams = Counter(self.ngramsOfNewList)

        #get all the grams of smaller size
        self.gramsOfSmaller = ngrams(self.newListingOfWords, self.numberGrams - 1)

        #get the counter of the gramsOfSmaller
        self.gramsOfSmallerCounter = Counter(self.gramsOfSmaller)


    #Returns the special word used
    def special_word(self):
        return self.SPECIALWORD

    #returns the frequency of a gram in a corpus
    def freq(self, l):
        #make sure the list l has same legnth of corpus trained on
        if len(l) != self.numberGrams:
            return -1

        term = tuple(l)
        #If our gram is in the corpus
        if term in self.numberOccurancesOfGrams:
            return self.numberOccurancesOfGrams[term]
        else:
            return 0

    def prob(self, l):
        # make sure the list l has same legnth of corpus trained on
        if len(l) != self.numberGrams:
            return -1

        gram = tuple(l)

        #get the shorter list
        shorterList = l[:-1]

        #get the total amount of occurrences for that gram
        numTotalGram = self.numberOccurancesOfGrams[gram]

        smallGram = tuple(shorterList)

        #return the probability
        return numTotalGram / self.gramsOfSmallerCounter[smallGram]

    """
    This is the function which populates the files with ngrams in a dictionary format. 
    Key for the dictionary is the tuple of words(with len=number of grams -1) and 
    the value is the list of tuple, each of those tuples has the word that directly follows the key item and the number
    of times it appears after the key in the corpus
    
    Example "the green shirt" appears 2 times, "the green grass" appears 3 times in corpus. 
    For this example they key is ('the','green') and the value is a [('shirt',2),('grass',3)] 
    """
    def getGramsDictionary(self):
        allgrams=self.numberOccurancesOfGrams.most_common()
        gramDictionary=dict()
        for gram in allgrams:
            if tuple(gram[0][:-1]) not in gramDictionary:
                gramDictionary[tuple(gram[0][:-1])]=[]
            gramDictionary[tuple(gram[0][:-1])].append(list([gram[0][-1],gram[1]]))
        filename=str(self.numberGrams)+'grams.txt'
        with open(filename, 'wb') as handle:
            pickle.dump(gramDictionary, handle)
        print(self.SPECIALWORD)

        return gramDictionary

#Testing Driver
threeGramModel = NGramModel(brown, 3, 5000)
twoGramModel = NGramModel(brown,2,5000)

gramsDictionary = threeGramModel.getGramsDictionary()
twoGramsDict = twoGramModel.getGramsDictionary()

print("end")




