# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 16:21:02 2016

@author: zsuzsa
"""

import numpy as np
import sys
import re
import unicodedata
import pandas as pd
import codecs
import nltk
import math
from nltk.tokenize import wordpunct_tokenize
from nltk import PorterStemmer
from stop_words import get_stop_words
import matplotlib.pyplot as plt



#####################################################################################################
###Class definitions
#####################################################################################################

class Document():
    
    """ The Doc class rpresents a class of individul documents
    """   
    def __init__(self, description):
        """
        Input is the tokenized text as a list
        """
        self.length = len(description)
        self.tokens = np.array([word.lower() for word in description])


    def stopword_remove(self, stopwords):

        """
        description: Remove stopwords from tokens.
        input: stopwords: a suitable list of stopwords
        """

        
        self.tokens = np.array([t for t in self.tokens if t not in stopwords])


    def word_match(self, wordset):
        
        """
        description: return word count for a given set of words as a dictionary
        input: set of words (or possibly stemmed version of words) as a list or tuple
        """
        word_dict = {}
        for word in wordset:
            word_dict[word] = 0

        for word in self.tokens:
            if word_dict.get(word) != None:
                 word_dict[word] = word_dict[word] + 1
        return(word_dict)
 

class Corpus():
    
    """ 
    The Corpus class represents a document collection
     
    """
    def __init__(self, descriptions):

        #Initialise documents by invoking the appropriate class
        self.docs = [Document(description) for description in descriptions] 
        
        self.N = len(self.docs)
        
        #get a list of stopwords
        self.create_stopwords()
        
        #stopword removal, token cleaning and stemming to docs
        self.clean_docs(self.stopwords)
        
        #create vocabulary
        self.corpus_tokens()
          
    
    def create_stopwords(self):
        
        self.stopwords = get_stop_words('hungarian')       
        
    def clean_docs(self, length):
        """ 
        Applies stopword removal
        """
        for doc in self.docs:
            doc.stopword_remove(self.stopwords)    
     
    def corpus_tokens(self):
        """
        description: create a set of all all tokens or in other words a vocabulary
        """
        #initialise an empty set
        self.token_set = set()
        for doc in self.docs:
            self.token_set = self.token_set.union(doc.tokens)
        self.token_set = list(self.token_set)

    def document_term_matrix(self,wordset):
        """
        description: create a D by V array of frequency counts 
        note: order of both documents and words of the input are retained
        input: set of words (or possibly stemmed version of words) as a list or tuple
        """
        V = len(wordset)
        D = self.N
        matrix = np.empty([D,V])
        for doc , i  in zip(self.docs, range(len(self.docs))) :
            worddict = doc.word_match(wordset)
            for word, j in zip(wordset, range(len(wordset))):
                matrix[i,j] = worddict[word]
        return(matrix)
    
    def tf_idf(self,wordset):
        """
        description: create a D by V tf_idf array
        note: order of both documents and words of the input are retained
        input: set of words (or possibly stemmed version of words) as a list or tuple
        """
        matrix = self.document_term_matrix(wordset)
        nonzero_matrix = matrix > 0
        df = (nonzero_matrix).sum(axis = 0)
        idf = np.log( self.N / df )
        idf[np.isinf(idf)] = 0
        tf = (1 + np.log(matrix))
        tf[np.isinf(tf) ] = 0
        return(tf*idf)
    
    def dict_rank(self, wordset, n , freq_rep = True ):
        """
        description: returns the n most highly ranked documents in terms of frequency count 
        or tf-idf score for a given wordlist
        output: list of documents (document instances) ordered by rank
        input: wordset - set of words (or possibly stemmed version of words) as a list or tuple
        n - number of documents to be returned
        freq_rep - True means we use document_term_matrix, tf_idf otherwise
        """
        if freq_rep:
            matrix = self.document_term_matrix(wordset)
        else:
            matrix = self.tf_idf(wordset)
        matrix_total = matrix.sum(axis = 1)
        #duplicating an array
        ordering = np.array(matrix_total)  
        ordering.sort()
        doc_list = []
        for j in range(n):
            maxvalues = ordering[-(j+1)]           
            for doc,i in zip(self.docs, range(len(self.docs))):
                if matrix_total[i] == maxvalues:
                    doc_list.append(self.docs[i])    
        return(doc_list)
        

    def weighted_dict_rank(self, sentiment_dict, freq_rep = True ):
        """
        description: Uses an imported weighted dictionary to return a sentiment score for each document
        output: list of year, president, sentiment score tuples sorted by sentiment score
        input: sentiment_dict - dictionary of words (or possibly stemmed version of words) with sentiment scores
        freq_rep - True means we use document_term_matrix, tf_idf otherwise
        """
       
        wordset = sentiment_dict.keys()
        values = sentiment_dict.values()
        
        if freq_rep:
            matrix = self.document_term_matrix(wordset)
        else:
            matrix = self.tf_idf(wordset)
        
        # multiply values in matrix by valence score of each word in sentiment_dict     
        matrix_total = np.dot(matrix, values)
        
        # year, president, score (sorted by score). ie a list of tuples
        doc_list = []        
        for doc, i in zip(self.docs, range(len(self.docs))):
            doc_list.append((self.docs[i].year, self.docs[i].pres, matrix_total[i]))
        
        # sort list by sentiment score (descending)
        sorted_doc_list = sorted(doc_list, key=lambda x: x[2], reverse=True)
        return(sorted_doc_list)
        
####################################################From here#####################################################

path = "/home/zsuzsa/Documents/text_mining_2016/project/text/"
df_clean= pd.read_csv(path + 'ingatlan_df.csv', encoding = "utf-8")
#Read in tokenized descriptions (have to run tokenizer from terminal first)    
descriptions_tokenized = pd.read_csv(path+"/description_output.txt", sep = "\t", header = None , engine = "python")

#Subset only words of interest
condition = ((descriptions_tokenized.iloc[:,2] ==  "ADJ") | (descriptions_tokenized.iloc[:,2] ==  "ADV") | (descriptions_tokenized.iloc[:,2] ==  "NOUN") | (descriptions_tokenized.iloc[:,2] ==  "VERB") | (descriptions_tokenized.iloc[:,1] == "*****"))
descriptions_tokenized_2 = descriptions_tokenized[condition].copy()
descriptions_tokenized_2 = list(descriptions_tokenized_2.iloc[:,1])

#Remove NaNs which are words can not be identified by the tokenizer (e.g: ragok, ilyesmik)
#and deacode as unicode
descriptions_tokenized_2 = [ i.decode('utf8') for i in descriptions_tokenized_2 if not isinstance(i, float)]

#Remove some frequent but not interesting words
descriptions_tokenized_2 = [ i for i in descriptions_tokenized_2 if i.lower() not in [u'm',u'nm',u'ft',u'forint']]

#Cut the list into list of lists where on element is one description tokenized
def group(seq, sep):
    g = []
    for el in seq:
        if el == sep:
            yield g
            g = []
        g.append(el)
    yield g

descriptions_tokenized_2 = list(group(descriptions_tokenized_2, '*****'))[:-1]
descriptions_tokenized_2 = [descriptions_tokenized_2[0]] + [ i[1:] for i in descriptions_tokenized_2[1:] ]

corpus = Corpus(descriptions_tokenized_2)

########################################Analysis###################################################

doc_term_matrix = corpus.document_term_matrix(corpus.token_set)

#######Descriptives

words_freq = doc_term_matrix.sum(axis = 0)

words_freq.max()
most_freq_words = [ (token, words_freq[idx]) for idx, token in enumerate(corpus.token_set) if words_freq[idx] in np.sort(words_freq)[-50:]]

doc_len = doc_term_matrix.sum(axis = 1)
sum(doc_len == 0)
plt.hist(doc_len)

doc_term_corr = np.tril(np.corrcoef(doc_term_matrix),-1)

#histogram for cleaning 
doc_term_corr_hist = [i for i in doc_term_corr.flatten() if not math.isnan(i) and i != 0 ]
plt.hist(doc_term_corr_hist)

high_corr = []
for id1 in range(doc_term_corr.shape[0]):
    for id2 in range(doc_term_corr.shape[1]):
        if (doc_term_corr[id1,id2] > 0.7):
            high_corr.append((id1,id2,doc_term_corr[id1,id2]))

high_corr_plus = []
for i in range(len(high_corr)):
    id1 = high_corr[i][0]
    id2 = high_corr[i][1]
    if (df_clean["location"].iloc[id1] ==  df_clean["location"].iloc[id2]) & (df_clean["size"].iloc[id1] ==  df_clean["size"].iloc[id2]) & (df_clean["type"].iloc[id1] ==  df_clean["type"].iloc[id2]) :
        high_corr_plus.append(high_corr[i])


high_corr_groups = [[high_corr_plus[0][0], high_corr_plus[0][1]]]
for i in range(1,len(high_corr_plus)):
    for j in range(len(high_corr_groups)):
        if (high_corr_plus[i][0] in high_corr_groups[j]) or (high_corr_plus[i][1] in high_corr_groups[j]):
            high_corr_groups[j].append(high_corr_plus[i][0])
            high_corr_groups[j].append(high_corr_plus[i][1])
            high_corr_groups[j] = list(np.unique(high_corr_groups[j]))
            break
        if j == len(high_corr_groups)-1:
            high_corr_groups.append([high_corr_plus[i][0], high_corr_plus[i][1]])

to_drop = []
for i in range(len(high_corr_groups)):
    add = high_corr_groups[i][1:]
    if type(add) == int:   
        to_drop.append(add)
    else:
        to_drop = to_drop + add

#####More cleaning#####
df_clean.drop(df_clean.index[to_drop], inplace=True)
to_keep = [i for i in range(len(descriptions_tokenized_2)) if i not in to_drop]
descriptions_tokenized_3 =  [ descriptions_tokenized_2[i] for i in range(len(descriptions_tokenized_2)) if i not in to_drop]


############################################Final corpus############################

corpus = Corpus(descriptions_tokenized_3)

doc_term_matrix = corpus.document_term_matrix(corpus.token_set)

#######Descriptives

words_freq = doc_term_matrix.sum(axis = 0)

words_freq.max()
words_freq.min()
words_freq.mean()

most_freq_words = [ (token, words_freq[idx]) for idx, token in enumerate(corpus.token_set) if words_freq[idx] in np.sort(words_freq)[-10:]]

doc_len = doc_term_matrix.sum(axis = 1)
sum(doc_len == 0)
plt.hist(doc_len, bins = 200)
plt.savefig(path + 'length.png')

doc_term_corr = np.tril(np.corrcoef(doc_term_matrix),-1)

#histogram for cleaning 
doc_term_corr_hist = [i for i in doc_term_corr.flatten() if not math.isnan(i) and i != 0 ]
plt.hist(doc_term_corr_hist)

tf_idf_matrix = corpus.tf_idf(corpus.token_set)

tf_idf_corr = np.tril(np.corrcoef(tf_idf_matrix),-1)

       