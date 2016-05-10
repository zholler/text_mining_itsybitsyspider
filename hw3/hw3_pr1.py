# -*- coding: utf-8 -*-

import numpy as np
import codecs
import nltk
import re
import math
from nltk.tokenize import wordpunct_tokenize
from nltk import PorterStemmer
import pandas as pd

#####################################################################################################
###Class definitions
#####################################################################################################

class Document():
    
    """ The Doc class rpresents a class of individul documents
    
    """
    
    def __init__(self, speech_year, speech_pres, speech_text):
        self.year = speech_year
        self.pres = speech_pres
        self.text = speech_text.lower()
        self.tokens = np.array(wordpunct_tokenize(self.text))
        
        
        
    def token_clean(self,length):

        """ 
        description: strip out non-alpha tokens and tokens of length > 'length'
        input: length: cut off length 
        """

        self.tokens = np.array([t for t in self.tokens if (t.isalpha() and len(t) > length)])


    def stopword_remove(self, stopwords):

        """
        description: Remove stopwords from tokens.
        input: stopwords: a suitable list of stopwords
        """

        
        self.tokens = np.array([t for t in self.tokens if t not in stopwords])


    def stem(self):

        """
        description: Stem tokens with Porter Stemmer.
        """
        
        self.tokens = np.array([PorterStemmer().stem(t) for t in self.tokens])


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
    def __init__(self, doc_data, stopword_file, clean_length):
        """
        Notice that the __init__ method is invoked everytime an object of the class
        is instantiated
        """
        

        #Initialise documents by invoking the appropriate class
        self.docs = [Document(doc[0], doc[1], doc[2]) for doc in doc_data] 
        
        self.N = len(self.docs)
        self.clean_length = clean_length
        
        #get a list of stopwords
        self.create_stopwords(stopword_file, clean_length)
        
        #stopword removal, token cleaning and stemming to docs
        self.clean_docs(2)
        
        #create vocabulary
        self.corpus_tokens()
        
    def clean_docs(self, length):
        """ 
        Applies stopword removal, token cleaning and stemming to docs
        """
        for doc in self.docs:
            doc.token_clean(length)
            doc.stopword_remove(self.stopwords)
            doc.stem()        
    
    def create_stopwords(self, stopword_file, length):
        """
        description: parses a file of stowords, removes words of length 'length' and 
        stems it
        input: length: cutoff length for words
               stopword_file: stopwords file to parse
        """
        
        with codecs.open(stopword_file,'r','utf-8') as f: raw = f.read()
        
        self.stopwords = (np.array([PorterStemmer().stem(word) 
                                    for word in list(raw.splitlines()) if len(word) > length]))
        
     
    def corpus_tokens(self):
        """
        description: create a set of all all tokens or in other words a vocabulary
        """
        #initialise an empty set
        self.token_set = set()
        for doc in self.docs:
            self.token_set = self.token_set.union(doc.tokens)

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
        
#####################################################################################################
###Load in presidental speech data
#####################################################################################################


def parse_text(textraw, regex):
    """takes raw string and performs two operations
    1. Breaks text into a list of speech, president and speech
    2. breaks speech into paragraphs
    """
    prs_yr_spch_reg = re.compile(regex, re.MULTILINE|re.DOTALL)
    
    #Each tuple contains the year, last ane of the president and the speech text
    prs_yr_spch = prs_yr_spch_reg.findall(textraw)
    
    #convert immutabe tuple to mutable list
    prs_yr_spch = [list(tup) for tup in prs_yr_spch]
    
    for i in range(len(prs_yr_spch)):
        prs_yr_spch[i][2] = prs_yr_spch[i][2].replace('\n', '')
    
    #sort
    prs_yr_spch.sort()
    
    return(prs_yr_spch)
        
#Load speeches and define the corpus
#text = open('/home/didi/BGSE/semester3/text_mining_ta/text_mining/data/pres_speech/sou_all.txt', 'r').read()
text = open('/Users/annekespeijers/Desktop/BGSE/Term3/TextMining/Course_repo/text_mining/data/pres_speech/sou_all.txt', 'r').read()
regex = "_(\d{4}).*?_[a-zA-Z]+.*?_[a-zA-Z]+.*?_([a-zA-Z]+)_\*+(\\n{2}.*?)\\n{3}"
pres_speech_list = parse_text(text, regex)
corpus = Corpus(pres_speech_list, '/Users/annekespeijers/Desktop/BGSE/Term3/TextMining/Course_repo/text_mining/data/stopwords/stopwords.txt', 2)

################################################################################################################
###### Uncollapesed Gibbs Sampling
################################################################################################################

## parameters 

# Number of topics 
# assuming Republicans and Democrats
k= 2 #Note: it is an integer

# number of terms in the word set
V = len(corpus.token_set)

# dirichlet parametrs
# for the mixing probabilities (T)
alpha = 1
alpha_vec = [float(alpha)]*k

# for the topic-specific term probabilities (B)
eta = 1
eta_vec = [float(eta)]*V

# number of documents 
D = corpus.N


## initialize the uniform vectors

# total number of words in the corpus 
wordset = list(corpus.token_set)
wordset_dict = {key: value for (key,value) in zip(wordset,range(V))}
term_matrix = corpus.document_term_matrix(wordset)
N = np.sum(term_matrix) # Note: it is a float number 

# initial topic assignment vector
z = np.random.choice(range(k), int(N), replace = True, p = [1/float(k)]*k)

# initialize the mixing probabilities
# it should be Dxk matrix 
T_all = []
T=np.random.dirichlet(alpha_vec,D)
T_all.append(T)

# initialize the topic-specific term probability 
B_all = []
B = np.random.dirichlet(eta_vec,k)
B_all.append(B)

## The Gibbs sampling loop
# number of iterations 
S = 3
# initialize empty nested lists for topic probabilities and allocations
topics = []
topic_allocation = []
word_n = np.zeros(D)
# we simulatenously count the number of words in a doc and initialize an empty 
# array where later we will put the topic allocation for each word
for d in range(D):
    word_n[d] = sum(term_matrix[d])
    topic = np.zeros((int(word_n[d]),k))
    topics.append(topic)
    topic_allocation.append( np.zeros( int(word_n[d]) ))
    

#for updating the Dirichlet distribution T 
n = np.zeros((D,k))
#for updating the Dirichlet distribution B
m = np.zeros((k,V))

# Match the elements of each document to the terms in our wordset
# each element in the word_term_matrix tells us which element this word 
#corresponds to in the wordset. ie which v it is.
word_term_matrix = list()
for d in range(D):
    word_term_matrix_doc = np.zeros(int(word_n[d]))
    for word in range(len(corpus.docs[d].tokens)):
        word_term_matrix_doc[word] = wordset_dict[corpus.docs[d].tokens[word]]
    word_term_matrix.append(word_term_matrix_doc)


# Run the iterations
for i in range(S):
    print("iteration",i)
    # take the appropriate betas and thetas    
    B = B_all[i]
    T = T_all[i]
    print(T[0])
    
    # Matrix of normalising constants for the multinomial latent distribution
    normalizing_const = np.dot(T,B)    
    
    # loop through each word of each document and calculate, topic allocation, n and m
    for d in range(D):
        N_doc = int(word_n[d])
        
        for word in range(N_doc):
            #finding the term for the word 
            v = int(word_term_matrix[d][word])
            
            for j in range(k):
                numerator = T[d][j]*B[j][v]
                denominator = normalizing_const[d][v]                
                #probabilities of topic allocation              
                topics[d][word][j] = numerator / denominator
            
            # the actual topic allocation per word    
            topic_allocation[d][word] = np.random.choice(a=k, size=1, p=topics[d][word])
            
            # Update m (number of times a word is generated by a topic)
            m[topic_allocation[d][word]][v] += 1
        
        # Calculate n(number of words in a particular topic)
        for j in range(k):
            n[d][j] = sum(topic_allocation[d] == j)

            
    # update the dirichlet constants     
    eta_vec = (eta_vec + m) 
    alpha_vec = (alpha_vec + n)
    
    T_new = np.zeros(T.shape)    
    for d in range(D):
        T_new[d] = np.random.dirichlet(alpha_vec[d],1)
    T_all.append(T_new)
    print(T_new[0])
    
    B_new = np.zeros(B.shape)
    for v in range(V):
        B_new[:,v] = np.random.dirichlet(eta_vec[:,v],1)
    B_all.append(B_new)
    
    

    
    







