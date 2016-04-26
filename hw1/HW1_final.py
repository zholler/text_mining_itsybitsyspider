import numpy as np
import codecs
import nltk
import re
import math
from nltk.tokenize import wordpunct_tokenize
from nltk import PorterStemmer
import pandas as pd

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
 

###################################################################################################

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
      
            

###################################################################################################

def read_dictionary(path):
    '''
    Read in and format and stem dictionary
    output: list of stemmed words
    '''
    file_handle = open(path)
    file_content = file_handle.read()
    file_content = file_content.lower()
    stripped_text = re.sub(r'[^a-z\s]',"",file_content)
    stripped_text = stripped_text.split("\n")
    #remove the last entry
    del stripped_text[-1]
    # remove duplicates
    stripped_text = list(set(stripped_text))
    # we need to stem it
    stemmed = [PorterStemmer().stem(i) for i in stripped_text]
    return(stemmed)
    
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
    
def convert_to_dictionary(file):
        """
        convert the afinn text file to dictionary
        Note: all dictionary entries are lower case 
        Line_split is tab
        entries in the afinn list are stemmed and the average valence score is taken
        """
        #Open file with proper encoding
        with codecs.open(file,'r','utf-8') as f: row_list = [ line.split('\t') for line in f ]
        dict_elements = [ ( PorterStemmer().stem(word[0]) , int(word[1]) ) for word in row_list ]
        
        #Take unique elements after stemming
        #problem: we have words with the same root and diff. score - solution: average scores
        
        # turn into pandas dataframe
        dict_elements_df = pd.DataFrame(dict_elements, index = range(len(dict_elements)), columns=['stem', 'value']) #2477 entries
        
        # group by stemmed words and average
        grouped = dict_elements_df.groupby('stem', as_index=False)
        dict_elements_agg = grouped.aggregate(np.mean) #1482 entries
    
        # turn pandas df back into dictionary
        dict_stems_averaged = dict_elements_agg.set_index('stem')['value'].to_dict()
        
        return(dict_stems_averaged)    
        
###################################################################################################
##### Harvard IV dictionaries 

### Load in text and dictionaries
#Load speeches and define the corpus
text = open('../python_intro/sou_all.txt', 'r').read()
regex = "_(\d{4}).*?_[a-zA-Z]+.*?_[a-zA-Z]+.*?_([a-zA-Z]+)_\*+(\\n{2}.*?)\\n{3}"
pres_speech_list = parse_text(text, regex)
corpus = Corpus(pres_speech_list, 'stopwords.txt', 2)

# load econ dictionary and format it
stem_econ = read_dictionary('econ.csv')

# load military dictionary and format it
stem_mil = read_dictionary('military.csv')


# WITH Frequency count

#Extract the top 10 documents using the economics dictionary
econ_freq = corpus.dict_rank(stem_econ,10)
#Print top 10 president - year combinations orderes by score 
print([(i.pres,i.year) for i in econ_freq])

#Extract the top 10 documents using the military dictionary
mil_freq = corpus.dict_rank(stem_mil,10)
#Print top 10 president - year combinations orderes by score 
print([(i.pres,i.year) for i in mil_freq])

# WITH IDF-TF matrix

#Extract the top 10 documents using the economics dictionary
econ_idf = corpus.dict_rank(stem_econ,10, False)
#Print top 10 president - year combinations orderes by score 
print([(i.pres,i.year) for i in econ_idf])

#Extract the top 10 documents using the military dictionary
mil_idf = corpus.dict_rank(stem_mil,10,False)
#Print top 10 president - year combinations orderes by score 
print([(i.pres,i.year) for i in mil_idf])

######################################################################################################
#Load and format affin word dictionary
afinn_dict = convert_to_dictionary("AFINN-111.txt")

# 4. Analyse speeches using AFINN-111 weighted dictionary
#Using simple word counts as weights
sentiment_rank1 = corpus.weighted_dict_rank(afinn_dict)

#Using tf-idf as weights
sentiment_rank2 = corpus.weighted_dict_rank(afinn_dict,freq_rep = False )

