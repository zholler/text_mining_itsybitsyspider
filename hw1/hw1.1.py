# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 17:37:09 2016

@author: Zsuzsa, Anneke, Denitsa
"""
# load econ dictionary 
# format it
file_handle_econ = open('/home/didi/BGSE/semester3/text_mining_ta/spider/text_mining_itsybitsyspider/hw1/econ.csv')
file_content_econ = file_handle_econ.read()
file_content_econ = file_content_econ.lower()
import re
stripped_text_econ = re.sub(r'[^a-z\s]',"",file_content_econ)
stripped_text_econ = stripped_text_econ.split("\n")
#remove the last entry
del stripped_text_econ[-1]
# remove duplicates
stripped_text_econ = list(set(stripped_text_econ))
# we need to stem it
from nltk import PorterStemmer  
stem_econ = [PorterStemmer().stem(i) for i in stripped_text_econ]

# load military dictionary 
# format it
file_handle_mil= open('/home/didi/BGSE/semester3/text_mining_ta/spider/text_mining_itsybitsyspider/hw1/military.csv')
file_content_mil = file_handle_mil.read()
file_content_mil = file_content_mil.lower()

stripped_text_mil = re.sub(r'[^a-z\s]',"",file_content_mil)
stripped_text_mil = stripped_text_mil.split("\n")
#remove the last entry
del stripped_text_mil[-1]
# remove duplicates
stripped_text_mil = list(set(stripped_text_mil))
# we need to stem it
stem_mil = [PorterStemmer().stem(i) for i in stripped_text_mil]

#####################################################################################################################

#source a file
import HW1


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
    

text = open('/home/didi/BGSE/semester3/text_mining_ta/text_mining/data/pres_speech/sou_all.txt', 'r').read()
regex = "_(\d{4}).*?_[a-zA-Z]+.*?_[a-zA-Z]+.*?_([a-zA-Z]+)_\*+(\\n{2}.*?)\\n{3}"
pres_speech_list = parse_text(text, regex)

###########################################################################################################################################
##### Harvard IV dictionaries 

#pres_speech_list = pres_speech_list[0:2]

corpus = HW1.Corpus(pres_speech_list, '/home/didi/BGSE/semester3/text_mining_ta/text_mining/data/stopwords/stopwords.txt', 2)

# WITH Frequency count

econ_freq = corpus.dict_rank(stem_econ,10)
[(i.pres,i.year) for i in econ_freq]

mil_freq = corpus.dict_rank(stem_mil,10)
[(i.pres,i.year) for i in mil_freq]

# WITH IDF-TF matrix
econ_idf = corpus.dict_rank(stem_econ,10, False)
[(i.pres,i.year) for i in econ_idf]

mil_idf = corpus.dict_rank(stem_mil,10,False)
[(i.pres,i.year) for i in mil_idf]


####### AFINN dictionary 

 path = "/home/didi/BGSE/semester3/text_mining_ta/text_mining/data/AFINN/AFINN-111.txt"

    




