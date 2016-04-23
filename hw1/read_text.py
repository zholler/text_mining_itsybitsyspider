# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 18:36:26 2016

@author: zsuzsa
"""

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
    
import re
text = open('/home/didi/BGSE/semester3/text_mining_ta/text_mining/data/pres_speech/sou_all.txt', 'r').read()
regex = "_(\d{4}).*?_[a-zA-Z]+.*?_[a-zA-Z]+.*?_([a-zA-Z]+)_\*+(\\n{2}.*?)\\n{3}"
pres_speech_list = parse_text(text, regex)
pres_speech_list = pres_speech_list[0:2]

corpus = Corpus(pres_speech_list, '/home/didi/BGSE/semester3/text_mining_ta/text_mining/data/stopwords/stopwords.txt', 2)


