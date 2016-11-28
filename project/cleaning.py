# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 20:45:16 2016

@author: zsuzsa
"""
#import modules 
from bs4 import BeautifulSoup as bs
import os
import numpy as np
import sys
import re
import unicodedata
import cPickle as pickle
import json
import pandas as pd
import codecs
import nltk
import math
from nltk.tokenize import wordpunct_tokenize
from nltk import PorterStemmer
from stop_words import get_stop_words
import matplotlib.pyplot as plt


def remove_accents(input_str):
    """
    input: unicode string
    output: unicode string without accents
    """
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

def format_content(html):
    """
    """
    soup = bs(html,"lxml")
    description= soup.find('div',{"class": "long-description"})
    description = description.get_text()
    
    size= soup.find('div',{"class": "parameter parameter-area-size"})
    size = size.find('span',{"class": "parameter-value"}).get_text()
    size = int(re.sub('[^0-9]',"",size))
    
    lot_size= soup.find('div',{"class": "parameter parameter-lot-size"})
    if lot_size is None:
        lot_size = None
    else:
        lot_size = lot_size.find('span',{"class": "parameter-value"}).get_text()
        lot_size = int(re.sub('[^0-9]',"",lot_size))
    
    rooms = soup.find('div',{"class": "parameter parameter-room"})
    if rooms is None:
        rooms = None
        rooms_half = None
    else:
        rooms = rooms.find('span',{"class": "parameter-value"}).get_text()
        rooms = re.sub('[^0-9]'," ",rooms).split()    
        if len(rooms) == 2:
            rooms_half = int(rooms[1])    
        else:
            rooms_half = 0
        rooms = int(rooms[0])
    
    price = soup.find('div',{"class": "parameter parameter-price"})
    price = price.find('span',{"class": "parameter-value"}).get_text()
    price = price.split()
    price_num = float(re.sub(",",".",price[0]))
    price_unit = price[1]
    price_currency = price[2]
    
    card = soup.find('div',{"class": "card details"})
    
    short_desc = card.find('div',{"class": "card-title"}).get_text()
    short_desc_list = short_desc.split()
    if len(short_desc_list) < 5:
        house = short_desc_list[1:] 
        house = " ".join(house)
    else: #berleti jog eledo - remove
        house = None
    
    location = soup.find('header',{"class": "listing-header"}).find('h1').get_text()
    
    parameters =  card.find('div',{"class": "paramterers"})
    data = []    
    tables = parameters.find_all('table')
    for table in tables:
        rows = table.find_all('tr')
        for row in rows:
            cols = row.find_all('td')
            cols = [ele.text.strip() for ele in cols]
            cols[0] = remove_accents(cols[0].lower()).encode('ascii"')
            cols = tuple(cols)
            data.append([ele for ele in cols if ele]) # Get rid of empty values
    
    dictionary = {"type": house, 
                  "location": location,
                  "size": size,
                  "lot_size": lot_size,
                  "rooms": rooms,
                  "rooms_half": rooms_half,
                  "price": price_num,
                  "price_unit": price_unit,
                  "price_currency": price_currency,
                  "description": description}
    dictionary.update(data)
    
    erkely = dictionary.get(u"erkely")
    if erkely is not None:
        dictionary[u"erkely_size"] = float(re.sub('[^0-9.]',"",erkely))
        del dictionary[u"erkely"]
    
    return dictionary


def load_and_format(path):
#Load pkl files of raw html into one list and format them into a dictionary
    files_to_load = [f for f in os.listdir(path) if f[-4:] == ".pkl" ]    
    information_all = []
    #Save html for which code did not run   
    problem = []
      
    for fi in files_to_load:  
   
        property_info = pickle.load(open(fi,'rb'))
        for html in property_info:
            try:
                dictionary = format_content(html)
                information_all.append(dictionary)
            except:
                problem.append(html)            
    with open(path + '/info_all.txt', 'w') as outfile:
        json.dump(information_all, outfile)
    outfile.close()

    with open(path + '/problem.pkl', 'w') as outfile:
        pickle.dump(problem, outfile,protocol=pickle.HIGHEST_PROTOCOL)   
    outfile.close()


path = "/home/zsuzsa/Documents/text_mining_2016/project/text"
load_and_format(path)

#Load in dictionary and problematic html
information_all = json.loads(open(path + '/info_all.txt').read())
problem = pickle.load(open(path + '/problem.pkl','rb'))

################################ Ad hoc cleaning#######################################
#berleti jog eledo - remove (these were added as None)
information_all = [dictionary for dictionary in information_all if dictionary["type"] is not None ]

#Put into pandas dataframe
df = pd.DataFrame(information_all)

#clean duplicates
duplicates = df.duplicated(subset = "description")
df_duplicates = df.loc[duplicates].copy()
df_duplicates["description"]

df_clean = df.drop_duplicates(subset = "description", keep = "first") #125 rows dropped

duplicates = df_clean.duplicated(subset = df.columns.values[df.columns.values != "description"], keep = False)
df_duplicates = df_clean.loc[duplicates].copy()
df_duplicates["description"]


df_clean.to_csv(path + '/ingatlan_df.csv', encoding = "utf-8", index=False)


#Extract decriptions
descriptions = list(df_clean["description"].values)
#remove numbers
descriptions = [re.sub(r'[0-9]',"",description) for description in descriptions]
#remove - and +
descriptions = [re.sub(r'[-+]'," ",description) for description in descriptions]
#remove urls
webpage = re.compile('https?://[^/]+(/(\?.*|index\.php(\?.*)?)?)?')
descriptions = [re.sub(webpage," ",description) for description in descriptions]

#Write descriptions to file
text_file = open(path + "/description.txt", "w")
for idx, description in enumerate(descriptions):
    text = description.encode('utf8') + "*****"
    print>>text_file, text
text_file.close()      




    
    
    