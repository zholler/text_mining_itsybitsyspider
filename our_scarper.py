#import modules 
import urllib2
from bs4 import BeautifulSoup as bs
import os
import numpy as np
import sys
import time, random
import json

#function to read the url

def get_html(url):
    response = urllib2.urlopen(url) #successful opening
    text = response.read()
    return text

#define function to convert to stuctured object

def format_content(text):
    html = bs(text,"html.parser")
    all_panels = html.find_all(class="def-panel")
    for panel in range(len(all_panels)):
        
    

