import urllib2
from bs4 import BeautifulSoup as bs
import os
import numpy as np
import sys
import time, random
import json

url = "http://www.urbandictionary.com/"
response = urllib2.urlopen(url) #successful opening
text = response.read()
html = bs(text,"html.parser")
all_panels = html.findAll("div",{"class" : "def-panel"})
#print(all_panels[0])

first = all_panels[0]
word = first.find("a",{"class" : "word"})
word = word.get_text()
print()

