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
    html = response.read()
    return html 


def get_html_all(url):
    response = urllib2.urlopen(url) #successful opening
    html = response.read()
    soup = bs(html,"lxml")
    links = soup.find_all('td',{"class": "address"})   
    urls = []
    for link in links:
        href= link.find('a').get('href')    
        urls.append(href)
    return urls




#define function to convert to stuctured object

def format_content(html):
    soup = bs(html,"html.parser")
    description_raw= soup.find('div',{"class": "long-description"})
    description = description_raw.get_text()  
    return description
 

##### PAGE LOOP

a = get_html_all(main_url)   

#### Safety logs    



# define the scaper loop

def scrape_contracts(urls, file_prefix):
    #Uses a list of the xml url provided to scrape data from them and store to disk periodically
    #Writes various logs to monitor progress
    #input: xml_urls - a list of xml urls
    #output - None
    project_info = []
    
    for idx, url in enumerate(urls):
        #random delay between 1 and 2 secs
        delay = random.randint(1,2)
        #pause execution for delay seconds
        time.sleep(delay)
        print 'file: ' + file_prefix + str(idx) + ' delay= ' + str(delay) 
        try:
            #try to retrieve information from url
            project_info.append(get_html(url))
            
            #add completed url to the log of completed urls
            # a stands for appending
            # open a portal             
            with open("./completed_urls.txt", "a") as complete_file:
                complete_file.write(url + '\n')
                #close the portal
                complete_file.close()
        except:
            #add rejected urls to the log of rejected urls
            # open a portal             
            with open("./rejected_urls.txt", "a") as rejected_file:
                rejected_file.write(url + '\n')
                #close the portal                
                rejected_file.close()
        
        if idx % 2 == 0 and idx != 0: #modular division 
            #periodically write the data to file and reinitialise list for memory management
            file_name = file_prefix + str(idx) + '.gz'
            np.savetxt(file_name, project_info, delimiter=',', fmt='%s')
            project_info = []
            
            #add the index of last file to be written to disk
            with open("./saved_data_index.txt", "a") as saved_file:
                saved_file.write(file_prefix + str(idx) + '\n')
                saved_file.close()
    
    #Save remaining data to file    
    file_name = file_prefix + str(idx) + '.gz'
    np.savetxt(file_name, project_info, delimiter=',', fmt='%s')
    with open("./saved_data_index.txt", "a") as saved_file:
                saved_file.write(file_prefix + str(idx) + '\n')
                saved_file.close()
        
    
    
    
    
#### Trial

url1 = "http://ingatlan.com/kiskunhalas/elado+lakas/tegla-epitesu-lakas/bacs-kiskun+megye+kiskunhalas+esze+tamas+lakotelep/20421599"
url2 = "http://ingatlan.com/janoshalma/elado+lakas/tegla-epitesu-lakas/bacs-kiskun+megye+janoshalma+bajai+utca/5870111"
url3 = "http://ingatlan.com/kalocsa/elado+lakas/tegla-epitesu-lakas/bacs-kiskun+megye+kalocsa+obermayer+erno+ter/22246957"
url4 = "http://ingatlan.com/lajosmizse/elado+lakas/tegla-epitesu-lakas/bacs-kiskun+megye+lajosmizse+szent+imre+ter/22259702"

main_url = "http://ingatlan.com/lista/elado+lakas+bacs-kiskun-megye"

urls = [url1,url2,url3,url4]
tml_text = get_html(url)
desc = format_content(html_text)
