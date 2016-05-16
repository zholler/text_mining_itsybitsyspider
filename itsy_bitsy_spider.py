#import modules 
import urllib2
from bs4 import BeautifulSoup as bs
import os
import numpy as np
import sys
import time, random
import json
import re


#function to read the url

def get_html(url):
    """
    This function gets the whole html content of a specific url
    """
    response = urllib2.urlopen(url) 
    html = response.read()
    return html 


def get_html_all(url,website_prefix):
    """
    This function goes to one of the result pages and takes the urls of the 
    results in the particular page. The output is a list with those urls.
    """
    response = urllib2.urlopen(url) #successful opening
    html = response.read()
    soup = bs(html,"lxml")
    links = soup.find_all('td',{"class": "address"})   
    urls = []
    for link in links:
        href= link.find('a').get('href')
        href = website_prefix + href 
        urls.append(href)
    return urls



def format_content(html):
    """
    """
    soup = bs(html,"html.parser")
    description_raw= soup.find('div',{"class": "long-description"})
    description = description_raw.get_text()  
    return description
 
 

def get_no_pages(first_url):
    """
    This function takes the first page of the search reasults and
    outputs number of properties found, number of result pages
    """
    response = urllib2.urlopen(first_url) 
    html = response.read()
    soup = bs(html,"lxml")
    no_pages_text = soup.find('li',{"class": "numbers"}).get_text()
    no_pages_list = re.sub('[^0-9]'," ",no_pages_text).split()
    no_objects = int(no_pages_list[0])
    no_pages = int(no_pages_list[2])
    return(no_objects,no_pages)
    
def create_page_urls(first_url,no_pages):
    """
    This function creates artificially a list of urls for 
    the different result pages 
    """
    urls = [first_url]
    for i in range((no_pages-1)):
        url = first_url+"?page="+str(i+2)
        urls.append(url)
    return(urls)


def scrape_urls(urls):

    property_urls = []
    #enumerate creates tuples     
    for idx, url in enumerate(urls):
        #random delay between 1 and 2 secs
        delay = random.randint(1,2)
        #pause execution for delay seconds
        time.sleep(delay)
        print 'file: ' + str(idx) + ' delay= ' + str(delay) 
        try:
            # extract the property urls
            property_urls = property_urls + get_html_all(url,"http://ingatlan.com")
            
            #add completed url to the log of completed urls
            # a stands for appending
            # open a portal             
            with open("./completed_full_urls.txt", "a") as complete_file:
                complete_file.write(url + '\n')
                #close the portal
                complete_file.close()
        except:
            #add rejected urls to the log of rejected urls
            # open a portal             
            with open("./rejected_full_urls.txt", "a") as rejected_file:
                rejected_file.write(url + '\n')
                #close the portal                
                rejected_file.close()
        
    return(property_urls)
        
  
  

def scrape(urls, file_prefix):

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
        
    
 

    
def main(main_url):
    """
    """
    # find number of properties and pages
    no_properties, no_pages = get_no_pages(main_url)
    # find the result pages urls
    result_page_urls = create_page_urls(main_url,no_pages)
    # make a list of all properties urls
    property_urls = scrape_urls(result_page_urls)
    # check if there are rejected urls and if there are STOP
    if len(property_urls) != no_properties or os.path.isfile("./rejected_full_urls.txt"):
        print("Error in extracting page urls")        
        return(property_urls)
    scrape(property_urls,"property")

#automate the function    

if __name__=="__main__":
    main(sys.argv[1])
    
        
       
    
    
#### Trial

#main_url = "http://ingatlan.com/lista/elado+lakas+bacs-kiskun-megye"

