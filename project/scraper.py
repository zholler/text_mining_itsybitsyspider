#import modules 
import urllib2
from bs4 import BeautifulSoup as bs
import os
import numpy as np
import sys
import time, random
import re
import unicodedata
import cPickle as pickle


#function to read the url
def get_html(url):
    """
    This function gets the whole html content of a specific url
    """
    response = urllib2.urlopen(url) 
    html = response.read()
    return html 

def remove_accents(input_str):
    """
    input: unicode string
    output: unicode string without accents
    """
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])
    

def create_page_urls(first_url,no_pages):
    """
    input: url of the first result page and number of results pages for that search
    output: list of urls for the different result pages 
    """
    urls = [first_url]
    for i in range((no_pages-1)):
        url = first_url+"?page="+str(i+2)
        urls.append(url)
    return(urls)
    
def create_first_page_urls(search_terms, lakas = True):
    """
    input: search terms (unicode) - currently search terms are supposed to be city names
    + flat or house (to be done: more complex search terms are possible in principal) 
    output: url of first page of results for the given search
    """  
    urls = []
    for term in search_terms:
        term = term.lower()
        term = remove_accents(term)
        if lakas:
            term = "elado+lakas+" + term
        else:
            term = "elado+haz+" + term
        url = "http://ingatlan.com/lista/" + term
        urls.append(url)
    return(urls)

 
def result_page_error(soup,search_term):
    '''
    input: soup object of result page and search term used
    output: 0 - result page is OK
            1 - search is non existing (e.g.: city name is not in ingatlan.com)
            2 - existing search but no property found
            3 - http error
            null - none of these
    '''
    page_title = soup.find('h1',{"id": "page-title"}).get_text()
    if page_title is not None:
        page_title = page_title.lower()
        page_title = remove_accents(page_title)
        search_term = search_term.lower()
        search_term = remove_accents(search_term)
        #result page seems to be ok
        if search_term in page_title:
            return(0)
    #search is non existing (e.g.: city name is not in ingatlan.com)    
        elif page_title == u"elado hazak az orszag egesz teruleten" :
            return(1)
    #existing search but no property found
    noresult = soup.find('div',{"id": "search-results-noresults"})
    if noresult is not None:
        return(2)
    #http error
    httperror = soup.find('div',{"id": "httperror"})
    if httperror is not None:
        return(3)
    return
    
    
def property_page_error(soup):
    '''
    input: soup object of property_page
    output: 0 - no http error
            1 - http error
    '''
    #http error
    httperror = soup.find('div',{"id": "httperror"})
    if httperror is not None:
        return(1)
    else:
        return(0)
    

def scrape_result_pages(first_urls, search_terms):
    """
    input: list of first page urls and corresponding search terms 
    output: list of result page urls, number of properties 
    #and search term extended list (for checking the rest of the result pages later)
    log: non empty first page urls, empty first page urls, rejected first page urls
    """
    result_page_urls = []
    no_properties = 0
    search_terms_new = []
    
    for idx, url in enumerate(first_urls):
        #random delay between 1 and 2 secs
        delay = random.randint(1,2)
        #pause execution for delay seconds
        time.sleep(delay)
        print url + ' delay= ' + str(delay) 
        try:
            response = urllib2.urlopen(url)
            html = response.read()
            soup = bs(html,"lxml")
            #Check if there is any property found for the given search
            page_error = result_page_error(soup,search_terms[idx])
            #if yes: add all corresponding result page urls to list
            if  page_error == 0:  
                #extract number of pages and properties found
                no_pages_text = soup.find('li',{"class": "numbers"}).get_text()
                no_pages_text = re.sub(u'\xa0',"",no_pages_text)
                no_pages_list = re.sub('[^0-9]'," ",no_pages_text).split()
                no_properties = no_properties  + int(no_pages_list[0])
                no_pages = int(no_pages_list[2])
                #add corresponding urls
                pages = create_page_urls(url,no_pages)
                result_page_urls = result_page_urls + pages
                #extend search term list
                search_terms_new = search_terms_new + [search_terms[idx]]*len(pages)
                with open("./non_empty_first_page_urls.txt", "a") as file1:
                    file1.write(url + '\n')
                    #close the portal
                    file1.close()
            #if search is non existing (e.g.: city name is not in ingatlan.com)
            elif page_error == 1 :
                with open("./non_existing_first_page_urls.txt", "a") as file3:
                    file3.write(url + '\n')
                    #close the portal
                    file3.close()
            #if no result found (existing search but no property found)
            elif page_error == 2:
                with open("./empty_first_page_urls.txt", "a") as file2:
                    file2.write(url + '\n')
                    #close the portal
                    file2.close()
            elif page_error == 3:
                with open("./http_error_first_page_urls.txt", "a") as file4:
                    file4.write(url + '\n')
                    #close the portal
                    file4.close()
            else:
                with open("./unidentified_error_first_page_urls.txt", "a") as file5:
                    file5.write(url + '\n')
                    #close the portal
                    file5.close()
                
            
        except:
            #add rejected urls to the log of rejected urls
            # open a portal             
            with open("./rejected_first_page_urls.txt", "a") as rejected_file:
                rejected_file.write(url + '\n')
                #close the portal                
                rejected_file.close()
        
    return(result_page_urls, search_terms_new, no_properties)
        
    
def scrape_property_urls(urls, search_terms_new):
    '''
    input: list of result page urls and correponding search terms (extended list)
    output: urls of properties
    log: completed vs rejected results page urls
    '''

    property_urls = []
    #enumerate creates tuples     
    for idx, url in enumerate(urls):
        #random delay between 1 and 2 secs
        delay = random.randint(1,2)
        #pause execution for delay seconds
        time.sleep(delay)
        print url + ' delay= ' + str(delay) 
        try:
            response = urllib2.urlopen(url) #successful opening
            html = response.read()
            soup = bs(html,"lxml")
            page_error = result_page_error(soup,search_terms_new[idx])
            #if yes: add all corresponding result page urls to list
            if  page_error == 0:  
                #find proerty links
                links = soup.find_all('td',{"class": "address"})   
                urls = []
                for link in links:
                    href= link.find('a').get('href')
                    href = "http://ingatlan.com" + href 
                    urls.append(href)
                # extract the property urls
                property_urls = property_urls + urls
                
                with open("./non_empty_result_page_urls.txt", "a") as file1:
                    file1.write(url + '\n')
                    #close the portal
                    file1.close()
            #if search is non existing (e.g.: city name is not in ingatlan.com)
            elif page_error == 1 :
                with open("./non_existing_result_page_urls.txt", "a") as file3:
                    file3.write(url + '\n')
                    #close the portal
                    file3.close()
            #if no result found (existing search but no property found)
            elif page_error == 2:
                with open("./empty_result_page_urls.txt", "a") as file2:
                    file2.write(url + '\n')
                    #close the portal
                    file2.close()
            elif page_error == 3:
                with open("./http_error_result_page_urls.txt", "a") as file4:
                    file4.write(url + '\n')
                    #close the portal
                    file4.close()
            else:
                with open("./unidentified_error_result_page_urls.txt", "a") as file5:
                    file5.write(url + '\n')
                    #close the portal
                    file5.close()
            

        except:
            #add rejected urls to the log of rejected urls
            # open a portal             
            with open("./rejected_result_page_urls.txt", "a") as rejected_file:
                rejected_file.write(url + '\n')
                #close the portal                
                rejected_file.close()
        
    return(property_urls)
          

def scrape(urls, file_prefix):
    '''
    input: property urls
    output: html content saved to .gz file, new file after every 100 property
    logs: completed property urls, http error property urls, rejected property urls, saved data indeces
    '''
    property_info = []
    
    for idx, url in enumerate(urls):
        #random delay between 1 and 2 secs
        delay = random.randint(1,2)
        #pause execution for delay seconds
        time.sleep(delay)
        print file_prefix + str(idx) + ' delay= ' + str(delay) 
        try:
            #try to retrieve information from url
            html = get_html(url)
            soup = bs(html,"lxml")
            page_error = property_page_error(soup)
            if page_error ==0:
                property_info.append(html)            
                #add completed url to the log of completed urls
                # a stands for appending
                # open a portal             
                with open("./completed_property_urls.txt", "a") as complete_file:
                    complete_file.write(url + '\n')
                    #close the portal
                    complete_file.close()
            else: 
                with open("./http_error_property_urls.txt", "a") as file:
                    file.write(url + '\n')
                    #close the portal
                    file.close()
        except:
            #add rejected urls to the log of rejected urls
            # open a portal             
            with open("./rejected_property_urls.txt", "a") as rejected_file:
                rejected_file.write(url + '\n')
                #close the portal                
                rejected_file.close()
        
        if idx % 100 == 0 and idx != 0: #modular division 
            #periodically write the data to file and reinitialise list for memory management
            file_name = file_prefix + str(idx) + '.pkl'
            #np.savetxt(file_name, property_info, delimiter=',', fmt='%s')
            
            output = open(file_name, 'wb')
            pickle.dump(property_info, output,protocol=pickle.HIGHEST_PROTOCOL)
            output.close()

            property_info = []
            
            #add the index of last file to be written to disk
            with open("./saved_data_index.txt", "a") as saved_file:
                saved_file.write(file_prefix + str(idx) + '\n')
                saved_file.close()
    
    #Save remaining data to file    
    file_name = file_prefix + str(idx) + '.pkl'
    #np.savetxt(file_name, property_info, delimiter=',', fmt='%s')
    
    output = open(file_name, 'wb')
    pickle.dump(property_info, output,protocol=pickle.HIGHEST_PROTOCOL)
    output.close()
    
    with open("./saved_data_index.txt", "a") as saved_file:
                saved_file.write(file_prefix + str(idx) + '\n')
                saved_file.close()
        
    
 

    
def main(search_terms):
    """
    input: search terms (list of city names currently in unicode format)
    output: full html content of property pages saved
    """
    first_urls = create_first_page_urls(search_terms, lakas = False)
    
    result_page_urls, search_terms_new, no_properties = scrape_result_pages(first_urls, search_terms)
    
    property_urls = scrape_property_urls(result_page_urls, search_terms_new)
    
    # check if there are rejected urls and if there are STOP
    if len(property_urls) != no_properties: #or os.path.isfile("./rejected_full_urls.txt"):
        print("Error in extracting page urls")        
        return(property_urls)
    
    scrape(property_urls,"szeged_haz")

#automate the function    

if __name__=="__main__":
    main(sys.argv[1])
    
        
       
    
    
#### Trial
#http://ingatlan.com/lista/elado+lakas+veresegyhaz
#main_url = "http://ingatlan.com/lista/elado+lakas+bacs-kiskun-megye"

