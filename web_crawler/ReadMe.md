## Web crawler for ingatlan.com; Hungarian real estate website 

The input to the function is the url of the first page returned from a search at ingatlan.com using its in-built search engine. 

* The scraper first extracts the number of pages returned from the search, along with the total number of properties found in these pages, from the html of the first page.
* Next, it goes to each of these pages and scrapes the urls of the individual property pages (~20 per page). Logs are kept here for which result page urls the crawler could access and those it couldn't access. These are saved in files "completed_full_urls.txt" and "rejected_full_urls.txt", respectively.
* It then checks whether the number of urls extracted equals the number displayed on the first result page and if we have rejected urls. If either of these are true, the scraping process is stopped.
* Otherwise, we take the extracted property urls, and scrape the entire html content for each of them. This html contains the description of the properties found in the search. 
* Here logs are also kept to keep track of which property urls were able to be scraped and which were not. They are saved in files "completed_urls.txt" and "rejected_urls.txt" respectively.
* Every 100 properties, the html is written to disk. The html of these properties is saved as one long string.
* The scraper has been tested on a search that produces one page with 18 properties. It can be run from the command line using the following code:

python itsy_bitsy_spider.py "http://ingatlan.com/szukites/elado+lakas+u:Kmety_Gy%C3%B6rgy_utca|136516|136517"

Note. At the time of writing this url worked, however if any problems arise, go to ingatlan.com, run a search that will produce a narrow result set and use the url from the first results page. 
