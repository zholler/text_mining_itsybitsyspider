scraper.py:

Code to scrape data from the real estate website. It performs a seach for a given search term then extracts and saves the full html content of the resulting property advertisement pages. 

cleaning.py:

Cleans and organises the information from the htmls and saves the descriptions tokenized into the file description.txt and the tabular data in csv format to ingatlan_df.csv. Note that the stemming part of the data preparation is not done in python so the output of this code does not match with the input of the next code.

stemming:

Stemming is done separately using a natural language processing toolkit for hungarian text written in Java. For the deatils of the process see http://www.inf.u-szeged.hu/rgai/magyarlanc. 

The java code I run is the following:

java -Xmx1G -jar magyarlanc-3.0.jar -mode morphparse -input description.txt -output description_output.txt

analysis.py:

This files takes the stemmed descriptions (description_output.txt) and the tabular information (ingatlan_df.csv) and creates a corpus appropriate for the analysis. 

Then analysis is performed, results are printid and graphs are saved.  
