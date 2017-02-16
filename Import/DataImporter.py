import os # operation system
from collections import namedtuple
import json
import plotly.offline as offline
import plotly.plotly as py
import plotly.tools as pt
from plotly.tools import FigureFactory as FF
from tinydb import TinyDB, Query
crawlerPath = "/home/retkowski/Crawler/"

# NYT Corpus
import tarfile,os
import sys
import gzip
from contextlib import closing
from xml.etree import ElementTree as etree
import re


class DataImporter():
    def __init__(self, path = '/home/retkowski/Data/newsDB.json' ):
        self._NewsCollection = []
        self._db = TinyDB(path)
        self._News = namedtuple("News",["title","site","tags","text","abstract","url"])
        print("WARNING !!! DEPRICATED !!! WARNING !!! DEPRICATED !!! WARNING !!!")

    #def _updateCrawler(self):
        #!cd $crawlerPath && git stash && git pull
    
    def _writeIntoDatabase(self):
        path = crawlerPath+"data"
        sites = [site[:-5] for site in os.listdir(path) if os.path.isfile(os.path.join(path, site)) and not site[0]== "."]
        print("Reading sites:", sites)

        for site in sites:
            with open(path+"/"+site+".json") as dl:
                data = json.load(dl)
            for link in data:
                if link["tags"] is not None:
                    self._db.insert({'title'    : link["title"],
                               'url'      : link["url"],
                               'site'     : site,
                               'tags'     : link["tags"],
                               'text'     : link["text"],
                               'abstract' : link["abstract"]})
    
    def _buildDownloadList(self):
        with open(crawlerPath + "download-all.txt") as dl:
            downloadlist = json.load(dl)
    
        newDownloadList = []
        for dlsite in downloadlist:
            siteDownload = []
            for link in dlsite["links"]:
                if db.get( where("url") == link["url"]) == None:
                    siteDownload.append(link)
            newDownloadList.append({"name" : dlsite["name"], "links" : siteDownload})

        with open(crawlerPath + 'download.txt', 'w') as outfile:
            json.dump(newDownloadList, outfile)
    
    #def _runCrawler(self):
        #self._buildDownloadList()
        #self._updateCrawler()
        #!cd $crawlerPath && ./run-crawlers.sh > "/home/retkowski/Crawler/crawlerLog.txt" 2>&1
        #self._writeIntoDatabase()
        
    def _convertIntoTupleList(self, json):
        self._NewsCollection = []
        for line in json:
            self._NewsCollection.append(self._News(**line))
    
    def _getDataMatrix(self):
        # Dynamically create Matrix
        siteList = list(set([news.site for news in self._NewsCollection]))
        tagList = list(set([tag for news in self._NewsCollection for tag in news.tags]))
        data_matrix = [[0 for x in range(len(tagList)+2)] for y in range(len(siteList)+2)] 

        # Set Tag Label
        data_matrix[0] = ['']+tagList+['Σ']

        # Set Site Label
        for siteCount, site in enumerate(siteList):
            data_matrix[siteCount+1][0] = site

        # Count Elements
        for element in self._NewsCollection:
            for tag in element.tags:
                data_matrix[siteList.index(element.site)+1][tagList.index(tag)+1] += 1

        # Count Sites
        for x in range(1,len(siteList)+1):
             data_matrix[x][len(tagList)+1] = len([news for news in self._NewsCollection if news.site == siteList[x-1]])

        # Count Tags
        sum = 0
        for y in range(1, len(tagList)+1):
            tempsum = len([news for news in self._NewsCollection if tagList[y-1] in news.tags])
            data_matrix[len(siteList)+1][y] = tempsum
            sum += tempsum

        # Finishing Labeling
        data_matrix[len(siteList)+1][0] = "Σ"
        data_matrix[len(siteList)+1][len(tagList)+1] = sum

        return data_matrix
    
    def _printMatrix(self):
        offline.init_notebook_mode()
        colorscale = [[0, '#bbb5b5'],[.5, '#fafafa'],[1, '#fefefe']]
        table = FF.create_table(self._getDataMatrix(), index=True, colorscale=colorscale)
        offline.iplot(table)
    
    def _importAll(self):
        self._convertIntoTupleList(self._db.all())
        
    def _removeAll(self):
        self._db.purge()
    
    def _crawlNYTintoDB(self, no_text = None):
        tag_re = re.compile(r'(<!--.*?-->|<[^>]*>)')
        nyt_path = "/home/retkowski/nltk_data/nyt/data/2007"
        counter = 0
        max_documents = 10000000
        archives = [archive for archive in os.listdir(nyt_path) if os.path.isfile(os.path.join(nyt_path, archive)) and not archive[0]== "."]
        print("Reading archives:", archives)
        for archive in archives:
            with tarfile.open(nyt_path+"/"+archive) as afile:
                for member in afile:
                    if member.isreg() and member.name.endswith('.xml'): # regular xml file
                        with closing(afile.extractfile(member)) as xmlfile:
                            
                            counter += 1
                            #print(member)
                            if counter is max_documents :
                                return
                            
                            root = etree.parse(xmlfile).getroot()
                            
                            document = namedtuple("News",["title","site","abstract","tags","text","url"])
                            
                            findHelper = root.find(".//*[@name='online_sections']")
                            if findHelper is None:
                                continue
                            document.tags = findHelper.get("content").replace(" ","").split(";")
                            
                            # Labels [9]: Arts, Business, Science, Sports, Technology, Health, Opinion, Washington (-> Politics), Style
# Label entfernen [11]: Magazine, Travel, HomeandGarden, Obituaries, World, DiningandWine, NewYorkandRegion, US, Education, FrontPage, RealEstate
# Datensatz entfernen [4]: PaidDeathNotices, WeekinReview, Corrections, JobMarket
# Mappen [6]: (Books, Theater, Movies -> Arts), (Automobile -> Business), (Editors' Notes, The Public Editor -> Opinion)
                            
                            tags = ["Arts", "Business", "Science", "Sports", "Technology", "Health", "Opinion", "Style"]
                            for tag in document.tags:
                                # Whitelist
                                if tag in []
                                # Map tags
                                if tag in ["Books","Theater","Movies"]:
                                    tags.append("Arts")
                                    continue
                                if tag in ["Automobile"]:
                                    tags.append("Business")
                                    continue
                                if tag in ["Washington"]:
                                    tags.append("Politics")
                                    continue
                                if tag in ["ThePublicEditor","Editors'Notes"]:
                                    tags.append("Opinion")
                                    continue
                            
                            
                            # Remove Double Tags
                            tags = list(set(tags))
                
                            # Remove Empty Tags
                            if len(tags) == 0 :
                                continue
                            
                            #set(array1) & set(array2)
                            findHelper = root.find(".//*[@class='full_text']")
                            if findHelper is None:
                                continue
                            document.text = ""
                            if no_text is None:
                                for p in findHelper.findall("p"):                               
                                    # Remove well-formed tags, fixing mistakes by legitimate users
                                    no_tags = tag_re.sub('', p.text)
                                    document.text = str(document.text) + no_tags.replace("''",'"') +"\n"
                            # add Absatz?
                            
                            findHelper.find(".//hedline")
                            if findHelper is None:
                                continue
                            document.title = findHelper[0].text
                            document.title = str(document.title).replace("''",'"')
                            
                            # Check if supported tags and if nothing is empty
                            
                            #print(document.title)
                            #print(document.abstract)
                            #print(document.tags)
                            #print(document.text)
                            
                            #print( document.tags )   
                            self._db.insert({'title'    : document.title,
                                             'url'      : "",
                                             'site'     : "",
                                             'tags'     : document.tags,
                                             'text'     : document.text,
                                             'abstract' : ""})
                                

    