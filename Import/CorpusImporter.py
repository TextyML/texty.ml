import tarfile,os
import sys
import gzip
from contextlib import closing
from xml.etree import ElementTree as etree
import re
from collections import Counter
from BaseImporter import *


class CorpusImporter(BaseImporter):
    def __init__(self, path = '/home/retkowski/Private/newsDB.json'):
        self._db = TinyDB(path)
        self._Collection = []
        self._Item = namedtuple("Item", ["title", "tags", "text"])   
        
    def _getTags(self, tags_input, is_multilabel, mapping, whitelist, blacklist ):
        tags = []
        
        for tag in tags_input:
            
            if tag in blacklist:
                return []
            
            if tag in whitelist:
                tags.append(tag)
                continue
            
            for maps in mapping:
                if tag in maps[1]:
                    if(maps[0] in whitelist):
                        tags.append(maps[0])
                        continue
                    
         
        
        
        # Remove Double Tags
        tags = list(set(tags))
        
        if not is_multilabel :
            if len(tags) > 1 :
                tags = []
              
        
        return tags
    
    def _extractElements(self, xmlfile, no_text, is_multilabel, mapping, whitelist, blacklist):
        tag_re = re.compile(r'(<!--.*?-->|<[^>]*>)')
        root = etree.parse(xmlfile).getroot()
                            
        findHelper = root.find(".//*[@name='online_sections']")
        if findHelper is None:
            return
        tags = self._getTags(findHelper.get("content").replace(" ","").split(";"),
                             is_multilabel, mapping, whitelist, blacklist)                           
                
        # Remove Empty Tags
        if len(tags) == 0 :
            return
               
        findHelper = root.find(".//*[@class='full_text']")
        if findHelper is None:
            return
        text = ""
        if no_text is False:
            for p in findHelper.findall("p"):                               
                # Remove well-formed tags, fixing mistakes by legitimate users
                no_tags = tag_re.sub('', p.text)
                text = text + no_tags.replace("''",'"') +"\n"
                # add Absatz?
                            
        findHelper.find(".//hedline")
        if findHelper is None:
            return
        title = findHelper[0].text
        title = title.replace("''",'"')
        
        return self._Item( title = title,
                           tags = tags,
                           text = text)
    
    def crawlNYT(self, per_tag = 10, max_count = 1000, no_text = False, to_database = False, is_multilabel = True, 
                 mapping = [["Arts",["Books","Theater","Movies"]],
                            ["Business",["Automobile"]],
                            ["Politics",["Washington"]], 
                            ["Opinion",["ThePublicEditor","Editors'Notes"]]], 
                 whitelist = ["Arts", "Business", "Science", "Sports", "Technology", "Health", "Opinion", "Style", "Politics"],
                 blacklist = ["PaidDeathNotices", "WeekInReview", "Corrections", "JobMarket"],
                 corpusPath = "/home/retkowski/nltk_data/nyt/",
                 nytPaths = ["2007","2006","2005"]):   
        """Read the NYT Corpus and write it to DB or Memory

        Keyword arguments:
        max_count -- How many documents should be read in one crawl execution (default 1000)
        no_text -- Should the full text be saved (default True)
        to_database -- Should the dataset be saved to a physical dataset (default False) 
        is_multilabel -- Should multilabel be used? (default True)
        mapping -- Merge similiar labels in one label e.g. Books, Theater, Movies => Arts
        whitelist -- Labels that should be accepted
        blacklist -- Labels that will be ignored
        nytPath -- filepath of the archive to crawl in
        """
        count_tags = {}
        counter = 0
        for nytPath in nytPaths:
            nytPath = corpusPath + nytPath
            archives = [archive for archive in os.listdir(nytPath) if os.path.isfile(os.path.join(nytPath, archive)) and not archive[0]== "."]
            print("Reading archives:", archives)
            for archive in archives:
                with tarfile.open(nytPath+"/"+archive) as afile:
                    for member in afile:
                        if member.isreg() and member.name.endswith('.xml'): # regular xml file
                            with closing(afile.extractfile(member)) as xmlfile:

                                if counter is max_count :
                                    return

                                corpus_item = self._extractElements(xmlfile, no_text, is_multilabel, mapping, whitelist, blacklist)

                                if corpus_item is None:
                                    continue
                                
                                joined_tags = ''.join(corpus_item.tags)
                                
                                
                                count_tags[joined_tags] = count_tags.get(joined_tags, 0)
                                
                                if count_tags[joined_tags] is per_tag:
                                    continue                                
                                
                                count_tags[joined_tags] += 1
                                
                                if to_database is True:  
                                    self._db.insert(corpus_item._asdict())
                                else : 
                                    self._Collection.append(corpus_item)

                                counter += 1