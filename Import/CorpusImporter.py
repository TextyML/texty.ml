import tarfile,os
import sys
import gzip
from contextlib import closing
from lxml import etree
import re
from collections import Counter
from .BaseImporter import *

class CorpusImporter(BaseImporter):
    DATA_FIELDS  =  [                
                ('Text', 'String', 'Single', '/nitf/body/body.content/block[@class="full_text"]'),
                ('Tags', 'String', 'Single', '/nitf/head/meta[@name="online_sections"]/@content'),
                #('OnlineTitles', 'String', 'Multiple', '/nitf/head/docdata/identified-content/object.title[@class="online_producer"]'), 
                #('Titles', 'String', 'Multiple', '/nitf/head/docdata/identified-content/object.title[@class="indexing_service"]')
                ('Titles', 'String', 'Single', '/nitf/body[1]/body.head/hedline/hl1')
                ]
    
    def __init__(self, path = '/home/retkowski/Private/newsDB.json'):
        self._db = TinyDB(path)
        self._Collection = []
        
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
                      
    def _extractElements(self, xmlfile, no_text, is_multilabel, mapping, whitelist, blacklist, extractElements, ignore_tags):
        #tag_re = re.compile(r'(<!--.*?-->|<[^>]*>)')
        tree = etree.parse(xmlfile)
        item = {}
        
        data_fields = [f for f in self.DATA_FIELDS if f[0] in extractElements]
        
        for f in data_fields:
            item_element = f[0].lower()
            item[item_element] = item.get(item_element, None)
            
            #Retrieve from path
            a = tree.xpath(f[3])
            if f[2].lower() == 'single':
                if len(a):
                    if type(a[0]) == etree._Element:
                        s = etree.tostring(a[0], method='text', encoding='unicode')
                    else:
                        s = a[0]
                else:
                    s = None
            else:
                s = []
                for b in a:
                    c = etree.tostring(b, method='text', encoding='unicode').strip()
                    if c != '':
                        s.append(c)
                if s == []:
                    s = None         
            
            if item_element == "tags" and not ignore_tags and s is not None:
                item[item_element] = self._getTags(s.replace(" ","").split(";"),
                             is_multilabel, mapping, whitelist, blacklist)
            else:
                item[item_element] = s
                
                
            if item_element in ["tags"] and not item[item_element]:
                return None

        return namedtuple('Item', item.keys())(**item)
        
    def crawlNYT(self, per_tag = 10, max_count = 1000, no_text = False, to_database = False, is_multilabel = True, 
                 mapping = [["Arts",["Books","Theater","Movies"]],
                            ["Business",["Automobile"]],
                            ["Politics",["Washington"]], 
                            ["Opinion",["ThePublicEditor"]]], 
                 whitelist = ["Arts", "Business", "Science", "Sports", "Technology", "Health", "Opinion", "Style", "Politics"],
                 blacklist = ["PaidDeathNotices", "WeekInReview", "Corrections", "JobMarket", "Editors'Notes"],
                 corpusPath = "/home/retkowski/nltk_data/nyt/",
                 nytPaths = ["2007","2006","2005","2004","2003","2002","2001","2000","1999","1998","1997","1996","1995","1994"],
                 extractElements = ["Text","Tags","Titles"],
                 ignore_tags = False):   
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
        extractElements -- 
        """
        count_tags = {}
        counter = 0
        if not ignore_tags:
            max_count = len(whitelist) * per_tag
            
        print("Maximum Documents: ", str(max_count))
        
        for nytPath in nytPaths:
            nytPath = corpusPath + nytPath
            archives = [archive for archive in os.listdir(nytPath) if os.path.isfile(os.path.join(nytPath, archive)) and not archive[0]== "."]
            print("Reading archives ["+nytPath+"]: ", archives)
            for archive in archives:
                with tarfile.open(nytPath+"/"+archive) as afile:
                    for member in afile:
                        if member.isreg() and member.name.endswith('.xml'): # regular xml file
                            with closing(afile.extractfile(member)) as xmlfile:

                                if counter >= max_count :
                                    return

                                corpus_item = self._extractElements(xmlfile, no_text, is_multilabel, mapping, whitelist, blacklist, extractElements,ignore_tags)

                                if corpus_item is None:
                                    continue
                                
                                if not ignore_tags:
                                    joined_tags = ''.join(corpus_item.tags)


                                    count_tags[joined_tags] = count_tags.get(joined_tags, 0)

                                    if count_tags[joined_tags] >= per_tag:
                                        continue
                                    else:
                                        count_tags[joined_tags] += 1
                                
                                counter = counter + 1
                                if to_database:  
                                    self._db.insert(corpus_item._asdict())
                                else: 
                                    self._Collection.append(corpus_item)
