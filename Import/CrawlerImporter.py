from .BaseImporter import *
class CrawlerImporter(BaseImporter):
    def __init__(self, path = '/home/retkowski/Data/newsDB.json', crawlerPath = '/home/retkowski/Crawler/' ):
        self._db = TinyDB(path)
        self._crawlerPath = crawlerPath
        self._Collection = []
        self._Item = namedtuple("Item", ["title", "site", "tags", "text", "abstract", "url"])
    
    def _updateCrawler(self):
        os.system("cd "+ self._crawlerPath+ " && git stash && git pull")
    
    
    def _writeIntoDatabase(self):
        path = self._crawlerPath+"data"
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
        with open(self._crawlerPath + "download-all.txt") as dl:
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
            
    def runCrawler(self):
        self._buildDownloadList()
        self._updateCrawler()
        os.system("cd "+ self._crawlerPath+ " && ./run-crawlers.sh > \"/home/retkowski/Crawler/crawlerLog.txt\" 2>&1")
        self._writeIntoDatabase()
        
        
    def _getDataMatrix(self):
        # Dynamically create Matrix
        siteList = list(set([news.site for news in self._Collection]))
        tagList = list(set([tag for news in self._Collection for tag in news.tags]))
        data_matrix = [[0 for x in range(len(tagList)+2)] for y in range(len(siteList)+2)] 

        # Set Tag Label
        data_matrix[0] = ['']+tagList+['Σ']

        # Set Site Label
        for siteCount, site in enumerate(siteList):
            data_matrix[siteCount+1][0] = site

        # Count Elements
        for element in self._Collection:
            for tag in element.tags:
                data_matrix[siteList.index(element.site)+1][tagList.index(tag)+1] += 1

        # Count Sites
        for x in range(1,len(siteList)+1):
             data_matrix[x][len(tagList)+1] = len([news for news in self._Collection if news.site == siteList[x-1]])

        # Count Tags
        sum = 0
        for y in range(1, len(tagList)+1):
            tempsum = len([news for news in self._Collection if tagList[y-1] in news.tags])
            data_matrix[len(siteList)+1][y] = tempsum
            sum += tempsum

        # Finishing Labeling
        data_matrix[len(siteList)+1][0] = "Σ"
        data_matrix[len(siteList)+1][len(tagList)+1] = sum

        return data_matrix
    
    def printMatrix(self):
        offline.init_notebook_mode()
        colorscale = [[0, '#bbb5b5'],[.5, '#fafafa'],[1, '#fefefe']]
        table = FF.create_table(self._getDataMatrix(), index=True, colorscale=colorscale)
        offline.iplot(table)