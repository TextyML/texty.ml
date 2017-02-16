import os
from collections import namedtuple
import json
import plotly.offline as offline
import plotly.plotly as py
import plotly.tools as pt
from plotly.tools import FigureFactory as FF
from tinydb import TinyDB, Query

class BaseImporter():
    def __init__(self):
        pass
    
    def _convertIntoTupleList(self, json):
        self._Collection = []
        for line in json:
            self._Collection.append(self._Item(**line))
    
    def clearMemory(self):
        self._Collection = []
    
    def importAllFromDB(self):
        self._convertIntoTupleList(self._db.all())
        
    def removeAllFromDB(self):
        self._db.purge()
    
    def saveIntoDB(self):
        for item in self.Collection :
            self._db.insert(item._asdict())
    
        
    