from ipywidgets import widgets, interact, interactive, fixed
from IPython.display import display, clear_output
import plotly.offline as offline
import plotly.tools as tls
import numpy as np
from random import random
from plotly.graph_objs import *

import plotly.plotly as py
from plotly.tools import FigureFactory as FF
from plotly.graph_objs import graph_objs
import numpy as np
import pandas as pd
from scipy import stats


class FeatureAnalyzer():
    def __init__(self, topics, feature_names):
        offline.init_notebook_mode()
        self.topics = topics
        self.feature_names = np.array(feature_names)
    
    # Set transformed Data    
    def set_data(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
    
    def clear_data(self):
        self.x_data = None
        self.y_data = None
        
    def draw_violins(self):
        interact(self._draw_violins, feature_name=self.feature_names.tolist(), size="1")

    def _draw_violins(self, feature_name):
        index = np.where(self.feature_names == feature_name)[0][0]
        y = []
        gr = []
        for key in self.topics:
            tmp = [x[index] for i, x in enumerate(self.x_data, 0) if key in self.y_data[i]]
            y.extend(tmp)
            gr.extend([key]*len(tmp))

        df = pd.DataFrame(dict(Score = y, Group = gr))

        fig = FF.create_violin(df, data_header='Score', group_header='Group',
                               height=500, width=800, title='Violin Plot of ' + feature_name)
        offline.iplot(fig)

    
    def _draw_histogram(self, feature_name, size):
        size = float(size)
        index = np.where(self.feature_names == feature_name)[0][0]
        traces = []
        legend = [True for x in range(1)]

        for col in range(1):
            for key in self.topics:
                x_data = [x[col+index] for i, x in enumerate(self.x_data, 0) if key in self.y_data[i]]
                traces.append(Histogram(x=x_data, 
                              xbins=dict(start=np.min(x_data)-(size / 2), size= size, end= np.max(x_data)+(size/2)),
                              histnorm='probability',
                              opacity=0.75,
                              xaxis='x%s' %(col+1),
                              name=key,
                              showlegend=legend))

        data = Data(traces)

        layout = Layout(barmode='overlay',
                        yaxis=YAxis(title='count'),
                        title='Feature Extraction: '+self.feature_names[index])

        fig = Figure(data=data, layout=layout)
        offline.iplot(fig)
    
    def draw_histogram(self):
        interact(self._draw_histogram, feature_name=self.feature_names.tolist(), size="1")
    
    def _draw_scatterplot(self, feature1, feature2):
        data = []
        for topic in self.topics:
            data.append(
                Scatter(
                    x = [x[np.where(self.feature_names == feature1)[0][0]] for i, x in enumerate(self.x_data, 0) if topic in self.y_data[i]],
                    y = [x[np.where(self.feature_names == feature2)[0][0]] for i, x in enumerate(self.x_data, 0) if topic in self.y_data[i]],
                    mode ="markers",
                    name = topic))
        layout = Layout(
            showlegend = True,
            title='Scatterplot ' + feature1 + ' versus ' + feature2,
            xaxis = dict(
                title=feature1,
            ),
            yaxis = dict(
                title=feature2,
            ),
            hovermode = 'closest'
        )

        fig = Figure(data=data, layout=layout)


        offline.iplot(fig)
    
    def draw_scatterplot(self):
        interact(self._draw_scatterplot, feature1=self.feature_names.tolist(), feature2=self.feature_names.tolist())
        