import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import dash_renderer
import time
from collections import deque
import sqlite3
from scipy.stats.stats import pearsonr
import plotly
import plotly.graph_objs as go
import random
import matplotlib
import pandas as pd
import base64
import numpy as np
import plotly.graph_objs as go
import os

# import utils_func
# import stopwords_func
# import Doc2Vec_func
# import settings
import subprocess
import argparse
import json
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# general
ap.add_argument("-i", "--input", default="./Output",required=True,
                help="path to the input data file")
args = vars(ap.parse_args())

documents = pd.read_csv( args['input']+'/documents.csv')
topics= list(range(0,max(documents['gmm_top_topic'])+1))
timelines_df = pd.read_csv( args['input']+'/timelines_df.csv', index_col='Unnamed: 0')
app = dash.Dash()

app = dash.Dash()
app.layout = html.Div(children=[
    html.H1('PVTM Results', style={'textAlign': 'center','background-color': '#7FDBFF'}),
    dcc.Slider(
        id ='topic-slider',
        min = 0,
        max = max(documents['gmm_top_topic']),
        step = None,
        marks ={str(topic): str(topic) for topic in topics}
    ),
    html.Div([
        html.Div([
            html.H2('Word Cloud',style={'textAlign': 'center','color': '#1C4E80'}),
            html.Img(id='img', style={'width': '500px'})
        ], className="six columns"),
        html.Div([
            html.H2('Timeline',style={'textAlign': 'center','color': '#1C4E80'}),
            dcc.Graph(id='timeline', animate=False)
        ], className="six columns"),
    ], className="row")
])


@app.callback(Output(component_id='img', component_property='src'),
              [Input(component_id='topic-slider', component_property='value')]
              )

def update_img(topic):
    try:
        image_filename = args['input']+'/wordclouds/topic_{}.txt.svg.png'.format(
            topic)
        encoded_image = base64.b64encode(open(image_filename, 'rb').read())
        return 'data:image/png;base64,{}'.format(encoded_image.decode())
    except Exception as e:
        with open('errors.txt','a') as f:
            f.write(str(e))
            f.write('\n')


@app.callback(Output(component_id='timeline', component_property='figure'),
              [Input(component_id='topic-slider', component_property='value')]
              )

def update_timeline(topic):
    try:
        topic = str(topic)
        # dates = pd.DatetimeIndex(imp_per_my[topic].ewm(span=3).mean().index.values)
        #  X = dates.year
        X = timelines_df.index
        Y = timelines_df[topic].ewm(span=3).mean().values
        X1= timelines_df.index
        Y1= np.repeat(1/max(topics), len(X1))
        X2 = timelines_df.index
        Y2 = np.repeat(np.mean(Y), len(X2))
        figure = {
            'data': [
                {'x': X, 'y': Y, 'type': 'line', 'name': 'Probability'},
                {'x': X1, 'y': Y1, 'type': 'dotted', 'name': 'Mean Probability all Topics'},
                {'x': X2, 'y': Y2, 'type': 'dashed', 'name': 'Mean Probability current Topic'}
            ],
            'layout': {
                'plot_bgcolor': '#DADADA',
                'paper_bgcolor': '#DADADA'
            }
        }
        return figure
    except Exception as e:
        with open('errors.txt', 'a') as f:
            f.write(str(e))
            f.write((str(topic)))
            f.write((str(ww_in_js_imp_per_my[topic].ewm(span=1).mean().index.values)))
            f.write('\n')

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

if __name__ == '__main__':
    app.run_server(debug=True)