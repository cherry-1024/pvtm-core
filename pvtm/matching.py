import argparse
import base64
import random

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import matplotlib
import plotly
import plotly.graph_objs as go

from dash.dependencies import Input, Output
from app import app

ap = argparse.ArgumentParser()
# general
ap.add_argument("-i", "--input", default="./Output",
                help="path to the input data file. Default = './Output'")
# ap.add_argument("-p", "--port", default=8050, type=int,
#                help="dash app port. Default = 8050")
args = vars(ap.parse_args())

assign= pd.read_csv(args['input']+"/comparison_results_JSE_RWE/matching/assign.csv")
timeline_jse = pd.read_csv(args['input']+"/comparison_results_JSE_RWE/matching/timeline_overview_1.csv",index_col='Unnamed: 0')
timeline_rwe = pd.read_csv(args['input']+"/comparison_results_JSE_RWE/matching/timeline_overview_2.csv",index_col='Unnamed: 0')


layout = html.Div(children=[
    html.H1('RWE & JSE matching', style={'textAlign': 'center', 'color': '#7FDBFF'}),
    dcc.Input(
        id='input-value_4',
        placeholder='Topic...',
        type='number',
        value=0,
        min=0,
        max=max(assign.index)
    ),
    html.Div([
        html.H4(id='explanation', style={'textAlign': 'center', 'color': '#1C4E80'})
    ]),
    html.Div([
        html.Div([
            html.H2('Word Cloud RWE',style={'textAlign': 'center','color': '#1C4E80'}),
            html.Img(id='img_RWE4', style={'width': '500px'})
        ], className="six columns"),
        html.Div([
            html.H2('Word Cloud JSE',style={'textAlign': 'center','color': '#1C4E80'}),
            html.Img(id='img_JSE4', style={'width': '500px'})
        ], className="six columns"),
    ], className="row"),
    html.Div([
        html.H2('Timeline',style={'textAlign': 'center','color': '#1C4E80'} ),
        dcc.Graph(id='timeline_4', animate=False,style={'background-color': '#7FDBFF'})
    ]),
    html.Div(id='matching-content')
])
@app.callback(Output(component_id='explanation', component_property='children'),
              [Input(component_id='input-value_4', component_property='value')]
              )
def explanation(value):
    try:
        value= float(value)
        return 'Topic {} from RWE matches topic {} from JSE'.format(round(value),assign['JSE'][value])
    except Exception as e:
        with open(args['input'] + '/errors.txt', 'a') as f:
            f.write('Topic {}'.format(value))
            f.write('\n')
            f.write(str(type(value)))
            f.write('\n')
            f.write(str(e))
            f.write('\n')

@app.callback(Output(component_id='img_RWE4', component_property='src'),
              [Input(component_id='input-value_4', component_property='value')]
              )

def update_img_RWE(value):
    try:
        image_filename = args['input'] + '/RWE/wordclouds/topic_{}.pdf.png'.format(
            value)
        encoded_image = base64.b64encode(open(image_filename, 'rb').read())
        return 'data:image/png;base64,{}'.format(encoded_image.decode())
    except Exception as e:
        with open(args['input'] + '/errors.txt', 'a') as f:
            f.write(str(e))
            f.write('\n')

@app.callback(Output(component_id='img_JSE4', component_property='src'),
              [Input(component_id='input-value_4', component_property='value')]
              )
def update_img_JSE(value):
    try:
        image_filename = args['input'] + '/JSE/wordclouds/topic_{}.pdf.png'.format(assign['JSE'][float(value)])
        encoded_image = base64.b64encode(open(image_filename, 'rb').read())
        return 'data:image/png;base64,{}'.format(encoded_image.decode())
    except Exception as e:
        with open(args['input'] + '/errors.txt', 'a') as f:
            f.write(str(e))
            f.write('\n')

@app.callback(Output(component_id='timeline_4', component_property='figure'),
              [Input(component_id='input-value_4', component_property='value')]
              )
def update_timeline(topic):
    try:
        X = timeline_rwe.index  # timeline jse
        Y = timeline_rwe[str(topic)].values
        X1 = timeline_jse.index # timeline rwe
        Y1 = timeline_jse[str(assign['JSE'][float(topic)])].values
        figure = {
            'data': [
                {'x': X, 'y': Y, 'type': 'line', 'name': 'RWE'},
                {'x': X1, 'y': Y1, 'type': 'line', 'name': 'JSE'},
            ],
            'layout': {
                'plot_bgcolor': '#DADADA',
                'paper_bgcolor': '#DADADA',
                'yaxis': {'title': 'Topic Probability'}
            }
        }
        return figure
    except Exception as e:
        with open(args['input'] + '/errors.txt', 'a') as f:
            f.write(str(e))
            f.write('\n')
