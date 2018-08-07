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

docs_dist= pd.read_csv(args['input']+"/comparison_results_JSE_RWE/cross_model/RWE/document_distribution.csv")
js_out =pd.read_csv(args['input']+"/JSE/documents.csv") # jse data frame
ww_out =pd.read_csv(args['input']+"/RWE/documents.csv") # rwe data frame
timeline_jse=pd.read_csv(args['input']+"/comparison_results_JSE_RWE/cross_model/RWE/timeline_overview_1.csv",index_col='year')
timeline_rwe=pd.read_csv(args['input']+"/comparison_results_JSE_RWE/cross_model/RWE/timeline_overview_2.csv",index_col='year')

# load data for plots
bhtsne_3d = pd.read_csv(args['input'] + '/RWE/bhtsne_3d.csv', names=['x', 'y', 'z'])
bhtsne_2d = pd.read_csv(args['input'] + '/RWE/bhtsne_2d.csv', names=['x', 'y'])
joined_bhtsne_3d = ww_out.join(bhtsne_3d)
joined_bhtsne_2d = ww_out.join(bhtsne_2d)
traces_2d=[]
traces_3d = []
unique_topics = ww_out.gmm_top_topic.unique()

# 2D Plot
for topic in range(len(unique_topics)):
    r = lambda: random.randint(0, 255)
    colorhex = '#%02X%02X%02X' % (r(), r(), r())
    tmp = joined_bhtsne_2d[joined_bhtsne_2d.gmm_top_topic == topic]
    x = tmp['x'].values
    y = tmp['y'].values
    titles = tmp['title'].values
    texts = tmp['text'].values
    texts = ['{} ...'.format(text[:100]) for text in texts]
    # sources = tmp['source'].values
    labels = ["""
            Topic: {} <br>
            Title: {} <br>

            Text sample: {} <br>
            """.format(topic,
                       titles[i],
                       # sources[i],
                       texts[i])
              for i in range(len(titles))]
    trace = go.Scattergl(
        x=x,
        y=y,
        customdata=[topic],
        text=labels,
        name='Topic {}'.format(topic),
        mode='markers',
        hoverinfo='text',
        marker=dict(
            color=colorhex,
            line=dict(width=1)
        )
    )
    traces_2d.append(trace)
scatter_2d = {
    'data': traces_2d,
    'layout': {'height': 800,
               'width': 800,
               'hovermode': 'closest'},
}

# 3D Plot
print('prepare 3d tsne scatter..')

for topic in range(len(unique_topics)):
    r = lambda: random.randint(0, 255)
    colorhex = '#%02X%02X%02X' % (r(), r(), r())
    tmp = joined_bhtsne_3d[joined_bhtsne_3d.gmm_top_topic == topic]
    x = tmp['x'].values
    y = tmp['y'].values
    z = tmp['z'].values
    titles = tmp['title'].values
    texts = tmp['text'].values
    texts = ['{} ...'.format(text[:100]) for text in texts]
    # sources = tmp['source'].values
    labels = ["""
            Topic: {} <br>
            Title: {} <br>

            Text sample: {} <br>
            """.format(topic,
                       titles[i],
                       # sources[i],
                       texts[i])
              for i in range(len(titles))]
    trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        customdata=[topic],
        text=labels,
        name='Topic {}'.format(topic),
        mode='markers',
        hoverinfo='text',
        marker=dict(
            color=colorhex,
            line=dict(width=1)
        )
    )
    traces_3d.append(trace)
scatter_3d = {
    'data': traces_3d,
    'layout': {'height': 800,
               'width': 800,
               'hovermode': 'closest'},
}

layout = html.Div(children=[
    html.H2('Model 2 (JSE in RWE)', style={'textAlign': 'center', 'color': '#7FDBFF'}),
    dcc.Graph(
        id='topic-importance_2',
        figure={
            'data': [
                {'x': docs_dist.index, 'y': docs_dist['JSE'], 'type': 'bar', 'name': 'JSE'},
                {'x': docs_dist.index, 'y': docs_dist['RWE'], 'type': 'bar', 'name': 'RWE'},
            ],
            'layout': {
                'title': 'Document distribution',
                'xaxis': {'title': 'Topics'}
                }
        }),
    dcc.Input(
        id='input-value_2',
        placeholder='Topic...',
        type='number',
        value=0,
        min=0,
        max=max(docs_dist.index)
    ),
    html.Div([
        html.Div([
            html.H2('Word Cloud', style={'textAlign': 'center', 'color': '#1C4E80'}),
            html.Img(id='img_2', style={'width': '500px'})
        ], className="six columns"),
        html.Div([
            html.H2('Timeline', style={'textAlign': 'center', 'color': '#1C4E80'}),
            dcc.Graph(id='timeline_2', animate=False)
        ], className="six columns"),
    ], className="row"),
    html.Div([
        html.H2('Topic Explorer', style={'textAlign': 'center', 'color': '#1C4E80'}),
        dcc.Tabs(
            tabs=[
                {'label': '2D Plot', 'value': 2},
                {'label': '3D Plot', 'value': 3}
            ],
            value=2,
            id='tabs-scatter_2'),
        html.Div(id='scatter-output_2')
    ]),
    html.Div(id='cross_rwe-content')
])

@app.callback(Output(component_id='img_2', component_property='src'),
              [Input(component_id='input-value_2', component_property='value')]
              )
def update_img(value):
    try:
        image_filename = args['input'] + '/RWE/wordclouds/topic_{}.pdf.png'.format(
            value)
        encoded_image = base64.b64encode(open(image_filename, 'rb').read())
        return 'data:image/png;base64,{}'.format(encoded_image.decode())
    except Exception as e:
        with open(args['input'] + '/errors.txt', 'a') as f:
            f.write(str(e))
            f.write('\n')

@app.callback(Output(component_id='timeline_2', component_property='figure'),
              [Input(component_id='input-value_2', component_property='value')]
              )
def update_timeline(topic):
    try:
        topic = str(topic)
        X = timeline_rwe.index  # timeline jse
        Y = timeline_rwe[topic].values
        X1 = timeline_jse.index # timeline rwe
        Y1 = timeline_jse[topic].values
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

@app.callback(Output('scatter-output_2', 'children'),
              [Input('tabs-scatter_2', 'value')])
def tabs_scatter(value):
    if value == 2:
        return html.Div([
        dcc.Graph(id='scatter_2d', figure = scatter_2d )
    ])
    else:
        return html.Div([
        dcc.Graph(id='scatter_3d', figure = scatter_3d)
    ])
