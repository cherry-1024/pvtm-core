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

docs_dist= pd.read_csv(args['input']+"/comparison_results_JSE_RWE/cross_model/JSE/document_distribution.csv")
js_out =pd.read_csv(args['input']+"/JSE/documents.csv") # jse data frame
ww_out =pd.read_csv(args['input']+"/RWE/documents.csv") # rwe data frame
timeline_jse=pd.read_csv(args['input']+"/comparison_results_JSE_RWE/cross_model/JSE/timeline_overview_1.csv",index_col='year')
timeline_rwe=pd.read_csv(args['input']+"/comparison_results_JSE_RWE/cross_model/JSE/timeline_overview_2.csv",index_col='year')

# load data for plots
bhtsne_3d = pd.read_csv(args['input'] + '/JSE/bhtsne_3d.csv', names=['x', 'y', 'z'])
bhtsne_2d = pd.read_csv(args['input'] + '/JSE/bhtsne_2d.csv', names=['x', 'y'])
joined_bhtsne_3d = js_out.join(bhtsne_3d)
joined_bhtsne_2d = js_out.join(bhtsne_2d)
traces_2d=[]
traces_3d = []
unique_topics = js_out.gmm_top_topic.unique()

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

# the model app layout contains documents distribution bar chart, wordcloud, timelines and scatter plots

layout= html.Div(children=[
    html.H2('Model 1 (RWE in JSE)', style={'textAlign': 'center', 'color': '#7FDBFF'}),
    dcc.Graph(
        id='topic-importance',
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
        id='input-value',
        placeholder='Topic...',
        type='number',
        value=0,
        min=0,
        max=max(docs_dist.index) # number of topics
    ),
    html.Div([
        html.Div([
            html.H2('Word Cloud', style={'textAlign': 'center', 'color': '#1C4E80'}),
            html.Img(id='img', style={'width': '500px'})
        ], className="six columns"), # about 50 % of the layout
        html.Div([
            html.H2('Timeline', style={'textAlign': 'center', 'color': '#1C4E80'}),
            dcc.Graph(id='timeline', animate=False)
        ], className="six columns"),
    ], className="row"),
    html.Div([
        html.H2('Topic Explorer', style={'textAlign': 'center', 'color': '#1C4E80'}),
        dcc.Tabs(
            tabs=[
                {'label': '2D Plot', 'value': 2},
                {'label': '3D Plot', 'value': 3}
            ],
            value=2, # default: 2D plot, depending on corpus size 3d plot could slow down the app
            id='tabs-scatter'),
        html.Div(id='scatter-output')
    ]),
    html.Div(id='cross_jse-content')
])

@app.callback(Output(component_id='img', component_property='src'),
              [Input(component_id='input-value', component_property='value')]
              )
def update_img(value):
    try:
        image_filename = args['input'] + '/JSE/wordclouds/topic_{}.pdf.png'.format(
            value)
        encoded_image = base64.b64encode(open(image_filename, 'rb').read())
        return 'data:image/png;base64,{}'.format(encoded_image.decode())
    except Exception as e:
        with open(args['input'] + '/errors.txt', 'a') as f:
            f.write('Topic {}'.format(value))
            f.write('\n')
            f.write(str(e))
            f.write('\n')

@app.callback(Output(component_id='timeline', component_property='figure'),
              [Input(component_id='input-value', component_property='value')]
              )
def update_timeline(topic):
    try:
        topic = str(topic)
        X = timeline_jse.index  # timeline jse
        Y = timeline_jse[topic].ewm(span=5).mean().values
        X1 = timeline_rwe.index # timeline rwe
        Y1 = timeline_rwe[topic].ewm(span=5).mean().values
        figure = {
            'data': [
                {'x': X, 'y': Y, 'type': 'line', 'name': 'JSE'},
                {'x': X1, 'y': Y1, 'type': 'line', 'name': 'RWE'},
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
            f.write('Topic {}'.format(topic))
            f.write('\n')
            f.write(str(e))
            f.write('\n')

@app.callback(Output('scatter-output', 'children'),
              [Input('tabs-scatter', 'value')])
def tabs_scatter(value):
    if value == 2:
        return html.Div([
        dcc.Graph(id='scatter_2d', figure = scatter_2d )
    ])
    else:
        return html.Div([
        dcc.Graph(id='scatter_3d', figure = scatter_3d)
    ])

