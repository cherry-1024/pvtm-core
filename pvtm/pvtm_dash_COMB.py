# import utils_func
# import stopwords_func
# import Doc2Vec_func
# import settings
import argparse
import base64
import random

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# general
ap.add_argument("-i", "--input", default="./Output", required=True,
                help="path to the input data file")
args = vars(ap.parse_args())

documents = pd.read_csv(args['input'] + '/documents.csv')
topics = list(range(0, max(documents['gmm_top_topic']) + 1))
timelines_df = pd.read_csv(args['input'] + '/timelines_df.csv', index_col='Unnamed: 0')

# Scatter Plot
bhtsne = pd.read_csv(args['input'] + '/bhtsne.csv', names=['x', 'y'])
bhtsne_3d = pd.read_csv(args['input'] + '/bhtsne_3d.csv', names=['x', 'y', 'z'])
out = documents.copy()
out = out.join(bhtsne_3d)
traces = []
unique_topics = documents.gmm_top_topic.unique()
for topic in range(len(unique_topics)):
    r = lambda: random.randint(0, 255)
    colorhex = '#%02X%02X%02X' % (r(), r(), r())
    tmp = out[out.gmm_top_topic == topic]
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
        customdata=topic,
        text=labels,
        name='Topic {}'.format(topic),
        mode='markers',
        hoverinfo='text',
        marker=dict(
            color=colorhex,
            line=dict(width=1)
        )
    )
    traces.append(trace)
scatter = {
    'data': traces,
    'layout': {'height': 800,
               'width': 800,
               'hovermode': 'closest'},
}

app = dash.Dash()

app.layout = html.Div(children=[
    html.H1('PVTM Results', style={'textAlign': 'center', 'background-color': '#7FDBFF'}),
    html.Div([
        html.Div([
            html.H2('Topic Explorer', style={'textAlign': 'center', 'color': '#1C4E80'}),
            dcc.Graph(id='scatter', figure=scatter)
            ], className="six columns"),
        html.Div([
            html.H2('Mean Topic Probability', style={'textAlign': 'center', 'color': '#1C4E80'}),
            dcc.Graph(id='mean_probs',
                      figure={
                          'data': [
                              {'x': timelines_df.mean().values, 'y':topics, 'type': 'bar', 'orientation':'h'}
                          ]
                      }
                      )
            ], className="six columns"),
    ], className="row"),
    dcc.Slider(
        id='topic-slider',
        min=0,
        max=max(documents['gmm_top_topic']),
        step=None,
        marks={str(topic): str(topic) for topic in topics}
    ),
    html.Div([
        html.Div([
            html.H2('Word Cloud', style={'textAlign': 'center', 'color': '#1C4E80'}),
            html.Img(id='img', style={'width': '500px'})
        ], className="six columns"),
        html.Div([
            html.H2('Timeline', style={'textAlign': 'center', 'color': '#1C4E80'}),
            dcc.Graph(id='timeline', animate=False)
        ], className="six columns"),
    ], className="row")
])


@app.callback(Output(component_id='img', component_property='src'),
              [Input(component_id='topic-slider', component_property='value')]
              )
def update_img(topic):
    try:
        image_filename = args['input'] + '/wordclouds/topic_{}.pdf.png'.format(
            topic)
        encoded_image = base64.b64encode(open(image_filename, 'rb').read())
        return 'data:image/png;base64,{}'.format(encoded_image.decode())
    except Exception as e:
        with open(args['input']+'errors.txt', 'a') as f:
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
        Y = timelines_df[topic].ewm(span=5).mean().values
        X1 = timelines_df.index
        Y1 = np.repeat(1 / max(topics), len(X1))
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
        with open(args['input']+'errors.txt', 'a') as f:
            f.write(str(e))
            f.write((str(topic)))
            f.write('\n')


app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})
  
if __name__ == '__main__':
    app.run_server(debug=True,port=8050, host='0.0.0.0')
