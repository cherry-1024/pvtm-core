import argparse
import base64
import random

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.graph_objs as go

from dash.dependencies import Input, Output, Event
import pvtm_utils



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# general
ap.add_argument("-i", "--input", default="./Output", required=True,
                help="path to the input data file")
ap.add_argument("-p", "--port", default=8050, required=True,type=int,
                help="path to the input data file")
args = vars(ap.parse_args())

model, gmm, documents, topics_df = pvtm_utils.load_pvtm_outputs(args['input'])

print('extract date info..')
# documents['date'] = documents['date'].apply(pd.to_datetime, errors='coerce')
documents = pvtm_utils.extract_time_info(documents, 'date')
topics = list(range(0, max(documents['gmm_top_topic']) + 1))


print('generate topic timelines..')
timelines_year = pvtm_utils.get_topic_importance_df('year', documents)
timelines_month = pvtm_utils.get_topic_importance_df('month', documents)
timelines_week = pvtm_utils.get_topic_importance_df('week', documents)
timelines_day = pvtm_utils.get_topic_importance_df('day', documents)
timelines_hour = pvtm_utils.get_topic_importance_df('hour', documents)

print('prepare 3d tsne scatter..')

# 2d & 3d Scatter Plot
bhtsne_2d = pd.read_csv(args['input'] + '/bhtsne_2d.csv', names=['x', 'y'])

# 3d Scatter Plot

bhtsne_3d = pd.read_csv(args['input'] + '/bhtsne_3d.csv', names=['x', 'y', 'z'])
joined_bhtsne = documents.join(bhtsne_3d)

traces = []
traces_2d=[]
unique_topics = documents.gmm_top_topic.unique()
for topic in range(len(unique_topics)):
    r = lambda: random.randint(0, 255)
    colorhex = '#%02X%02X%02X' % (r(), r(), r())
    tmp = documents.join(bhtsne_2d)[documents.join(bhtsne_2d).gmm_top_topic == topic]
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
    traces_2d.append(trace)
scatter_2d = {
    'data': traces_2d,
    'layout': {'height': 800,
               'width': 800,
               'hovermode': 'closest'},
}


for topic in range(len(unique_topics)):
    r = lambda: random.randint(0, 255)
    colorhex = '#%02X%02X%02X' % (r(), r(), r())
    tmp = joined_bhtsne[joined_bhtsne.gmm_top_topic == topic]
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
scatter_3d = {
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

#            html.Button('3D Plot', id='show 3d plot', n_clicks=0),
#            html.Div(id='p_section',style={'display': 'none'}),
            dcc.Graph(id='scatter_2d', figure=scatter_2d)
            ], className="six columns"),

        html.Div([
            html.H2('Mean Topic Probability', style={'textAlign': 'center', 'color': '#1C4E80'}),
            dcc.Graph(id='mean_probs',
                      figure={
                          'data': [
                              {'x': timelines_year.mean().values, 'y': topics, 'type': 'bar', 'orientation': 'h'}
                          ]
                      }
                      )
        ], className="six columns"),
    ], className="row"),
    dcc.Input(
        id='input-value',
        placeholder='Topic...',
        type='number',
        value='',
        min=0,

        max=max(documents['gmm_top_topic'])

    ),
#    dcc.Slider(
#        id='topic-slider',
#        min=0,
#        max=max(documents['gmm_top_topic']),
#        step=None,
#        marks={str(topic): str(topic) for topic in topics}
#    ),
    html.Div([
        html.Div([
            html.H2('Word Cloud', style={'textAlign': 'center', 'color': '#1C4E80'}),
            html.Img(id='img', style={'width': '500px'})
        ], className="six columns"),
        html.Div([
            html.H2('Timeline', style={'textAlign': 'center', 'color': '#1C4E80'}),
            dcc.Dropdown(
                id='dropdown',
                options=[
                    {'label': 'year', 'value': 'year'},
                    {'label': 'month', 'value': 'month'},
                    {'label': 'day', 'value': 'day'},
                    {'label': 'week', 'value': 'week'}
                ]
            ),
            dcc.Dropdown(
                id='dropdown_smoothing',
                options=[
                    {'label': 5, 'value': 5},
                    {'label': 4, 'value': 4},
                    {'label': 3, 'value': 3},
                    {'label': 2, 'value': 2},
                    {'label': 1, 'value': 1},
                ]
            ),
            dcc.Graph(id='timeline', animate=False)
        ], className="six columns"),
    ], className="row")
])

#@app.callback(
#    dash.dependencies.Output('p_section', 'children'),
#    events=[Event('show 3d plot', 'n_clicks')])

#def update_output_graph(n_clicks):
#    if nclick > 0 :
#        figure_3d = dcc.Graph(
#        id='scatter_3d',
#        figure=scatter_3d
#    )
#    return figure_3d

@app.callback(Output(component_id='img', component_property='src'),
              [Input(component_id='input-value', component_property='value')]
              )
def update_img(value):
    try:
        image_filename = args['input'] + '/wordclouds/topic_{}.pdf.png'.format(
            value)
        encoded_image = base64.b64encode(open(image_filename, 'rb').read())
        return 'data:image/png;base64,{}'.format(encoded_image.decode())
    except Exception as e:
        with open(args['input'] + '/errors.txt', 'a') as f:
            f.write(str(e))
            f.write('\n')


@app.callback(dash.dependencies.Output('timeline', 'figure'),

              [dash.dependencies.Input('input-value', 'value'),
               dash.dependencies.Input('dropdown', 'value'),
               dash.dependencies.Input('dropdown_smoothing', 'value')]
              )
def update_timeline(topic, dropvalue, smoothing):

    try:
        if dropvalue == 'year':
            timelines_df = timelines_year.copy()
        elif dropvalue == 'month':
            timelines_df = timelines_month.copy()
        elif dropvalue == 'week':
            timelines_df = timelines_week.copy()
        elif dropvalue == 'day':
            timelines_df = timelines_day.copy()
        elif dropvalue == 'hour':
            timelines_df = timelines_hour.copy()
        else:
            timelines_df = pvtm_utils.get_topic_importance_df(dropvalue, documents)


        X = timelines_df.index

        Y = timelines_df[topic].ewm(span=smoothing).mean().values

        X1 = timelines_df.index
        Y1 = np.repeat(1 / (max(topics) + 1), len(X1))
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
        with open(args['input'] + '/errors.txt', 'a') as f:
            f.write(str(e))
            f.write('\n')
            f.write('Topic:'.format(str(topic)))
            f.write('\n')


app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

if __name__ == '__main__':

    app.run_server(debug=True,port=args['port'], host='0.0.0.0')

