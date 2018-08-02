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
from app import app
import cross_JSE

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# general
ap.add_argument("-i", "--input", default="./Output",
                help="path to the input data file. Default = './Output'")
#ap.add_argument("-p", "--port", default=8050, type=int,
#                help="dash app port. Default = 8050")
args = vars(ap.parse_args())

docs_dist= pd.read_csv(args['input']+"/comparison_results_JSE_RWE/cross_model/JSE/document_distribution.csv")



app.scripts.config.serve_locally = True

app.layout = html.Div( children=[
    html.H1('JSE & RWE Comparison Results', style={'textAlign': 'center', 'background-color': '#7FDBFF'}),
    html.Div([
        html.Div([
            dcc.Tabs(
                tabs=[
                    {'label':'Cross Model (JSE)', 'value':1},
                    {'label':'Cross Model (RWE)', 'value':2},
                    {'label':'Combined Model', 'value':3},
                    {'label':'Matching Model', 'value':4}
                ],
                value=1,
                id='tabs'),
            html.Div(id='tab-output')
            ])
    ]),
])


@app.callback(Output('tab-output', 'children'), [Input('tabs', 'value')])
def tabs_output(value):
    if value == 1:
        return cross_JSE.layout
    else:
        return html.Div([
        html.H1('results coming soon!')
    ])



app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

if __name__ == '__main__':
    app.run_server(debug=True,port=8050, host='0.0.0.0')