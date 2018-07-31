import argparse
import pvtm_utils
import pandas as pd
import random
import plotly.graph_objs as go
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# general
ap.add_argument("-i", "--input", default="./Output", required=True,
                help="path to the input data file")
ap.add_argument("-p", "--port", default=8050, required=True,type=int,
                help="dash app port")
args = vars(ap.parse_args())



model, gmm, documents, topics_df = pvtm_utils.load_pvtm_outputs(args['input'])
documents = pvtm_utils.extract_time_info(documents, 'date')
# load data for plots
bhtsne_3d = pd.read_csv(args['input'] + '/bhtsne_3d.csv', names=['x', 'y', 'z'])
bhtsne_2d = pd.read_csv(args['input'] + '/bhtsne_2d.csv', names=['x', 'y'])
joined_bhtsne_3d = documents.join(bhtsne_3d)
joined_bhtsne_2d = documents.join(bhtsne_2d)
traces_2d=[]
traces_3d = []
unique_topics = documents.gmm_top_topic.unique()

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
    traces_3d.append(trace)
scatter_3d = {
    'data': traces_3d,
    'layout': {'height': 800,
               'width': 800,
               'hovermode': 'closest'},
}
