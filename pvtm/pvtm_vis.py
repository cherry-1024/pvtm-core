# import the necessary packages
import argparse
import os
import subprocess

import doc2vec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# custom functions
import pvtm_utils
from bhtsne import tsne
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from reportlab.graphics import renderPDF
from sklearn.externals import joblib
from svglib.svglib import svg2rlg

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True,
                help="path to the output data folder")
ap.add_argument("-tp", "--tsne_perplexity", default=15, required=False,
                help="Perplexity value for tsne")
ap.add_argument("-al", "--agg_lvl", default='year', required=False,
                help="Unit of aggregation for topic importance over time. Can be 'year', 'month','week','day'. Maybe more.")
ap.add_argument("-bhtsne", action='store_true',
                help="If one of -bhtsne, -timelines, -wordclouds is used, then only do the mentioned ones. Otherwise do all visualizations.")
ap.add_argument("-timelines", action='store_true',
                help="If one of -bhtsne, -timelines, -wordclouds is used, then only do the mentioned ones. Otherwise do all visualizations.")
ap.add_argument("-wordclouds", action='store_true',
                help="If one of -bhtsne, -timelines, -wordclouds is used, then only do the mentioned ones. Otherwise do all visualizations.")

parsed_args = ap.parse_args()
args = vars(parsed_args)

print("Use data: {}".format(os.path.abspath(args["path"])))
print(args)


def svg_to_pdf(in_path, out_path):
    # svg to pdf
    drawing = svg2rlg(in_path)
    renderPDF.drawToFile(drawing, out_path)


# Load doc2vec model
model = doc2vec.Doc2Vec.load(args['path'] + '/doc2vec.model')

# load document dataframe
data = pvtm_utils.load_document_dataframe('{}/documents.csv'.format(args['path']),
                                          ['gmm_topics', 'gmm_probas'])

# load topics dataframe
topics = pvtm_utils.load_topics_dataframe('{}/topics.csv'.format(args['path']))

# load gmm model
clf = joblib.load('{}/gmm.pkl'.format(args['path']))

# docvecs 
vectors = np.array(model.docvecs.vectors_docs).astype('float64')
vecs_with_center = pd.read_csv('{}/vectors_with_center.tsv'.format(args['path']), sep='\t', index_col=0)


def timelines(data, args):
    # if args.timelines or not(args.timeline or args.bhtsne or args.wordclouds):
    print('timelines')
    # timelines

    data.date = data.date.apply(lambda x: pd.to_datetime(x, errors='coerce'))
    data = pvtm_utils.extract_time_info(data, 'date')

    print('Extracted datetime information, starting topic importance aggregation..')
    topic_importance_df = pvtm_utils.get_topic_importance_df(args['agg_lvl'], data)
    topic_importance_df.to_csv('{}/timelines_df.csv'.format(args['path']))

    imp_per_my = topic_importance_df.copy()
    mean_of_means = imp_per_my.mean().mean()

    savepath = '{}/topics/timelines'.format(args['path'])
    print('Store timelines in folder:', savepath)
    pvtm_utils.check_path(savepath)

    for topic in data.gmm_top_topic.unique():
        plt.plot(imp_per_my[topic].mean(), c='b', label='Mean Probability current Topic', linestyle=':')
        plt.axhline(mean_of_means, c='r', label='Mean Probability all Topics', linestyle='--')

        imp_per_my[topic].plot(label='Probability', linestyle=':')

        plt.plot(np.nan, '-g', label='Number of Documents (right)')  # Make an agent for the twinx axis
        plt.legend(loc='best', prop={'size': 8})
        plt.title('Topic importance over time. Topic: {}'.format(topic))
        plt.grid()

        ##################
        # absolute articles timeline
        ##################

        granulars = data.sort_values('date')[args['agg_lvl']].unique()
        _list = [
            pvtm_utils.show_topics_per_choosen_granularity(data, 'gmm_top_topic', [topic], args['agg_lvl'], granular)
            for granular in granulars]
        df = pd.concat(_list).fillna(0)
        # df.index = new_index

        ax2 = plt.twinx()
        ax2.plot(df, c='g', label='Number of Documents', linestyle='--')
        plt.xlim(imp_per_my.index[0], imp_per_my.index[-1])
        #     plt.grid()
        #     plt.gca().set_position([0, 0, 1, 1])


        file_name = 'timeline_Topic_{}'.format(topic)
        plt.savefig('{}/topics/timelines/{}.svg'.format(args['path'], file_name), bbox_inches='tight')
        plt.savefig('{}/topics/timelines/{}.png'.format(args['path'], file_name), bbox_inches='tight')
        plt.close()
        # plt.show()


def bhtsne(vectors, vecs_with_center, args):
    # if args.bhtsne or not(args.timeline or args.bhtsne or args.wordclouds):
    # bhtnse
    print('Bhtsne..')
    Y = tsne(vectors, perplexity=args["tsne_perplexity"])
    pd.DataFrame(Y).to_csv('{}/bhtsne.csv'.format(args['path']))
    plt.scatter(Y[:, 0], Y[:, 1], s=0.3)
    plt.savefig('{}/bhtsne.svg'.format(args['path']), bbox_inches='tight')
    plt.savefig('{}/bhtsne.png'.format(args['path']), bbox_inches='tight')
    pd.DataFrame(Y).to_csv('{}/bhtsne_2d.csv'.format(args['path']))
    pvtm_utils.svg_to_pdf('{}/bhtsne.svg'.format(args['path']))
    plt.close()

    print('Bhtsne with center..')
    Y = tsne(vecs_with_center.values, perplexity=args["tsne_perplexity"])
    pd.DataFrame(Y).to_csv('{}/bhtsne_with_center.csv'.format(args['path']))
    plt.scatter(Y[:len(vectors), 0], Y[:len(vectors), 1], s=0.3)
    plt.scatter(Y[len(vectors):, 0], Y[len(vectors):, 1], s=0.8, c='r', marker='x')
    plt.savefig('{}/bhtsne_with_center.svg'.format(args['path']), bbox_inches='tight')
    plt.savefig('{}/bhtsne_with_center.png'.format(args['path']), bbox_inches='tight')
    pd.DataFrame(Y).to_csv('{}/bhtsne_with_center_2d.csv'.format(args['path']))

    pvtm_utils.svg_to_pdf('{}/bhtsne_with_center.svg'.format(args['path']))
    plt.close()

    print('3D tsne...')

    Y = tsne(vectors, dimensions=3, perplexity=args["tsne_perplexity"])
    fig = pyplot.figure(frameon=False, figsize=(8, 5))
    ax = Axes3D(fig)

    ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], s=1, c='b', marker='^')
    ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], s=20, c='r', marker='^')
    # pyplot.axis('off')
    xmax, ymax, zmax = Y[:len(vectors), 0].max(), Y[:len(vectors), 1].max(), Y[:len(vectors), 2].max()
    xmin, ymin, zmin = Y[:len(vectors), 0].min(), Y[:len(vectors), 1].min(), Y[:len(vectors), 2].min()

    ax.set_xlim(xmin + 4, xmax - 4)
    ax.set_ylim(ymin + 4, ymax - 4)
    ax.set_zlim(zmin + 4, zmax - 4)
    pyplot.savefig('{}/bhtsne_3d.svg'.format(args['path']), bbox_inches='tight')
    pyplot.savefig('{}/bhtsne_3d.png'.format(args['path']), bbox_inches='tight')
    pvtm_utils.svg_to_pdf('{}/bhtsne_3d.svg'.format(args['path']))

    pd.DataFrame(Y).to_csv('{}/bhtsne_3d.csv'.format(args['path']))

    Y = tsne(vecs_with_center.values, dimensions=3, perplexity=args["tsne_perplexity"])
    fig = pyplot.figure(frameon=False, figsize=(8, 5))
    ax = Axes3D(fig)

    ax.scatter(Y[:len(vectors), 0], Y[:len(vectors), 1], Y[:len(vectors), 2], s=1, c='b', marker='^')
    ax.scatter(Y[len(vectors):, 0], Y[len(vectors):, 1], Y[len(vectors):, 2], s=20, c='r', marker='^')
    # pyplot.axis('off')
    xmax, ymax, zmax = Y[:len(vectors), 0].max(), Y[:len(vectors), 1].max(), Y[:len(vectors), 2].max()
    xmin, ymin, zmin = Y[:len(vectors), 0].min(), Y[:len(vectors), 1].min(), Y[:len(vectors), 2].min()

    ax.set_xlim(xmin + 4, xmax - 4)
    ax.set_ylim(ymin + 4, ymax - 4)
    ax.set_zlim(zmin + 4, zmax - 4)
    pyplot.savefig('{}/bhtsne_with_center_3d.svg'.format(args['path']), bbox_inches='tight')
    pyplot.savefig('{}/bhtsne_with_center_3d.png'.format(args['path']), bbox_inches='tight')
    pvtm_utils.svg_to_pdf('{}/bhtsne_with_center_3d.svg'.format(args['path']))

    pd.DataFrame(Y).to_csv('{}/bhtsne_with_center_3d.csv'.format(args['path']))


def wordclouds(data, args):

    print('wordclouds..')
    pvtm_utils.check_path('{}/wordclouds'.format(args['path']))

    topicgroup = data.groupby('gmm_top_topic')
    for i, group in topicgroup:
        cc = [word.lower().strip().replace('ä', 'ae').replace('ü', 'ue').replace('ö', 'oe').replace('ß', 'ss') for _list
              in group.data.values for word in _list]

        with open('{}/wordclouds/topic_{}.txt'.format(args['path'], i), 'w', encoding='utf-8') as textfile:
            textfile.write('\n'.join(cc))

    # create pdf wordclouds
    commands = ["RScript", "wordclouds.R", args['path']]
    subprocess.call(commands)

    print('Wordclouds to svg..')
    command = 'FOR %A IN ({}\wordclouds\*.pdf) DO inkscape %A --export-plain-svg=%A.svg --export-area-drawing'.format(
        args['path'])
    os.system(command=command)

    print('Wordclouds to png..')
    command = 'FOR %A IN ({}\wordclouds\*.pdf) DO inkscape %A --export-png=%A.png --export-area-drawing -b "white" -d 800'.format(
        args['path'])
    os.system(command=command)


if parsed_args.timelines or not (parsed_args.timelines or parsed_args.bhtsne or parsed_args.wordclouds):
    timelines(data, args)

if parsed_args.bhtsne or not (parsed_args.timelines or parsed_args.bhtsne or parsed_args.wordclouds):
    bhtsne(vectors, vecs_with_center, args)

if parsed_args.wordclouds or not (parsed_args.timelines or parsed_args.bhtsne or parsed_args.wordclouds):
    wordclouds(data, args)

print('Finished')
