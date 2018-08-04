# import the necessary packages
import argparse
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# custom functions
import pvtm_utils
from bhtsne import tsne
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition.pca import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True,
                help="path to the output data folder")
ap.add_argument("-tp", "--tsne_perplexity", default=15, required=False,
                help="Perplexity value for tsne")
ap.add_argument("-al", "--agg_lvl", default='year', required=False,
                help="Unit of aggregation for topic importance over time. Can be 'year', 'month','week','day'. Maybe more.")
ap.add_argument("-vmin", "--vectorizermin", default=0.01, required=False, type=float,
                help="max number of documents in which a word has to appear to be considered. Default = 0.01")
ap.add_argument("-vmax", "--vectorizermax", default=0.65, required=False, type=float,
                help="max number of documents in which a word is allowed to appear to be considered. Default = 0.65")

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


def timelines(data, args):
    print('timelines')
    data.date = data.date.apply(lambda x: pd.to_datetime(x, errors='coerce'))
    data = pvtm_utils.extract_time_info(data, 'date')

    print('Extracted datetime information, starting topic importance aggregation..')
    topic_importance_df = pvtm_utils.get_topic_importance_df(args['agg_lvl'], data)
    topic_importance_df.to_csv('{}/timelines_df.csv'.format(args['path']))

    mean_of_means = topic_importance_df.mean().mean()

    savepath = '{}/topics/timelines'.format(args['path'])
    print('Store timelines in folder:', savepath)
    pvtm_utils.check_path(savepath)

    for topic in data.gmm_top_topic.unique():
        topic_importance_df.loc[:, topic].ewm(span=3).mean().plot(label='Probability', linestyle=':')
        plt.axhline(topic_importance_df[topic].mean(), c='b', label='Mean Probability current Topic', linestyle=':')
        plt.axhline(mean_of_means, c='r', label='Mean Probability all Topics', linestyle='--')

        plt.plot(np.nan, '-g', label='Number of Documents (right)')  # Make an agent for the twinx axis
        plt.legend(loc='best', prop={'size': 8})
        plt.title('Topic importance over time. Topic: {}'.format(topic))
        plt.grid()

        timesteps = data.sort_values('date')[args['agg_lvl']].unique()
        _list = [
            pvtm_utils.show_topics_per_choosen_granularity(data, 'gmm_top_topic', [topic], args['agg_lvl'], granular)
            for granular in timesteps]
        df = pd.concat(_list).fillna(0)

        ax2 = plt.twinx()
        ax2.plot(df.ewm(span=3).mean(), c='g', label='Number of Documents', linestyle='--')
        plt.xlim(topic_importance_df.index[0], topic_importance_df.index[-1])

        file_name = 'timeline_Topic_{}'.format(topic)
        plt.savefig('{}/topics/timelines/{}.svg'.format(args['path'], file_name), bbox_inches='tight')
        plt.savefig('{}/topics/timelines/{}.png'.format(args['path'], file_name), bbox_inches='tight')
        plt.close()
        # plt.show()


def bhtsne(vectors, vecs_with_center, args):
    # if args.bhtsne or not(args.timeline or args.bhtsne or args.wordclouds):
    # bhtnse


    pca = PCA(n_components=50)
    vectors = pca.fit_transform(vectors)

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


def get_vocabulary_from_tfidf(data, COUNTVECTORIZER_MINDF, COUNTVECTORIZER_MAXDF):
    print('start vectorizer')
    vec = TfidfVectorizer(min_df=COUNTVECTORIZER_MINDF,
                          max_df=COUNTVECTORIZER_MAXDF,
                          stop_words='english')

    vec.fit(data)
    print('finished vectorizer')

    vocabulary = set(vec.vocabulary_.keys())
    print(len(vocabulary), 'words in the vocabulary')

    return vocabulary


def wordclouds(data, args):
    print('wordclouds..')
    pvtm_utils.check_path('{}/wordclouds'.format(args['path']))

    print('get tf-idf vocabulary..')
    vocabulary = get_vocabulary_from_tfidf(data.text.values, args['vectorizermin'], args['vectorizermax'])
    #
    stopwords = pvtm_utils.get_all_stopwords()
    print('# stopwords:', len(stopwords))

    # popularity based pre-filtering. Ignore rare and common words. And we don't want stopwords and digits.
    print('start pop based prefiltering')
    pp = []
    for i, line in enumerate(data.text.values):
        rare_removed = list(filter(lambda word: word in vocabulary, line.split()))

        stops_removed = [word.strip() for word in rare_removed if word not in stopwords and not word.isdigit()]
        pp.append(stops_removed)

    print('finished pop based prefiltering')

    data['data_clean'] = pp
    topicgroup = data.groupby('gmm_top_topic')

    for i, group in topicgroup:
        cc = [word.lower().strip().replace('ä', 'ae').replace('ü', 'ue').replace('ö', 'oe').replace('ß', 'ss') for _list
              in group['data_clean'].values for word in _list]

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


if __name__ == "__main__":

    model, clf, data, topics = pvtm_utils.load_pvtm_outputs(args['path'])

    # docvecs
    vectors = np.array(model.docvecs.vectors_docs).astype('float64')
    vecs_with_center = pd.read_csv('{}/vectors_with_center.tsv'.format(args['path']), sep='\t', index_col=0)

    if parsed_args.timelines or not (parsed_args.timelines or parsed_args.bhtsne or parsed_args.wordclouds):
        timelines(data, args)

    if parsed_args.bhtsne or not (parsed_args.timelines or parsed_args.bhtsne or parsed_args.wordclouds):
        bhtsne(vectors, vecs_with_center, args)

    if parsed_args.wordclouds or not (parsed_args.timelines or parsed_args.bhtsne or parsed_args.wordclouds):
        wordclouds(data, args)

    print('Finished')
