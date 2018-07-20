

# import the necessary packages
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals import joblib
from bhtsne import tsne

# custom functions
import pvtm_utils
import doc2vec
import subprocess
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True,
                help="path to the output data folder")
ap.add_argument("-tp", "--tsne_perplexity", default=30, required=False,
                help="Perplexity value for tsne")
ap.add_argument("-al", "--agg_lvl", default='year', required=False,
                help="Unit of aggregation for topic importance over time. Can be 'year', 'month','week','day'. Maybe more.")
args = vars(ap.parse_args())
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



# timelines
agg_lvl = args['agg_lvl']
print(agg_lvl)
out = data.copy()

out.date = out.date.apply(lambda x: pd.to_datetime(x, errors='coerce'))
out = pvtm_utils.extract_time_info(out, 'date')

print('Extracted datetime information, starting topic importance aggregation..')
topic_importance_df = pvtm_utils.get_topic_importance_df(agg_lvl, out)
new_index = pd.DatetimeIndex([pd.to_datetime('{}-01-01'.format(t)) for t in topic_importance_df.index.values])
topic_importance_df.index = new_index

top_n_trending_topics = pvtm_utils.get_top_n_trending_topics(topic_importance_df, 1,'gmm_top_topic')

imp_per_my=topic_importance_df.copy()
mean_of_means = imp_per_my.mean().mean()

pvtm_utils.check_path('{}/topics/timelines'.format(args['path']))

for topic in out.gmm_top_topic.unique():
    plt.axhline(imp_per_my[topic].mean(), c='b', label='Mean Probability current Topic', linestyle=':')
    plt.axhline(mean_of_means, c='r', label='Mean Probability all Topics', linestyle='--')

    imp_per_my[topic].plot(label='Probability', linestyle=':')

    plt.plot(np.nan, '-g', label='Number of Documents (right)')  # Make an agent for the twinx axis
    plt.legend(loc='best', prop={'size': 8})
    plt.title('Topic importance over time. Topic: {}'.format(topic))
    plt.grid()

    ##################
    # absolute articles timeline
    ##################

    granulars = out.sort_values('date')[agg_lvl].unique()
    _list = [pvtm_utils.show_topics_per_choosen_granularity(out, 'gmm_top_topic', [topic], agg_lvl, granular)
             for granular in granulars]
    df = pd.concat(_list).fillna(0)
    df.index = new_index

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


# bhtnse
print('Bhtsne..')
Y = tsne(vectors, perplexity=args["tsne_perplexity"])
pd.DataFrame(Y).to_csv('{}/bhtsne.csv'.format(args['path']))
plt.scatter(Y[:, 0], Y[:, 1], s=2)
plt.savefig('{}/bhtsne.svg'.format(args['path']), bbox_inches='tight')
plt.savefig('{}/bhtsne.png'.format(args['path']), bbox_inches='tight')
plt.close()



print('Bhtsne with center..')
Y = tsne(vecs_with_center.values, perplexity=args["tsne_perplexity"])
pd.DataFrame(Y).to_csv('{}/bhtsne_with_center.csv'.format(args['path']))
plt.scatter(Y[:len(vectors), 0], Y[:len(vectors), 1], s=2)
plt.scatter(Y[len(vectors):, 0], Y[len(vectors):, 1], s=3, c='r', marker='x')
plt.savefig('{}/bhtsne_with_center.svg'.format(args['path']), bbox_inches='tight')
plt.savefig('{}/bhtsne_with_center.png'.format(args['path']), bbox_inches='tight')
plt.close()

# wordclouds
print('wordclouds..')
pvtm_utils.check_path('{}/wordclouds'.format(args['path']))

topicgroup = data.groupby('gmm_top_topic')
for i,group in topicgroup:

    cc = [word.lower().strip().replace('ä', 'ae').replace('ü', 'ue').replace('ö', 'oe').replace('ß', 'ss') for _list
      in group.data.values for word in _list]

    with open('{}/wordclouds/topic_{}.txt'.format(args['path'], i), 'w', encoding='utf-8') as textfile:

        textfile.write('\n'.join(cc))
commands = ["RScript", "wordclouds.R", args['path']]
subprocess.call(commands)

print('Clean wordcloud svgs')
pvtm_utils.clean_svg(args['path'])

print('Wordclouds to png..')
command = 'FOR %A IN ({}\wordclouds\*.svg) DO inkscape %A --export-png=%A.png --export-area-drawing -b "white" -d 800'.format(args['path'])
os.system(command=command)

print('Finished')