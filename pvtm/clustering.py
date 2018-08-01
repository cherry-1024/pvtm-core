import itertools
import time
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvtm_utils
from scipy import linalg
from sklearn import mixture
from sklearn.cluster import KMeans, MeanShift


def kmeans_cluster(NUM_CLUSTERS, vectors):
    t0 = time.time()
    kcluster = KMeans(n_clusters=NUM_CLUSTERS, init='k-means++', n_init=10, max_iter=30, tol=0.0001,
                      precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=4,
                      algorithm='auto')
    assigned_clusters = kcluster.fit_predict(vectors)
    center = kcluster.cluster_centers_
    print('KMeans Clustering took {:.2f} seconds'.format(time.time() - t0))
    print('Found {} clusters'.format(len(center)))
    return kcluster, assigned_clusters, center


def meanshift_cluster(bandwidth, vectors):
    """ Mean shift clustering. Finds bin centers via sklearns meanshift clustering. """

    t0 = time.time()
    if not bandwidth:
        print('no bandwidth given, will estimate best.')
        mscluster = MeanShift(seeds=None, bin_seeding=False, min_bin_freq=1,
                              cluster_all=True, n_jobs=-1)
    else:

        mscluster = MeanShift(bandwidth=bandwidth, seeds=None, bin_seeding=False, min_bin_freq=1,
                              cluster_all=True, n_jobs=-1)
    assigned_clusters = mscluster.fit_predict(vectors)
    center = mscluster.cluster_centers_
    print('MeanShift Clustering took {:.2f} seconds'.format(time.time() - t0))
    print('Found {} clusters with bandwith = {}'.format(len(center), bandwidth))
    return mscluster, assigned_clusters, center


def optimize_gmm_components(vectors, n_components_range, cv_types, GMM_N_INITS, GMM_VERBOSITY):
    """

    :param vectors:
    :param n_components_range:
    :param cv_types:
    :param GMM_N_INITS:
    :param GMM_VERBOSITY:
    :return:  return clf, bic
    """
    X = vectors

    lowest_bic = np.infty
    bic = []

    for cv_type in cv_types:
        for n_components in n_components_range:
            print(cv_type, n_components)
            # Fit a Gaussian mixture with EM
            t0 = time.time()
            gmm = mixture.GaussianMixture(n_components=n_components,
                                          covariance_type=cv_type,
                                          verbose=GMM_VERBOSITY,
                                          n_init=GMM_N_INITS)

            gmm.fit(X)
            print('BIC: {}'.format(gmm.bic(X)))
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
            print('training took {:.2f} seconds'.format(time.time() - t0))
    bic = np.array(bic)
    clf = best_gmm

    return clf, bic


# group vectors by topic and take the mean to approximate the cluster center
def get_gmm_cluster_center(GMM_N_COMPONENTS, out, vectors):
    """
    Approximates cluster centers for a given clustering from a GMM.
    Only takes the topic with the highest probability per document into account.
    Averaging the document vectors per topic cluster provides the cluster center for the topic.
    Returns a list of the cluster centers.
    """
    gmm_clustercenter = []
    grouped_out = out.groupby('gmm_top_topic').indices
    for i in range(GMM_N_COMPONENTS):
        gmm_centerindexe = grouped_out.get(i)
        gmm_clustercenter.append(vectors[gmm_centerindexe].mean(0))

    return gmm_clustercenter


def add_gmm_probas_to_out(out, vectors, clf):
    import numpy as np
    import pandas as pd
    gm = clf

    # store the number of components from the best classifier
    GMM_N_COMPONENTS = gm.get_params()['n_components']

    # predict topic probabilites for documents
    gmprobs = gm.predict_proba(vectors)

    # get the index with the highest prob
    gm_top_probs = gmprobs.argmax(-1)

    # get all topics with prob above threshold
    gmm_topics = [np.where(probs > 0.005)[0].tolist() for probs in gmprobs]

    # join  gmm probabilities and topics on dataframe
    out['gmm_probas'] = pd.Series(gmprobs.tolist()).values
    out['gmm_topics'] = gmm_topics
    out['gmm_top_topic'] = gm_top_probs
    return out, GMM_N_COMPONENTS


def plot_BIC(bic, N_COMPONENTS_RANGE, CV_TYPES, args):
    import pandas as pd
    bic_scores_df = pd.DataFrame(bic.reshape(-1, len(N_COMPONENTS_RANGE))).T
    bic_scores_df.columns = CV_TYPES
    bic_scores_df.index = N_COMPONENTS_RANGE
    best_bics = pd.DataFrame(list(zip(bic_scores_df.idxmin().values, bic_scores_df.min().values)),
                             columns=['Index', 'BIC Score'],
                             index=CV_TYPES)
    print(best_bics)

    bic_scores_df.plot(kind='bar', figsize=(24, 4))
    upper, lower = bic_scores_df.max().max(), bic_scores_df.min().min()
    add = abs((upper - lower) / 15)
    plt.ylim(lower-add, upper+add)
    # plttmp = bic_scores_df.plot(kind='bar', figsize=(24, 4), title='A')
    # fig = plttmp.get_figure()
    plt.grid()
    plt.savefig('{}/BIC_scores.svg'.format(args['output']), bbox_inches='tight')
    plt.savefig('{}/BIC_scores.png'.format(args['output']), bbox_inches='tight')
    pvtm_utils.svg_to_pdf('{}/BIC_scores.svg'.format(args['output']))
    bic_scores_df.to_csv('{}/BIC_scores.csv'.format(args['output']))
    plt.clf()


def plot_topic_distribution(out, column_name, FILENAME, amount=100):
    tmpplot = out[column_name].value_counts().iloc[:amount].plot(kind='bar', figsize=(24, 4), color='k')

    plt.grid()
    fig = tmpplot.get_figure()
    plt.title('Number of Documents in top 100 most frequent topics')
    plt.savefig('{}'.format(FILENAME), bbox_inches='tight')
    plt.savefig('{}.svg'.format(FILENAME), bbox_inches='tight')
    return fig
