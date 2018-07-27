# coding: utf-8
# Topic Modelling using Document Vectors with Gaussian Mixture Clustering
# A Gaussian Mixture Model is used to cluster the resulting DocVecs into meaningful Topics.

# import the necessary packages
import argparse
import os
import json
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# general
ap.add_argument("-i", "--input", required=True,
                help="path to the input data file")
ap.add_argument("-o", "--output", default="./Output", required=False,
                help="Name of the folder where outputs are stored. Default = './Output'")
ap.add_argument("-l", "--language", default="en", required=False,
                help="Language of the text documents used for lemmatization. Default = 'en'")

# d2v
ap.add_argument("-d2vp", "--d2v_model", default="", required=False,
                help="Provide a path to a folder where a Doc2Vec.model file is stored. "
                     "No new model will be trained but the pre-trained model will be used instead.")
ap.add_argument("-gmmp", "--gmm_model", default="", required=False,
                help="Provide a path to a folder where a gmm.pkl file is stored. "
                     "No new model will be trained but the pre-trained model will be used instead.")
ap.add_argument("-e", "--epochs", default=15, required=False, type=int,
                help="Doc2Vec number epochs. Default = 15")
ap.add_argument("-d", "--dimension", default=100, required=False, type=int,
                help="Doc2Vec embedding dimension. Default = 100")

# preprocessing
ap.add_argument("-lt", "--lemmathreads", default=-1, required=False, type=int,
                help="Number of threads for the lemmatizer. Default = '-1'")
ap.add_argument("-lbs", "--lemmabatchsize", default=300, required=False, type=int,
                help="Batch size for lemmatizer. Default = '300'")
ap.add_argument("-vmin", "--vectorizermin", default=0.01, required=False, type=float,
                help="max number of documents in which a word has to appear to be considered. Default = 0.01")
ap.add_argument("-vmax", "--vectorizermax", default=0.75, required=False, type=float,
                help="max number of documents in which a word is allowed to appear to be considered. Default = 0.75")

# gmm
ap.add_argument("-gv", "--gmmverbose", default=1, required=False,
                help="GMM verbosity during training. Default = 1")
ap.add_argument("-gi", "--gmmninits", default=2, required=False,
                help="GMM number of initializations per Component size. Default = 2")
ap.add_argument("-gr", "--gmmrange", default=[10, 40, 4], nargs=3, type=int, metavar=('start', 'end', 'step'),
                help='specify a range', required=False)
ap.add_argument("-gcv", "--gmmcvtypes", nargs='+', default=['spherical', 'diag', 'tied'], required=False,
                help="GMM covariance matrix constraints. "
                     "Takes a  values from [spherical diag tied full]. Default = ['spherical','diag','tied']")
ap.add_argument("-ntp", "--numtopicwords", default=50, required=False,
                help="How many top words per topic to store. Default = 50")

ap.add_argument("-vis", "--visualizations", required=False,  action='store_true',
                help="If flag is active then pvtm_vis.py will be run on the pvtm outputs.")
ap.add_argument("-a", "--app", required=False,  action='store_true',
                help="If flag is active then the dash app for visualizations will be started on port 8050 using the just created outputs.")

parsed_args = ap.parse_args()
args = vars(parsed_args )
# display a friendly message to the user
print("Use data: {}".format(os.path.abspath(args["input"])))

print(args)

if __name__ == '__main__':

    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import re, json, random
    import subprocess
    from sklearn.externals import joblib
    from sklearn import mixture

    # custom functions
    import pvtm_utils
    import clustering
    import doc2vec

    import stopwords_generator

    pvtm_utils.check_path(args["output"])

    # store settings to file for later reference
    with open('{}/file.txt'.format(args['output']), 'w') as file:
        file.write(json.dumps(args))  # use `json.loads` to do the reverse

    args['gmmrange'] = range(args['gmmrange'][0], args['gmmrange'][1], args['gmmrange'][2])
    ###################################
    # # Load Model, Data and Stopwords
    ###################################
    # Load the specified data into a dataframe, 'out',
    # and load the trained Doc2Vec model(or train a new one, if NEW_Doc2Vec = 1).
    # train a new model if specified, otherwise load pretrained model
    if args['d2v_model'] == "":
        print('Training New Doc2Vec Model.')
        out, model = doc2vec.run_script(args["input"],
                                        args['output'] + '/doc2vec.model',
                                        args['output'] + '/documents.csv',
                                        args['epochs'],
                                        args['dimension'],
                                        args['language'],
                                        args['vectorizermax'],
                                        args['vectorizermin'],
                                        args['lemmathreads'],
                                        args['lemmabatchsize'],
                                        args['output']
                                        )

    else:
        print('Using pre-trained Doc2Vec Model.')
        model = doc2vec.Doc2Vec.load(args['d2v_model'] + '/doc2vec.model')

        # load document dataframe
        out = pvtm_utils.load_document_dataframe('{}/documents.csv'.format(args['output']),
                                                 ['gmm_topics', 'gmm_probas'])
        if args['gmm_model'] != "":
            print('Loading Topic Dataframe.')
            topics = pvtm_utils.load_topics_dataframe('{}/topics.csv'.format(args['output']))
            clf = joblib.load('{}/gmm.pkl'.format(args['gmm_model']))

    # store the DocVecs to tsv
    vectors = np.array(model.docvecs.vectors_docs)
    pd.DataFrame(vectors).to_csv('{}/vectors.tsv'.format(args['output']), sep='\t', header=False)

    # Detect the language of the documents and load the respective stopwords
    vocab = list(model.wv.vocab.keys())
    stopwordssss, LANGUAGE = stopwords_generator.get_all_stopwords(' '.join(vocab[:100]))

    print('Document Df: ', out.shape)
    print('Vectors: ', vectors.shape)

    if args['gmm_model'] == "":
        # ## GMM for Topic clustering
        #
        # We use a Gaussian Mixture Model to cluster the Document Vectors learned by the Doc2Vec model into soft topics.
        # The number of topics is optimized according to the Bayesian information criterion (BIC) score the GMM achieved on the dataset.
        # The model with the lowest BIC score is used for the final document clustering.

        # optimize the number of Topics (GMM-Components)
        clf, bic = clustering.optimize_gmm_components(vectors, args['gmmrange'], args['gmmcvtypes'], args['gmmninits'],
                                                      args['gmmverbose'])

        # plot the results from the optimization
        clustering.plot_BIC(bic, args['gmmrange'], args['gmmcvtypes'], args)
        print(args['gmmrange'])

        # parameters of the best gmm
        best_params = pd.DataFrame(pd.DataFrame(clf.get_params(), index=range(len(clf.get_params()))).iloc[0]).T
        gmm_center = clf.means_
        vectors_with_center = np.append(vectors, gmm_center, axis=0)
        pd.DataFrame(vectors_with_center).to_csv('{}/vectors_with_center.tsv'.format(args['output']), sep='\t',
                                                 header=False)

        joblib.dump(clf, '{}/gmm.pkl'.format(args['output']))
        print('Parameter: ', best_params.T)

        out, GMM_N_COMPONENTS = clustering.add_gmm_probas_to_out(out, vectors, clf)

        clustering.plot_topic_distribution(out, 'gmm_top_topic', FILENAME=args['output'] + '/GMM_Topics.png')


        """
        # Topic labeling

        Here the found clusters/topics are labelled in different ways.

        - The 'num_words' most frequent words found in the articles, stopwordssss excluded, are used to label the topic.
        - The most similar words to the embedding vector of each cluster center are used to further describe the cluster.
        - The most similar documents to the embedding vector of each cluster center get appended.

        The resulting dataframe, 'topics', holds a row for every cluster/topic with the described information.
        """

        print('find gmm clustercenter')
        # find the cluster center by taking the mean over the DocVecs that belong to a topic (hard vote style)
        center = clustering.get_gmm_cluster_center(GMM_N_COMPONENTS, out, vectors)

        # create a dataframe from the cluster center holding the topics
        print('extract topics...')
        topics, articles = pvtm_utils.get_all_topics_from_centers(center, out, 'gmm_top_topic', stopwordssss,
                                                                  num_words=args['numtopicwords'])
        pd.DataFrame(topics).to_csv(args['output'] + '/topics_single.csv')

        # Unpack tuples and rename columns
        topics = pd.DataFrame([list(zip(*[('', 0) if v is None else v for v in topic])) for topic in topics.values])
        topics.columns = ['top_words', 'top_words_count']

        # add the most representative documents to the topics dataframe
        # topics = topics.drop(['sim_words', 'sim_words_prob', 'sim_docs_indx', 'sim_docs_prob'], axis=1)
        simsdf = pvtm_utils.get_most_similar_words_and_docs(center, model, num_words=30, num_docs=30)
        topics = topics.join(simsdf)

        # store the dataframe with the cluster information (topics) and the
        # dataframe that holds the documents (out)
        print('store topics and out')
        topics.to_csv(args['output'] + '/topics.csv', encoding='utf-8-sig')
        out.to_csv(args['output'] + '/documents.csv', encoding='utf-8-sig')

    if parsed_args.visualizations:
        os.system('python pvtm/pvtm_vis.py -p {}'.format(args['output']))

    if parsed_args.app:
        os.system('python pvtm/pvtm_dash.py -i {}'.format(args['output']))




    print('All done')
