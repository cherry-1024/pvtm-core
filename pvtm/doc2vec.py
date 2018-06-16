import gensim
import os
import pandas as pd
# custom
import pvtm_utils
import re
import spacy
import stopwords_generator
import time
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

print('gensim version: ', gensim.__version__)


def get_documents_from_text(out, LANGUAGE, COUNTVECTORIZER_MAXDF, COUNTVECTORIZER_MINDF, LEMATIZER_N_THREADS,
                            LEMMATIZER_BATCH_SIZE):
    """ Given a pandas dataframe this function generates documents for use in gensim."""

    # lowercase text values
    lines = out.text.str.lower()
    preprocessed = []
    for t in lines[:]:
        t = pvtm_utils.preprocess(t)
        fixed = ''.join([x if x.isalnum() or x.isspace() else " " for x in t])
        preprocessed.append(fixed)
    print(len(preprocessed))

    print('start lemmatizer.. ')
    nlp = spacy.load(LANGUAGE)
    nlp.disable_pipes('tagger', 'ner')
    time0 = time.time()
    data = pvtm_utils.spacy_lemmatizer(preprocessed, nlp, LEMATIZER_N_THREADS, LEMMATIZER_BATCH_SIZE)
    print("Lemmatization took ", time.time() - time0, "sec")

    print('start vectorizer')
    vec = TfidfVectorizer(min_df=COUNTVECTORIZER_MINDF,
                          max_df=COUNTVECTORIZER_MAXDF,
                          stop_words='english')
    X = vec.fit_transform(data)
    print('finished vectorizer')

    allowed_vocab = list(vec.vocabulary_.keys())
    common_words = set(allowed_vocab)
    print(len(allowed_vocab), ' allowed words')

    stopwords, language = stopwords_generator.get_all_stopwords()
    print('len stopwords \n', len(stopwords))

    pp = []
    for i, line in enumerate(preprocessed):
        # popularity based pre-filtering. Ignore rare and common words. But we don't want stopwords and digits.
        rare_removed = list(filter(lambda word: word in common_words, line.split()))

        stops_removed = [word.strip().replace('ä', 'ae').replace('ü', 'ue').replace('ö', 'oe').replace('ß', 'ss')
                         for word in rare_removed if word not in stopwords and not word.isdigit()]
        pp.append(stops_removed)

    print('finish preprocessing')
    documents = pvtm_utils.Documents(pp)

    return documents


def train_doc_2_vec(Doc2Vec_EPOCHS, EMBEDDING_DIM, documents, count, MODEL_SAVE_NAME):
    print('Initialize Model..')
    model = Doc2Vec(size=EMBEDDING_DIM,
                    dbow_words=1,
                    dm=0,
                    iter=1,
                    window=5,
                    seed=123,
                    min_count=5,
                    workers=6,
                    alpha=0.025,
                    min_alpha=0.025)

    print('Building vocab')
    model.build_vocab(documents)

    losses = []
    for epoch in range(Doc2Vec_EPOCHS):
        print("epoch " + str(epoch))
        model.train(documents, total_examples=count, epochs=1, compute_loss=True)
        losses.append(model.get_latest_training_loss())
        model.save(MODEL_SAVE_NAME)
        model.alpha -= 0.002  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay

    return model


def run_script(FILENAME, MODEL_SAVE_NAME, DATAFRAME_NAME, Doc2Vec_EPOCHS, EMBEDDING_DIM, LANGUAGE,
               COUNTVECTORIZER_MAXDF, COUNTVECTORIZER_MINDF, LEMATIZER_N_THREADS, LEMMATIZER_BATCH_SIZE):
    print('Load data..')
    out = pd.read_csv('{}'.format(FILENAME))
    print('Original shape:', out.shape)
    out = out.dropna().reset_index()
    print('Shape after dropping nans:', out.shape)
    print('lowercasing text.')
    out['text'] = out['text'].str.lower()
    out['text'] = out['text'].str.replace('\n', '')
    out['text'] = out['text'].str.replace('\r', '')
    out['text'] = out['text'].str.replace('\t', '')
    out['title'] = out['title'].str.replace('\t', '').replace('\r', '').replace('\n', '')

    print('prepare data for Doc2Vec..')
    documents = get_documents_from_text(out, LANGUAGE, COUNTVECTORIZER_MAXDF, COUNTVECTORIZER_MINDF,
                                        LEMATIZER_N_THREADS, LEMMATIZER_BATCH_SIZE)
    data = documents.documents
    out['data'] = data
    print('Doc2Vec training start')
    model = train_doc_2_vec(Doc2Vec_EPOCHS, EMBEDDING_DIM, documents, len(data), MODEL_SAVE_NAME)
    print('store document dataframe to ', DATAFRAME_NAME)
    out.to_csv(DATAFRAME_NAME, encoding='utf-8-sig')
    return out, model
