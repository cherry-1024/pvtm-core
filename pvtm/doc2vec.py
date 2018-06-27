import glob
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


def preprocess_documents(out, text_column='text'):
    lines = out[text_column].str.lower()
    preprocessed = []
    for t in lines[:]:
        t = pvtm_utils.preprocess(t)
        fixed = ''.join([x if x.isalnum() or x.isspace() else " " for x in t])
        preprocessed.append(fixed)
    print(len(preprocessed), 'documents after preprocessing')
    return preprocessed


def look_for_existing_lemmatized_dataset(path, filename):
    relevant_file = path + '/' + filename

    relevant_file_folder = os.path.split(relevant_file)[0]
    files = glob.glob(relevant_file_folder)
    print(files)

    print('Looking for file ', relevant_file)
    df = pd.read_csv(relevant_file)
    print(df.shape)
    print(df.head(1))
    return df.values.reshape(-1).tolist()

def lemmatize(preprocessed, LANGUAGE, LEMATIZER_N_THREADS, LEMMATIZER_BATCH_SIZE, OUTPUTPATH, FILENAME):
    try:
        data = look_for_existing_lemmatized_dataset(OUTPUTPATH, FILENAME)
        print('found lemmatized dataset')
    except Exception as e:
        print(e)
        print('start lemmatizer.. ')
        nlp = spacy.load(LANGUAGE)
        nlp.disable_pipes('tagger', 'ner')
        time0 = time.time()
        data = pvtm_utils.spacy_lemmatizer(preprocessed, nlp, LEMATIZER_N_THREADS, LEMMATIZER_BATCH_SIZE)
        savepath = OUTPUTPATH + '/' + FILENAME
        print(savepath)
        savepath_folder = os.path.split(savepath)[0]
        pvtm_utils.check_path(savepath_folder)

        print(savepath)
        pd.DataFrame(data).to_csv(savepath, index=False)
        print("Lemmatization took ", time.time() - time0, "sec")
    return data


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


def get_documents_from_text(out, LANGUAGE, COUNTVECTORIZER_MAXDF, COUNTVECTORIZER_MINDF, LEMATIZER_N_THREADS,
                            LEMMATIZER_BATCH_SIZE, OUTPUTPATH, FILENAME):
    """ Given a pandas dataframe this function generates documents for use in gensim."""

    preprocessed = preprocess_documents(out)
    data = lemmatize(preprocessed, LANGUAGE, LEMATIZER_N_THREADS, LEMMATIZER_BATCH_SIZE, OUTPUTPATH, FILENAME)
    vocabulary = get_vocabulary_from_tfidf(data, COUNTVECTORIZER_MINDF, COUNTVECTORIZER_MAXDF)

    stopwords, language = stopwords_generator.get_all_stopwords()
    print('len stopwords \n', len(stopwords))

    # popularity based pre-filtering. Ignore rare and common words. But we don't want stopwords and digits.
    pp = []
    for i, line in enumerate(data):
        rare_removed = list(filter(lambda word: word in vocabulary, line.split()))

        stops_removed = [word.strip() for word in rare_removed if word not in stopwords and not word.isdigit()]
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
               COUNTVECTORIZER_MAXDF, COUNTVECTORIZER_MINDF, LEMATIZER_N_THREADS, LEMMATIZER_BATCH_SIZE, OUTPUTPATH):
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
    out['text'] = out['text'].str.replace('ä', 'ae').replace('ü', 'ue').replace('ö', 'oe').replace('ß', 'ss')
    out['title'] = out['title'].str.replace('\t', '').replace('\r', '').replace('\n', '')

    print('prepare data for Doc2Vec..')
    documents = get_documents_from_text(out, LANGUAGE, COUNTVECTORIZER_MAXDF, COUNTVECTORIZER_MINDF,
                                        LEMATIZER_N_THREADS, LEMMATIZER_BATCH_SIZE, OUTPUTPATH, FILENAME)
    data = documents.documents
    out['data'] = data

    print('Doc2Vec training start')
    model = train_doc_2_vec(Doc2Vec_EPOCHS, EMBEDDING_DIM, documents, len(data), MODEL_SAVE_NAME)

    print('store document dataframe to ', DATAFRAME_NAME)
    out.to_csv(DATAFRAME_NAME, encoding='utf-8-sig')
    return out, model
