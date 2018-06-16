import nltk
import stop_words
from langdetect import detect

nltk.download('stopwords')
from nltk.corpus import stopwords
import ast


def get_all_stopwords(text_sample='This is an englisch sentence'):
    """ Combines Stopwords for englisch, german, french, and spanish from NLTK. Further adds stopwords from the stop_words module.
    Finally, stopwords from a text file stopwords.txt are added to come up with a list of stopwords."""
    # detect language
    lang = detect(text_sample)
    print('DETECTED LANGUAGE : {}'.format(lang))

    # get nltk stopwords for common languages
    stopwordssss = stopwords.words('german') + \
                   stopwords.words('english') + \
                   stopwords.words('french') + \
                   stopwords.words('spanish')

    # read from stopwords.txt file
    aa = []
    with open('stopwords.txt', encoding='utf-8-sig') as f:
        aa.append(f.read())
    stopword_dict = ast.literal_eval(aa[0])

    # join stop words from nltk, txt and from library stop_words
    stopwordss = set(stopwordssss) | set(stop_words.get_stop_words(lang)) | set(stopword_dict)
    stopwordlist = [*stopwordss]

    return stopwordlist, lang


def _find_language(text):
    if text != '':
        return detect(text[:5000])
