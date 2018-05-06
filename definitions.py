import random
import sys
import string
import traceback
import re
import pyfscache
from nltk.corpus import wordnet
from nltk.stem.snowball import SnowballStemmer
import wikipedia
from wordnik import swagger, WordApi

# import os
# print os.sys.path
# sys.path.insert(0, "/Users/rohunsaxena/Documents/def2vec/libs/")

# print os.sys.path
#from vocabulary.vocabulary import Vocabulary as vb


reload(sys)
sys.setdefaultencoding('utf8')

apiUrl = 'http://api.wordnik.com/v4'
apiKey = 'a1b28252f1c2bd049897a03d4e81e85c5d6dbca71cb8dcac8'
client = swagger.ApiClient(apiKey, apiUrl)
wordApi = WordApi.WordApi(client)
stemmer = SnowballStemmer("english")

fs_cache = pyfscache.FSCache('data/cache/')
wikipedia.set_rate_limiting(True)

DEBUG = False
PUNC = set(string.punctuation)
def clean_str(string):
    """
    Cleans a str by making it all lower case, removing punctuation, and removing any html

    Args:
        string: the str to clean
    Returns:
        the cleaned string
    """
    if string=='<unk>':
        return string
    no_punc = "".join([c if c not in PUNC else " " for c in string.lower()])
    no_html = re.sub('<[^<]+?>', '', no_punc)
    return no_html


@fs_cache 
def get_wiki_summary(word, sentences = 1):
    try:
        return wikipedia.summary(word, sentences=sentences).strip()
    except:#ignore 404
        if DEBUG:
            traceback.print_exc()
        return []

@fs_cache
def get_wordnik_definitions(word):
    try: 
        defns = wordApi.getDefinitions(word)
    except Exception as e:#ignore 404
        if DEBUG:
            traceback.print_exc()
        if '401' in str(e):
            return None
        return []
    if defns is None:
        return []
    return [d.text for d in defns]

def get_wordnet_synsets(word):
    return wordnet.synsets(word)

@fs_cache
def get_wordnet_definitions(word):
    return [s.definition() for s in get_wordnet_synsets(word)]

@fs_cache
def get_glosbe_definitions(word):
    """
    returns list of form
    '[{"text": "Someone who is from the hills; especially from a rural area, with a connotation of a lack of refinement or sophistication.", "seq": 0}, {"text": "someone who is from the hills", "seq": 1}, {"text": "A white person from the rural southern part of the United States.", "seq": 2}]'
    """
    return vb.meaning(word, format="list")

def get_a_definition(word, filter_repetition = False):
    """
    Get a definition from any source, or None if not available
    
    Args:
        word: str of word to define
        filter_repetition: whether to remove reocurrence of word, 
                           eg healthy = the state of having health
                           will become
                              healthy = the state of having stem
    """
    definition = None
    definitions = get_wordnet_definitions(word)
    try:
        #if not definitions:
        #    definitions = get_glosbe_definitions(word)

        #if not definitions:
        #    definitions = get_wordnik_definitions(word)
            
        if definitions:
            definition = str(random.choice(definitions))

        if definition and filter_repetition:
            stem = stemmer.stem(word)
            definition.replace(' %s '%stem,' stem ')
    except Exception as e:
        traceback.print_exc()
        return None
    
    return definition

def combine_defs(current_list, new_list):
    """
    Helper method for combining lists of definitions


    Args:
        current_list: 
        new_list:  
    """
    if new_list:
        for definition in new_list:
            clean_def = clean_str(definition)
            if clean_def not in current_list:
                current_list.add(clean_def)

def get_definitions_concat(word, filter_repetition = False, concat_str=" . . . "):
    """
    Get definitions as one str from all source, or None if not available
    
    Args:
        word: str of word to define
        filter_repetition: whether to remove reocurrence of word, 
                           eg healthy = the state of having health
                           will become
                              healthy = the state of having stem
        concat_str: the str to put between definitions
    """
    definitions = None
    try:
	definitions = get_wordnet_definitions(word)
	#combine_defs(definitions, get_wordnet_definitions(word))
	#combine_defs(definitions, get_glosbe_definitions(word))
	#combine_defs(definitions, get_wordnik_definitions(word))
            
        if definitions:
            definitions = concat_str.join(definitions)
        else:
            definitions = None

        if definitions and filter_repetition:
            stem = stemmer.stem(word)
            definition.replace(' %s '%stem,' stem ')
    except Exception as e:
        traceback.print_exc()
        return None
    
    return definitions
