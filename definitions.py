import random
from nltk.corpus import wordnet
from nltk.stem.snowball import SnowballStemmer
from vocabulary.vocabulary import Vocabulary as vb
import sys
from wordnik import swagger, WordApi

reload(sys)
sys.setdefaultencoding('utf8')

apiUrl = 'http://api.wordnik.com/v4'
apiKey = 'a1b28252f1c2bd049897a03d4e81e85c5d6dbca71cb8dcac8'
client = swagger.ApiClient(apiKey, apiUrl)
wordApi = WordApi.WordApi(client)
stemmer = SnowballStemmer("english")

def get_wordnik_definitions(word):
    defns = wordApi.getDefinitions(word)
    if defns is None:
        return []
    return [d.text for d in defns]

def get_wordnet_synsets(word):
    return wordnet.synsets(word)

def get_wordnet_definitions(word):
    return [s.definition() for s in get_wordnet_synsets(word)]

def get_glosbe_definitions(word):
    """
    returns list of form
    '[{"text": "Someone who is from the hills; especially from a rural area, with a connotation of a lack of refinement or sophistication.", "seq": 0}, {"text": "someone who is from the hills", "seq": 1}, {"text": "A white person from the rural southern part of the United States.", "seq": 2}]'
    """
    return vb.meaning(word, format="list")

def get_a_definition(word, filter_repetition = True):
    """
    Get a definition from any source, or None if not available
    
    Args:
        word: str of word to define
        filter_repetiyion: whether to remove reocurrence of word, eg healthy = the state of having health
    """
    definition = None
    definitions = get_wordnet_definitions(word)
    try:
        if not definitions:
            definitions = get_glosbe_definitions(word)

        #if not definitions:
        #    definitions = get_wordnik_definitions(word)
            
        if definitions:
            definition = str(random.choice(definitions))

        if definition and filter_repetition:
            stem = stemmer.stem(word)
            definition.replace(' %s '%stem,' stem ')
    except Exception as e:
        print(e)
        return None
    
    return definition
