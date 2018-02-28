import random
from nltk.corpus import wordnet
from vocabulary.vocabulary import Vocabulary as vb

def get_wordnet_synsets(word):
    return wordnet.synsets(word)

def get_wordnet_definitions(word):
    return [s.definition for s in get_wordnet_synsets(word)]

def get_glosbe_definitions(word):
    """
    returns list of form
    '[{"text": "Someone who is from the hills; especially from a rural area, with a connotation of a lack of refinement or sophistication.", "seq": 0}, {"text": "someone who is from the hills", "seq": 1}, {"text": "A white person from the rural southern part of the United States.", "seq": 2}]'

    """
    return vb.meaning(word, format="list")

def get_a_definition(word):
    """
    Get a definition from any source, or None if not available
    """
    definitions = get_wordnet_definitions(word)

    if len(definitions) == 0:
        print(get_glosbe_definitions(word))
        print([g for g in get_glosbe_definitions(word)])
        definitions = [g['text'] for g in get_glosbe_definitions(word)]

    if len(definitions) != 0:
        return random.choice(definition)

    return None
