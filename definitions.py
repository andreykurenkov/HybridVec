import random
from nltk.corpus import wordnet
from vocabulary.vocabulary import Vocabulary as vb

def get_wordnet_definition(word):
    synsets = wordnet.synsets(word)
    return synsets.definition

def get_glosbe_definitions(word):
    """
    returns list of form
    '[{"text": "Someone who is from the hills; especially from a rural area, with a connotation of a lack of refinement or sophistication.", "seq": 0}, {"text": "someone who is from the hills", "seq": 1}, {"text": "A white person from the rural southern part of the United States.", "seq": 2}]'

    """
    return vb.meaning(word)

def get_a_definition(word):
    """
    Get a definition from any source, or None if not available
    """
    definition = get_wordnet_definition(word)
    print(definition)
    if definition is None:
        definition = random.choice(get_glosbe_definitions(word))['text']
    return definition
