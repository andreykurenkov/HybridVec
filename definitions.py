from PyDictionary import PyDictionary
dictionary=PyDictionary()

def get_wordnet_definition(word):
    return dictionary.meaning(word)
