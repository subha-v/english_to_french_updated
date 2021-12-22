import unicodedata
import re # stands for redex

def unicode_to_ascii(word): #this changes stuff to ascii
    return "".join(char for char in unicodedata.normalize('NFD', word) if unicodedata.category(char) != 'Mn') # this is the opposite of split basically
    #not very important just makes us go from unicode to ascie

def normalize_word(word):
    word = unicode_to_ascii(word.lower().strip()) # made it lowercase 
    # hello! => hello ! 
    #seperate words from exclamation mark
    #this is called regex, it looks for patterns in the strings
    # we are looking for anything that is a .,!, or ? 
    word = re.sub(r"([.!?])", r" \1", word) # this says we want to replace these charactesr by a space and then the mark
    # we are going to replace it with a space and then adds the question mark:
    # like this hello? -> hello ?
    # now we want to get rid of those
    word = re.sub(r"[^a-zA-Z.!?]+", r" ", word) # replaces anything thats not a-z or A-Z or .!? and then replaces it with a space
    #the plus means one or more characters

    return word

    
