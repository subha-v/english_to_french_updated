from unicodedata import normalize
from utils import *
import random
import torch
SOS_TOKEN = 0
EOS_TOKEN = 1

# we're taking the part of the dataset thats 10 or some word s less and starts with english_prefixes


english_prefixes = ("i am", "he is", "i m ", "he is", "he s " "she is", "she s", 
"you are", "you re", "we are", "we re" "they are", "they re")

class Vocabulary:
    def __init__(self, language_name):
        self.language_name = language_name
        self.word_to_index = {"EOS": EOS_TOKEN, "SOS": SOS_TOKEN}
        self.word_to_count = {"EOS": 1, "SOS":1}
        self.index_to_word = {SOS_TOKEN: "SOS", EOS_TOKEN: "EOS"}
        self.num_words = 2
        

    def add_word(self, word):
        #only add a word to the dictionary if its not in it
        if word not in self.word_to_index:
            self.word_to_index[word] = self.num_words
            self.word_to_count[word] = 1
            self.index_to_word[self.num_words] = word # inverse mapping
            self.num_words += 1


    def add_phrase(self, phrase):
        words = phrase.split(" ")
        for word in words:
            self.add_word(word)


def create_sanitized_dataset(dataset_path):
    lines = open(dataset_path, encoding ='utf-8').read().strip().split("\n")

    pairs = [[normalize_word(word) for word in line.split('\t')] for line in lines]
    with open('data/eng_fr_train.txt', 'w') as f:
        with open('data/eng_fr_val.txt', 'w') as f2:
            for pair in pairs:
                #if(len(pair[0].split(' ')) !=8 ): # we need all the sentences to have the same length to stack them
                    #continue
                if(random.random() < 0.8):

                    f.write(f"{pair[0]}\t{pair[1]}\n")
                else:
                    f2.write(f"{pair[0]}\t{pair[1]}\n")

def create_pairs(dataset_path):
    lines = open(dataset_path, encoding ='utf-8').read().strip().split("\n")

    pairs = [[normalize_word(word) for word in line.split('\t')] for line in lines]
    with open("data/eng_fr_pairs.txt", 'w') as f:
        for pair in pairs:
            f.write(f"{pair[0]}\t{pair[1]}\n")

def load_vocabs(input_vocab, target_vocab):
    lines = open('data/eng_fr_pairs.txt', encoding="utf-8").read().strip().split("\n")
    pairs = [line.split('\t') for line in lines]
    for input_phrase, target_phrase in pairs:
        input_vocab.add_phrase(input_phrase)
        target_vocab.add_phrase(target_phrase)

    return input_vocab, target_vocab




class TranslationDataset:
    def __init__(self, dataset_path, language_name1, language_name2):
        self.dataset_path = dataset_path
        self.max_length = 10 #max length of a dataset since our thing is large
        lines = open(self.dataset_path, encoding ='utf-8').read().strip().split("\n")
        # this creates a list of all of the words
        # the split function creates a list of all the words
        # the delimiter between the english and french is the tab
        # we want to split on the tab therefore
        # we want to create english french pairs
        #pairs = [[normalize_word(word) for word in line.split('\t')] for line in lines]
        pairs = [line.split('\t') for line in lines] #line.split gives a list for all the 
        
        #this normalizes all the words in the english and french pairs
        # so for example ['go', 'va !']
        # the outer list comprehension does this for all the pairs in the dataset
        self.input_vocab = Vocabulary(language_name1)
        self.target_vocab = Vocabulary(language_name2)
        #elem for elem in lst if elem %2 == 0
        self.pairs = [pair for pair in pairs if self.is_simple_sentence(pair)]
        # this makes the pairs only the one with simple sentences
        # for input_pair, target_pair in pairs:
        #     self.input_vocab.add_phrase(input_pair)
        #     self.target_vocab.add_phrase(target_pair)
        #     #this adds english and french words tot he inuput vocab and target vocab classes
        self.input_vocab, self.target_vocab = load_vocabs(self.input_vocab, self.target_vocab)
        # we call this on the input vocab and target vocab as we load the phrases and then store the input and target vocab in these variables


    def __len__(self):
        return len(self.pairs) # this returns the length of a dataset


    # this is a python built in method that gets an item 
    def __getitem__(self, idx): 
        pair = self.pairs[idx]
        lang1_indicies = self.sentenceToIndicies(pair[0], self.input_vocab)
        lang2_indicies = self.sentenceToIndicies(pair[1], self.target_vocab)

        lang1_idx_tensor = torch.tensor(lang1_indicies,dtype = torch.long)
        lang2_idx_tensor = torch.tensor(lang2_indicies, dtype = torch.long)

        return lang1_idx_tensor, lang2_idx_tensor



    def is_simple_sentence(self, pair):
        cond1 = len(pair[0].split(' '))< self.max_length #this checks if there are more than 10 words
        cond2 = len(pair[1].split(' ')) < self.max_length # this checks if there are more than 10 words int he french sentence
        cond3 = pair[0].startswith(english_prefixes) # using starts with to show if it starts with an english prefixes

        return(cond1 and cond2 and cond3) #returns true or false based on this 

    def sentenceToIndicies(self, sentence, vocab):
        indicies = []


        words = sentence.split(" ")
        for word in words:
            idx = vocab.word_to_index[word]
            indicies.append(idx)

        indicies.append(EOS_TOKEN)
        return indicies


    def indicies_to_sentence(self, indicies, vocab):
        sentence = "" + vocab.index_to_word[indicies[0]]
        for i in range (1, len(indicies)):
            sentence+= " " + vocab.index_to_word[indicies[i]]

        return sentence


    def get_random_sample(self):
        rand_idx = random.randint(0,len(self.pairs)-1) # this gets a random pair
        return self[rand_idx]


    def idx_tensor_to_words(self, idx_tensor, vocab):
        indicies = [idx.item() for idx in idx_tensor]
        words = [vocab.index_to_word[idx] for idx in indicies]
        return words


     


        


        












