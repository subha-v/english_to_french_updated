from torch.nn.functional import dropout
from model import *
from data import *
from utils import *
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class TranslatorInference:
    def __init__(self, save_encoder_path, save_decoder_path, show_context_weights, arch, hidden_dim , dropout,
                train_dataset_path, source_language, target_language):

        self.save_encoder_path = save_encoder_path
        self.save_decoder_path = save_decoder_path
        self.show_context_weights = show_context_weights
        self.arch = arch 
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.source_language = source_language
        self.target_language = target_language

        encoder_type = EncoderLSTM if arch =="LSTM" else EncoderGRUWeighted
        decoder_type = DecoderLSTM if arch == "LSTM" else DecoderGRUWeighted
        self.train_dataset_path = train_dataset_path 
        self.train_data = TranslationDataset(self.train_dataset_path, self.source_language, self.target_language)


        # attributes allow us to access things everywhere in the class

        self.encoder = encoder_type(self.train_data.input_vocab.num_words, hidden_dim)
        if(self.target_language == "english"):
            self.decoder= decoder_type(hidden_dim, self.train_data.input_vocab.num_words, self.train_data.max_length, self.dropout)
        else:
            self.decoder = decoder_type(hidden_dim , self.train_data.target_vocab.num_words, self.train_data.max_length, self.dropout)
      
    def load_model(self):
        self.encoder.load_state_dict(torch.load(self.save_encoder_path))
        self.decoder.load_state_dict(torch.load(self.save_decoder_path))


    def predict(self, input_tensor):

        if(self.target_language=="english"):
            vocab = self.train_data.input_vocab
        else:
            vocab = self.train_data.target_vocab

        input_words = self.train_data.idx_tensor_to_words(input_tensor, self.train_data.input_vocab)

        with torch.no_grad(): # we dont want to change the gradients in this stage
            if(self.arch == "LSTM"):
                encoder_state = (self.encoder.init_hidden(), self.encoder.init_hidden())
            else:
                encoder_state = self.encoder.init_hidden()
        # we have both the hidden state and the cell state in the encoder, thats why there are 2 of these

            input_length = input_tensor.shape[0]
            

            encoder_outputs = torch.zeros(self.train_data.max_length, self.hidden_dim) # this is max length by length of hidden dim

        
            for i in range(input_length):
                encoder_output, encoder_state = self.encoder(input_tensor[i], encoder_state)
                encoder_outputs[i] = encoder_output

            decoder_input = torch.tensor([SOS_TOKEN])
            decoder_state = encoder_state

            user_teacher_forcing = True

            loss = 0


            output = []

            context_weights = torch.zeros(self.train_data.max_length, self.train_data.max_length)
            
            for j in range(self.train_data.max_length):
                if(self.arch == "GRU"):
                    decoder_output, decoder_state, weights = self.decoder(decoder_input, decoder_state, encoder_outputs)
                    context_weights[i] = weights

                else: 
                    decoder_output, decoder_state = self.decoder(decoder_input, decoder_state)

                output_idx = decoder_output.argmax(dim=1)
                decoder_input = output_idx # the highest probability tensor is the next input 

                output.append(output_idx.item()) #appending the index to the output 

                
                if (output_idx.item()== EOS_TOKEN):
                    break # if it predicts the end of string token then break


            output_sentence = self.train_data.indicies_to_sentence(output, vocab)

            return output_sentence, context_weights
           



 

    def predict_sentence(self):
        self.load_model()
        self.encoder.eval()
        self.decoder.eval()

        input_sentence = input("Input sentence to predict: ")
        input_words = input_sentence.split(" ") + [EOS_TOKEN]

        input_indicies = self.train_data.sentenceToIndicies(input_sentence, self.train_data.input_vocab) # turns the input sentence into indicies
        input_tensor = torch.tensor(input_indicies, dtype =torch.long)
        
        output_sentence, context_weights = self.predict(input_tensor)
        output_words = output_sentence.split(" ")


        print("Prediction: " + output_sentence)

        if(self.show_context_weights):
            fig = plt.figure()
            ax = fig.add_subplot(111) # this means 1x1x1
            cax = ax.matshow(context_weights.numpy(), cmap = 'bone')
            fig.colorbar(cax)
            ax.set_xticklabels([''] + input_words)
            ax.set_yticklabels([''] + output_words)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
            plt.show()        

    def closest_words(self, is_input=True, k=5): # to run in the terminal for french, say is_input=False in the parenthesis
        word = input("Please specify a word: ")
        vocab = self.train_data.input_vocab
        model = self.encoder

        if(not is_input):
            vocab = self.train_data.target_vocab
            model = self.decoder
        try:
            word_idx = vocab.word_to_index[word]
        except:
            print(f"The word {word} does not exist in the vocabulary")

        word_idx_tensor = torch.tensor([word_idx])
        word_embedding = model.embedding(word_idx_tensor)
        vocab_embeddings = model.embedding.weight
        word_embedding = word_embedding.permute(1,0) # permuting it to 256 x 1 so we can do the dot product
        scores = vocab_embeddings @ word_embedding
        print(scores.shape)
        top_scores, top_score_indicies = torch.topk(scores[:,0], k) # topk gives the top 5 scores

        top_scores = [elem.item() for elem in list(top_scores)]
        top_score_indicies = [elem.item() for elem in list(top_score_indicies)]
        top_words = [vocab.index_to_word[idx] for idx in top_score_indicies]

        print(f"Closest words to '{word}': ")
        for i in range(len(top_words)):
            print(f"Word: {top_words[i]}, Score: {top_scores[i]}")
  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_encoder_path", type=str, required=True)
    parser.add_argument("--save_decoder_path", type =str, required=True)
    parser.add_argument("--show_context_weights", action="store_true") # if this is true shows the diagram
    parser.add_argument("--arch", type=str, required=True)
    parser.add_argument("--hidden_dim", type=int, required=True)
    parser.add_argument("--dropout", action="store_true")
    parser.add_argument("--train_dataset_path", type=str, required =True)
    parser.add_argument("--source_language", type=str, required=True)
    parser.add_argument("--target_language", type=str, required=True)


    args = parser.parse_args()
    ti = TranslatorInference(args.save_encoder_path, args.save_decoder_path, args.show_context_weights, args.arch, args.hidden_dim, args.dropout, args.train_dataset_path,
                    args.source_language, args.target_language)
    #ti.predict_sentence()




