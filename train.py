from torch.nn.modules.rnn import GRU
from data import *
from model import *
import argparse
import json
import time
import matplotlib.pyplot as plt
DEBUG_MODE = 0 # global variable that is constant
import matplotlib.ticker as ticker
from torch.optim.lr_scheduler import StepLR



class EngFreTranslator:
    def __init__(self, train_dataset_path, val_dataset_path, hidden_dim, num_iters,
     save_encoder_path, save_decoder_path, is_pretrain, use_pretrained_encoder, arch, dropout, pretrain_encoder_path, save_pretrain_type):
        self.num_iters = num_iters
        self.iter_2_val_acc = {}
        self.iter_2_loss = {}
        #self.save_encoder_path = save_encoder_path
        #self.save_decoder_path = save_decoder_path 
        self.arch = arch
        self.pretrain_type = save_pretrain_type
        self.dropout = dropout

        encoder_type = EncoderLSTM if arch == "LSTM" else EncoderGRUWeighted
        decoder_type = DecoderLSTM if arch == "LSTM" else DecoderGRUWeighted


        self.train_data = TranslationDataset(train_dataset_path, "english", "french")
        self.is_pretrain = is_pretrain
        self.use_pretrained_encoder = use_pretrained_encoder

        self.hidden_dim = hidden_dim
        self.pretrain_encoder_path = pretrain_encoder_path

        self.val_data = TranslationDataset(val_dataset_path, "english", "french")
        #elf.data = TranslationDataset(dataset_path, "english", "french")
        self.encoder = encoder_type(self.train_data.input_vocab.num_words, hidden_dim)
        self.n_print = 1000
        
        # the decoder has an if statement because it depends whether we are pretraining
        # if we already pretrained, we want to use the input vocab, else we want to use the target_vocab
        # if we pretrain, english-> english


        if self.is_pretrain:
            self.decoder = decoder_type(hidden_dim, self.train_data.input_vocab.num_words, self.train_data.max_length, self.dropout) # this ist he output dim

        else:
            self.decoder = decoder_type(hidden_dim, self.train_data.target_vocab.num_words, self.train_data.max_length, self.dropout)

        self.loss_func = nn.NLLLoss()
        if(self.pretrain_type == "matching_contexts"):
            self.loss_func = nn.MSELoss()

        self.save_encoder_path = save_encoder_path
        self.save_decoder_path = save_decoder_path
        #self.save_model_path = save_model_path

        self.learning_rate = 0.01 # one of the most important hyperparameters, in general the faster the learning rate goes

        self.encoder_optimizer = torch.optim.SGD(self.encoder.parameters(), lr = self.learning_rate)
        self.decoder_optimizer = torch.optim.SGD(self.decoder.parameters(), lr= self.learning_rate)

        self.encoder_lr_scheduler = StepLR(self.encoder_optimizer, 50_000, gamma = 0.1) # this changes the learning rate by 0.1 every 100_000 iterations
        self.decoder_lr_scheduler = StepLR(self.decoder_optimizer, 50_000, gamma=0.1)

    def load_pretrained_encoder(self):
        self.encoder.load_state_dict(torch.load(self.pretrain_encoder_path))


    def load_model(self):
        self.encoder.load_state_dict(torch.load(self.save_encoder_path))
        self.decoder.load_state_dict(torch.load(self.save_decoder_path))

    def save_model(self):
        torch.save(self.encoder.state_dict(), self.save_encoder_path)
        torch.save(self.decoder.state_dict(), self.save_decoder_path)
    
    def save_data(self):
        with open("data.json", "w") as f:
            json.dump(self.iter_2_val_acc, f)

        with open("loss_data.json", "w") as f:
            json.dump(self.iter_2_loss, f)





    def train_step(self, lang1_idx_tensor, lang2_idx_tensor):
       
        encoder_state = self.encoder.init_hidden() # we refactored
       

        # we have both the hidden state and the cell state in the encoder, thats why there are 2 of these

        input_length = lang1_idx_tensor.shape[0]
        target_length = lang2_idx_tensor.shape[0]

        encoder_outputs = torch.zeros(self.train_data.max_length, self.hidden_dim) # this is max length by length of hidden dim

        
        for i in range(input_length):
            encoder_output, encoder_state = self.encoder(lang1_idx_tensor[i], encoder_state)
            encoder_outputs[i] = encoder_output

        decoder_input = torch.tensor([SOS_TOKEN])
        decoder_state = encoder_state

        user_teacher_forcing = True

        loss = 0

        if user_teacher_forcing:
            for j in range(target_length):
                if(self.arch == "GRU"):
                    decoder_output, decoder_state, weights = self.decoder(decoder_input, decoder_state, encoder_outputs)
                
                else:
                    decoder_output, decoder_state = self.decoder(decoder_input, decoder_state)
                
                #decoder_output, decoder_state, weights  = self.decoder(decoder_input, decoder_state, encoder_outputs)
                loss +=self.loss_func(decoder_output, lang2_idx_tensor[j].unsqueeze(0))
                decoder_input = lang2_idx_tensor[j][None]

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss.backward() # computes the gradients
        # loss is what we compute the gradeints on

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item()/target_length


    def train(self):

        if(self.use_pretrained_encoder):
            self.encoder.load_state_dict(torch.load(self.pretrain_encoder_path)) # torch.load loads from the path

            # load _state a-dict puts it in the encoder


            # freezes the gradeint computation

            for p in self.encoder.parameters():
                p.requires_grad = False

        print(self.num_iters)

        for i in range(self.num_iters):
            lang1_idx_tensor, lang2_idx_tensor = self.train_data.get_random_sample()
            

            if(self.is_pretrain):
                loss = self.train_step(lang1_idx_tensor, lang1_idx_tensor)
            else:
                loss = self.train_step(lang1_idx_tensor, lang2_idx_tensor)



            
            # if (i % 1000 == 0):
           

            if(i % self.n_print == 0):
                
                self.encoder.eval()
                self.decoder.eval()
                print(f"Iter[{i}]: Loss = {loss:.4f}")

                self.predict()
                accuracy = self.evaluate()
                self.iter_2_val_acc[i] = accuracy
                self.iter_2_loss[i] = loss
                

                
                self.encoder.train()
                self.decoder.train()

            self.encoder_lr_scheduler.step()
            self.decoder_lr_scheduler.step()



    def evaluate(self):
        with torch.no_grad():
            num_correct = 0
            total = 0 
            for i in range(50):
                
                lang1_idx_tensor, lang2_idx_tensor = self.val_data.get_random_sample()# the reason we can do this is because of getitem in data.py so we can check the iterables at those values
                # we could also do this for lists of pairs , but its more clunky. this is a clearner way of dealing with this
                
                if(self.is_pretrain):
                    lang2_idx_tensor = lang1_idx_tensor
                
                encoder_state = self.encoder.init_hidden()
                

                input_length = lang1_idx_tensor.shape[0]
                encoder_outputs = torch.zeros(self.train_data.max_length, self.hidden_dim)

                for i in range(input_length): # each i refers to an index in our sentence
                    #lang1_idx_tensor corresponds to a word in english, in our deep leaning model we give a tensor of indicies

                    encoder_output, encoder_state = self.encoder(lang1_idx_tensor[i], encoder_state)
                    encoder_outputs[i] = encoder_output #storing the encoder_output in encoderpoutpsuts

                decoder_input = torch.tensor([SOS_TOKEN])
                decoder_state = encoder_state

                output = []

                context_weights = torch.zeros(self.val_data.max_length, self.val_data.max_length)

                target_length = lang2_idx_tensor.shape[0] # this is the target length
                    # this is possible as we're predicting, our output sentence is either shorter or longer than our actual length
                    # assume we always start at the begining and go up until the target_length and if we have any discrepancy in length, we count that as something that is wrong

                for j in range(self.val_data.max_length):
                    if(self.arch == "GRU"):
                        decoder_output, decoder_state, weights = self.decoder(decoder_input, decoder_state, encoder_outputs)
                        context_weights[i] = weights
                    else:
                        decoder_output, decoder_state = self.decoder(decoder_input, decoder_state)


                    output_idx = decoder_output.argmax(dim=1)
                    decoder_input = output_idx

                    if(j < target_length):
                        if(output_idx.item() == lang2_idx_tensor[j].item()):
                            num_correct += 1 # incrementing the number correct


                    if(output_idx.item() == EOS_TOKEN):
                        break



                total += target_length

            accuracy = (num_correct/total) * 100
            print(f"Validation Accuracy: {accuracy}%, Num correct: {num_correct}, Total: {total}")
            return accuracy


    def predict(self, visualize_context_weights = False):
        self.encoder.eval()
        self.decoder.eval()

       

        lang1_idx_tensor, lang2_idx_tensor = self.val_data.get_random_sample()
        # we are getting a random 
        vocab = self.val_data.target_vocab

        if self.is_pretrain:
            lang2_idx_tensor = lang1_idx_tensor
            vocab = self.val_data.input_vocab

        input_words = self.val_data.idx_tensor_to_words(lang1_idx_tensor, self.val_data.input_vocab)
        output_words = self.val_data.idx_tensor_to_words(lang2_idx_tensor, vocab)



        with torch.no_grad(): # we dont want to change the gradients in this stage
            if(self.arch == "LSTM"):
                encoder_state = (self.encoder.init_hidden(), self.encoder.init_hidden())
            else:
                encoder_state = self.encoder.init_hidden()
        # we have both the hidden state and the cell state in the encoder, thats why there are 2 of these

            input_length = lang1_idx_tensor.shape[0]
            #target_length = lang2_idx_tensor.shape[0]

            encoder_outputs = torch.zeros(self.train_data.max_length, self.hidden_dim) # this is max length by length of hidden dim

        
            for i in range(input_length):
                encoder_output, encoder_state = self.encoder(lang1_idx_tensor[i], encoder_state)
                encoder_outputs[i] = encoder_output


            #for i in range(input_length):
             #   encoder_output, encoder_state = self.encoder(lang1_idx_tensor[i], encoder_state)

            decoder_input = torch.tensor([SOS_TOKEN])
            decoder_state = encoder_state

            user_teacher_forcing = True

            loss = 0


            output = []

            context_weights = torch.zeros(self.val_data.max_length, self.val_data.max_length)
            
            for j in range(self.val_data.max_length):
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


            output_sentence = self.val_data.indicies_to_sentence(output, vocab)


            correct_indicies = list(lang2_idx_tensor)
            correct_indicies = [x.item() for x in correct_indicies]


            output_sentence = self.val_data.indicies_to_sentence(output, vocab)
            print(f"Model Output: {output_sentence} ")

            print(f"Correct answer: {self.val_data.indicies_to_sentence(correct_indicies, vocab)}")

            # Create a variable context_weights

            if(visualize_context_weights):
                fig = plt.figure()
                ax = fig.add_subplot(111) # this means 1x1x1
                cax = ax.matshow(context_weights.numpy(), cmap = 'bone')
                fig.colorbar(cax)
                ax.set_xticklabels([''] + input_words)
                ax.set_yticklabels([''] + output_words)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
                ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
                plt.show()





if __name__ == '__main__':

    if DEBUG_MODE:
        hidden_dim = 256
        translator = EngFreTranslator("data/eng_fr_train.txt", "data/eng_fr_val.txt", hidden_dim, 1000000, 
        "encoder.pth", "decoder.pth", True, False, "GRU")
        translator.train()


    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--train", action = "store_true")
        parser.add_argument("--continue_training", action="store_true")
        parser.add_argument("--use_pretrained_encoder", action="store_true")
        parser.add_argument("--pretrain", action="store_true")
        parser.add_argument("--train_dataset_path", type=str, required=True)
        parser.add_argument("--val_dataset_path", type=str, required=True)
        parser.add_argument("--num_iters", type=int, default=100_000)
        parser.add_argument("--save_encoder_path", type=str, default = "encoder.pth")
        parser.add_argument("--save_decoder_path", type=str, default= "decoder.pth")
        parser.add_argument("--pretrain_encoder_path", type=str)
        parser.add_argument("--hidden_dim", type=int, default=256)
        parser.add_argument("--arch", type = str, default = "GRU")
        parser.add_argument("--dropout", action="store_true")
        parser.add_argument("--pretrain_type", type = str, default = "english_to_english") # matching contexts
        args = parser.parse_args()
        translator = EngFreTranslator(args.train_dataset_path, args.val_dataset_path, 
        args.hidden_dim, args.num_iters, args.save_encoder_path, 
        args.save_decoder_path, args.pretrain, args.use_pretrained_encoder, args.arch, args.dropout, args.pretrain_encoder_path, args.pretrain_type)



        if(args.train or args.pretrain):
            if(args.continue_training):
                translator.load_model()

            try:
                start = time.time()
                translator.train()
            except KeyboardInterrupt as e:
                pass
            finally:
                end = time.time()
                translator.save_model()
                print(f"Training Duration: {end-start} (s)")
                print("The model has been saved.")
        else:
            translator.load_model()
            translator.predict(True)



