from data import *
from model import *
from utils import *
from torch.optim.lr_scheduler import StepLR


class PretrainTranslator:
    def __init__(self, train_dataset_path, val_dataset_path):
        self.train_data = TranslationDataset(train_dataset_path,  "english", "french")
        self.hidden_dim = 128
        self.encoder_eng = EncoderGRUWeighted(self.train_data.input_vocab.num_words, self.hidden_dim)
        self.encoder_fr = EncoderGRUWeighted(self.train_data.target_vocab.num_words, self.hidden_dim)
        self.decoder_eng = DecoderGRUWeighted(self.hidden_dim, self.train_data.input_vocab.num_words, self.train_data.max_length)
        self.decoder_fr = DecoderGRUWeighted(self.hidden_dim, self.train_data.target_vocab.num_words, self.train_data.max_length)
        self.mse_loss_func = nn.MSELoss() # mean squared loss function
        self.classification_loss_func = nn.NLLLoss()
        self.learning_rate = 0.001 # 0.01
        self.num_iters = 50_000

        # self.encoder_lr_scheduler = StepLR(self.encoder_eng_optimizer, 25_000, gamma = 0.1) # this changes the learning rate by 0.1 every 100_000 iterations
        

        self.encoder_eng_optimizer = torch.optim.SGD(self.encoder_eng.parameters(), lr=self.learning_rate)
        self.encoder_fr_optimizer = torch.optim.SGD(self.encoder_fr.parameters(), lr = self.learning_rate)
        self.decoder_eng_optimizer = torch.optim.SGD(self.decoder_eng.parameters(), lr=self.learning_rate)
        self.decoder_fr_optimizer = torch.optim.SGD(self.decoder_fr.parameters(), lr=self.learning_rate)


    def models_zero_grad(self):
        self.encoder_eng_optimizer.zero_grad()
        self.encoder_fr_optimizer.zero_grad()
        self.encoder_eng_optimizer.zero_grad()
        self.decoder_fr_optimizer.zero_grad()

    def save_model(self):
        torch.save(self.encoder_eng.state_dict(), "pretrain_encoder_eng.pth")
        torch.save(self.encoder_fr.state_dict(), "pretrain_encoder_fr.pth")
        torch.save(self.decoder_eng.state_dict(), "pretrain_encoder_eng.pth")
        torch.save(self.decoder_fr.state_dict(), "pretrain_decoder_fr.pth")

    def models_step(self):
        self.encoder_eng_optimizer.step()
        self.encoder_fr_optimizer.step()
        self.decoder_eng_optimizer.step()
        self.decoder_fr_optimizer.step()

        # english, encode it into the hidden state
        # french, encode the same sentence into hidden state
        #compare the hidden states
        # we are looking to match two tensors and make their numbers the same because the tensors encode contxt/meaning so we will match them to make sure  the numbers are the same
        # this type of loss function does that 


    def train_step(self, lang1_idx_tensor, lang2_idx_tensor):
        encoder_eng_state = self.encoder_eng.init_hidden()
        encoder_fr_state = self.encoder_fr.init_hidden()

        input_eng_length = lang1_idx_tensor.shape[0]
        input_fr_length = lang2_idx_tensor.shape[0]
        encoder_eng_outputs = torch.zeros(self.train_data.max_length, self.hidden_dim)
        encoder_fr_outputs = torch.zeros(self.train_data.max_length, self.hidden_dim)

        for i in range(input_eng_length):
            encoder_eng_output, encoder_eng_state = self.encoder_eng(lang1_idx_tensor[i], encoder_eng_state)
            encoder_eng_outputs[i] = encoder_eng_output
        
        for i in range(input_fr_length):
            encoder_fr_output, encoder_fr_state = self.encoder_fr(lang2_idx_tensor[i], encoder_fr_state)
            encoder_fr_outputs[i] = encoder_fr_output

        # we are preforming a loss_func(encoder_eng_state, encoder_fr_state)
        # we need the decoder to say from each of these states make sure we can translate back to the sentences
        # we can do a cross hatch situation, eng->fr, fr->eng and make sure meaning in the middle is the same
        # or we can do eng->eng, fr->fr, and then ensure it that way

        context_loss = self.mse_loss_func(encoder_eng_state, encoder_fr_state)
        decoder_eng_input = torch.tensor([SOS_TOKEN])
        decoder_eng_state = encoder_eng_state
        decoder_fr_state = encoder_fr_state 
        eng_translation_loss = 0 

        for i in range(input_eng_length):
            decoder_eng_output, decoder_eng_state, eng_context_weights = self.decoder_eng(decoder_eng_input, decoder_eng_state, encoder_eng_outputs)
            eng_translation_loss += self.classification_loss_func(decoder_eng_output, lang1_idx_tensor[i][None])
            decoder_eng_input = lang1_idx_tensor[i][None]

        fr_translation_loss = 0 
        decoder_fr_input = torch.tensor([SOS_TOKEN])


        for i in range(input_fr_length):
            decoder_fr_output, decoder_fr_state, fr_context_weights = self.decoder_fr(decoder_fr_input, decoder_fr_state, encoder_fr_outputs)
            fr_translation_loss += self.classification_loss_func(decoder_fr_output, lang2_idx_tensor[i][None])
            decoder_fr_input = lang2_idx_tensor[i][None]
            #print()

        loss = 0.6 * 20 * context_loss + 0.2 * eng_translation_loss + 0.2 * fr_translation_loss
        self.models_zero_grad()
        loss.backward()
        self.models_step()

        return loss, context_loss, eng_translation_loss, fr_translation_loss

    def train(self):
        for i in range(self.num_iters):
            lang1_idx_tensor, lang2_idx_tensor = self.train_data.get_random_sample()
            loss, context_loss, eng_translation_loss, fr_translation_loss = self.train_step(lang1_idx_tensor, lang2_idx_tensor)
            if(i%100 == 0):
                print(f"Iter [{i}]:, Loss = {loss}, Context loss = {context_loss}, Eng Loss = {eng_translation_loss}, Fr Loss = {fr_translation_loss}")


if __name__ == "__main__":
    pretrain_translator = PretrainTranslator("data/eng_fr_train.txt", "data/eng_fr_val.txt")
    try:
        pretrain_translator.train()
    except:
        pass
    finally:
        pretrain_translator.save_model()
        print("Model has been saved.")








        

