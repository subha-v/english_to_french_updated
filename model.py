import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMCell(nn.Module):
    
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lin_f = nn.Linear(input_dim+hidden_dim, hidden_dim) # this outputs to what C_t-1 is
        self.lin_i = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.lin_c = nn.Linear(input_dim + hidden_dim,hidden_dim)
        self.lin_o = nn.Linear(input_dim + hidden_dim, hidden_dim)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self,x,state):
        h,c, = state # unpacking a tuple
        xh = torch.cat([x,h], dim=1) # concatenating x and h #this has shape 512
        f = self.sigmoid(self.lin_f(xh)) # input flows through the linear layers, and then passes through a sigmoid function
        # f has shape 256 because of the self.lin_f
        # check the definition of lin_f

        # f is the values we are forgetting
        # after this line we get a tensor of weights
        # Update path
        i = self.sigmoid(self.lin_i(xh))
        ct = self.tanh(self.lin_c(xh))
        ot = self.sigmoid(self.lin_o(xh)) # output, its the weights that say what are we multiplying our cell state by
        c = f * c + i * ct

        h = ot * self.tanh(c)  # this is the new tanh


        return h, (h,c)



        # previous cell state is c
        # modifying c in place

        # xh is our input with our previous hidden
        # we need to see what values we are forgetting


        c = f * c + i * ct # we use a * because it preforms the multiplication element wise



class EncoderLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.lstm = LSTMCell(hidden_dim, hidden_dim)


    def forward(self,x,state): # x refers to the input
        x = self.embedding(x).unsqueeze(0) #this makes x into a rank 2 tensor
        output, state = self.lstm(x, state)
        return output, state


    def init_hidden(self):
        return (torch.zeros(1,self.hidden_dim), torch.zeros(1, self.hidden_dim))


# the embedding layer is basicallyt he list of one+hot encoding


class DecoderLSTM(nn.Module):
    def __init__(self, hidden_dim, output_dim, max_length, is_dropout = False, dropout_prob = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim # we need to access the hidden dim
        self.embedding = nn.Embedding(output_dim, hidden_dim)
        self.lstm = LSTMCell(hidden_dim, hidden_dim) # we want it to output a hidden dim because 
        self.lin_out = nn.Linear(hidden_dim,output_dim)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.max_length = max_length
        self.is_dropout = is_dropout
        self.dropout = nn.Dropout(dropout_prob)


    def forward(self, x, state):
        out = self.embedding(x)
        if(self.is_dropout):
            out = self.dropout(out)
            
        out = F.relu(out)
        out, state = self.lstm(out, state)
        out = self.lin_out(out)
        out = self.log_softmax(out)
        return out, state

    def init_hidden(self):
        return (torch.zeros(1, self.hidden_dim), torch.zeros(1, self.hidden_dim))

        


class EncoderGRUWeighted(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(input_dim,hidden_dim)
        
        self.gru = nn.GRU(hidden_dim, hidden_dim) # Gated recurrent unit
        #self.output_layer = nn.Linear(self.hidden_dim, self.output_dim) this is not needed.>

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_dim) # we need 3 dimensions because GRU expects this to be a batch size of 1


    def forward(self, x, hidden):
    


        # the 1,1,01 makes sure we have a rank 3 tensor
        embed = self.embedding(x).view(1, 1,-1) # x is a tensor of indicies, it is the representation of words that is learned
        # what does the view do?
        

        # we specify that we want a 1,1 first and the -1 fills in the appropriate thing for the last dimension to fit the original number of elements
        out, hidden  = self.gru(embed, hidden)
        return out, hidden

class DecoderGRUWeighted(nn.Module):
    def __init__(self, hidden_dim, output_dim, max_length, is_dropout=False, dropout_prob=0.1): #10% of the time it drops out
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_length = max_length
        self.embedding = nn.Embedding(output_dim, hidden_dim) # we want to go through like we start with the SOS token but we continue to pass in
        # the previous output word or the correct thing into the function
        self.weights_layer = nn.Linear(self.hidden_dim * 2, self.max_length)
        self.combine_layer = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        #self.final_output_layer = nn.Linear(self.hidden_dim, self.output_dim)
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)
        
        self.is_dropout = is_dropout
        self.dropout = nn.Dropout(dropout_prob)

        #sometimes it zeros out some of the input data on random
        #makes the model have better generalization capabilities


        # the GRU is a simplified version of the LSTM cell, and its faster to train because there are less gates 

    def forward(self, x, hidden, encoder_outputs): 
        embed = self.embedding(x).view(1, 1 ,-1) # he calls it x
        if(self.is_dropout):
            embed = self.dropout(embed)



        # the last dim=1 tells us that the softmax function should be applied on dim=1
        # x is the index, embed is the embedding

        weights = F.softmax(self.weights_layer(torch.cat([embed[0], hidden[0]], dim=1)), dim = 1) # we want to concatinate across dim =1
        weighted_encoder_outputs = torch.bmm(weights[None],encoder_outputs[None]) # the [None] adds a dimension on the dimension 0
        # instead of 1x3, it does to 1x1x3
        # this performs the matrix multiplication
        # a batch of data is just multiple inputs at the same time
        # this stands for batch matrix multiplication, but this does rank 3 tensors with batches
        out = torch.cat([embed[0] , weighted_encoder_outputs[0]], dim =1)
        out = self.combine_layer(out)
        out = out[None] # adding back the third dimension
        out = F.relu(out)
        out, hidden = self.gru(out, hidden)
        out = self.output_layer(out[0])
        out = F.log_softmax(out, dim=1)
        return out, hidden, weights

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_dim)


