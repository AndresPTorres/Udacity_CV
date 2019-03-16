import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        ''' Initialize the layers of this model.'''
        super().__init__()
        
        self.hidden_size = hidden_size

        # embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)

        # the LSTM takes embedded word vectors (of a specified size) as inputs 
        # and outputs hidden states of size hidden_dim
        self.num_layers = num_layers
        
        # I never got this to work with batch_first = True
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

        # the linear layer that maps the hidden state output dimension 
        # to the number of tags we want as output, tagset_size (in this case this is 3 tags)
        self.hidden2tag = nn.Linear(hidden_size, vocab_size)

    def init_hidden(self, batch_size):
        ''' At the start of training, we need to initialize a hidden state;
           there will be none because the hidden state is formed based on perviously seen data.
           So, this function defines a hidden state with all zeroes and of a specified size.'''
        # The axes dimensions are (n_layers, batch_size, hidden_dim)
        return (torch.zeros((self.num_layers, batch_size, self.hidden_size), device=device),
                torch.zeros((self.num_layers, batch_size, self.hidden_size), device=device))
    
    def forward(self, features, captions):
        ''' Define the feedforward behavior of the model.'''
        
        batch_size = features.shape[0]
        
        # create embedded vectors for the image and each word in the caption
        captions = captions[:, :-1]
        embeds = self.word_embeddings(captions)
        
        #join_embeds_image_text = torch.cat((features.view(batch_size, 1, features.shape[1]), embeds), 1)
        features = features.unsqueeze(1)
        join_image_text = torch.cat((features, embeds), 1)
        
        self.hidden = self.init_hidden(batch_size)
        
        # get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hidden state
        lstm_out, _ = self.lstm(join_image_text, self.hidden)
        
        # get the scores for the most likely tag for a word
        #tag_outputs = self.hidden2tag(lstm_out.view(join_embeds_image_text.shape[1], -1))
        tag_outputs = self.hidden2tag(lstm_out)
        
        #tag_scores = F.log_softmax(tag_outputs, dim = 2)
        
        return tag_outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        #current_embeds = inputs.unsqueeze(1)
        current_embeds = inputs
        
        output = []
        batch_size = inputs.shape[0]
        
        if not states:
            hidden = self.init_hidden(batch_size)
        else:
            hidden = states
            
        for word_idx in range(0, max_len):
            lstm_out, hidden = self.lstm(current_embeds, hidden)
            tag_outputs = self.hidden2tag(lstm_out)
            tag_outputs = tag_outputs.squeeze(1)
            next_word = torch.argmax(tag_outputs, dim = 1)
            
            output.append(next_word.cpu().numpy()[0].item())
            
            if (next_word == 1):
                #<end> word, do not continue predicting
                break
            
            current_embeds = self.word_embeddings(next_word)
            current_embeds = current_embeds.unsqueeze(1)
            
        return output
            
            