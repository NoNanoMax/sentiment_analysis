import torch
import os
import numpy as np
import nltk
from nltk.stem import SnowballStemmer
import pymorphy2
nltk.download('punkt')
import torch
import torch.nn as nn
import json

def encode_t(token):
  if token in tok2idx.keys():
    return tok2idx[token]
  return 0 #UNKNOWN

MAX_LEN = 256

def preproc(s):
  words = nltk.word_tokenize(s, language="russian")
  morph = pymorphy2.MorphAnalyzer()
  new_words= [morph.parse(word)[0].normal_form for word in words if word.isalnum()]
  sent = words
  sent = ['[SOS]'] + sent
  sent = sent + ['[EOS]']
  idx = [encode_t(word) for word in sent]
  return torch.tensor(idx).reshape(1, -1)

class RNN(nn.Module):
  def __init__(self, vocab_size, emb_dim, hidden_dim, output_dim, n_layers, 
                bidirectional, dropout, pad_idx):
      
    super().__init__()
    
    self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx = pad_idx)
    
    self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, bidirectional=bidirectional,
                  dropout=dropout) 
    
    self.fc = nn.Linear(2*hidden_dim, output_dim)
    self.dropout = nn.Dropout(p=dropout)
        
        
  def forward(self, text, text_lengths=MAX_LEN):
      
    #text = [sent len, batch size]
    
    embedded = self.embedding(text).permute(1, 0, 2)
    
    #embedded = [sent len, batch size, emb dim]
    
    packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, [text_lengths])
    
    packed_output, (hidden, cell) = self.rnn(packed_embedded)
   
    output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)  
    
    hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

    return self.fc(hidden)



if __name__ == "__main__":
  f = open("a.ex", "r")
  sent = f.read()
  with open("tok2idx.json", "r") as my_file:
    tok2idx = my_file.read()
  tok2idx = json.loads(tok2idx)
  idx = preproc(sent)
  vocab_size = len(tok2idx)
  emb_dim = 100
  hidden_dim = 256
  output_dim = 3
  n_layers = 2
  bidirectional = True
  dropout = 0.45
  PAD_IDX = 3
  patience=3
  model = RNN(
    vocab_size=vocab_size,
    emb_dim=emb_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    n_layers=n_layers,
    bidirectional=bidirectional,
    dropout=dropout,
    pad_idx=PAD_IDX
  )
  ans = model(idx, len(idx))
  ans = torch.argmax(ans[0]).item()
  if ans == 0:
    print("negative")
  if ans == 1:
    print("positive")
  if ans == 2:
    print("neutral")
