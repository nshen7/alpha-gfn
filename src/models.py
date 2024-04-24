import torch
import torch.nn as nn
import math

from config import *
from torch import Tensor

## nn utils
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('_pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        "x: ([batch_size, ]seq_len, embedding_dim)"
        seq_len = x.size(0) if x.dim() == 2 else x.size(1)
        return x + self._pe[:seq_len]  # type: ignore
    
class LSTMSharedNet(nn.Module):
    def __init__(
        self,
        # observation_space: gym.Space,
        n_layers: int,
        d_model: int,
        dropout: float,
        device: torch.device
    ):
        super().__init__()

        self._device = device
        self._d_model = d_model
        self._n_actions: float = SIZE_ACTION

        self._token_emb = nn.Embedding(self._n_actions+1, d_model, 0)   # Last one is [BEG]
        self._pos_enc = PositionalEncoding(d_model).to(device)

        self._lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )

    def forward(self, obs: Tensor) -> Tensor:
        '''
        Input: tensor of dimension (1 x seqlen), consisting of integer-indexed actions (padded to length of MAX_EXPR_LENGTH)
        '''
        bs, seqlen = obs.shape
        beg = torch.full((bs, 1), fill_value=self._n_actions, dtype=torch.long, device=obs.device)
        obs = torch.cat((beg, obs.long()), dim=1)
        real_len = (obs != 0).sum(1).max() 

        src = self._pos_enc(self._token_emb(obs))
        res = self._lstm(src[:,:real_len])[0]
        
        return res.mean(dim=1)
        

class TBModel(nn.Module):
    def __init__(self, num_hid_1, num_hid_2):
        
        super().__init__()
        self.device = DEVICE
        self.lstm = LSTMSharedNet( # feature extractor for token sequences       
            n_layers=2,
            d_model=num_hid_1,
            dropout=0.1,
            device=self.device
        ) 
        self.mlp = nn.Sequential( 
            nn.Linear(num_hid_1, num_hid_2),  
            nn.LeakyReLU(),
            nn.Linear(num_hid_2, 2*SIZE_ACTION),  # number of outputs: SIZE_ACTION for P_F and SIZE_ACTION for P_B.
        )
        self.logZ = nn.Parameter(torch.ones(1))  # log Z is just a single number.

    def forward(self, x):
        
        lstm_output = self.lstm(x)
        logits = self.mlp(lstm_output)
        # Slice the logits into forward and backward policies.
        P_F = logits[..., :SIZE_ACTION]
        P_B = logits[..., SIZE_ACTION:]

        return P_F, P_B