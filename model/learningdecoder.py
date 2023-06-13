# Assumption
# --------------------------------------------------------------------------
# Authors do not specify if the attention mechanism of Agent Learning Decoder 
# is multi-head or not, so we assume it is not multi-head.
# For Representation Encoder and Agent Matching Decoder, they mention the 
# equations are implemented with multi-head mechanism.
# We implement the attention as if it is multi-head and setting num_heads=1
# will make it a single head.

import math
import torch
import torch.nn as nn

# This attention mechanism is not the same with original self-attention.
# Here we implement eqn.3 and eqn.4, i.e. 
# S = softmax(QK^T / sqrt(d_k) + M)
# There will be another algorithm, Optimal Transport (eqn.5-6) before
# we scale the attention with Value.
class Attention_Eqn3(nn.Module):
    def __init__(self, hidden_dims):
        super().__init__()
        
        self.d_k = hidden_dims
        self.W_a_Q = nn.Linear(hidden_dims, hidden_dims)
        self.W_s_K = nn.Linear(hidden_dims, hidden_dims)
    
    
    def forward(self, F_a, F_s, M):
        
        # F_a has shape (batchsize, num_tokens, c=d_k)
        # F_s has shape (batchsize, h*w, c=d_k)
        
        Q_a = self.W_a_Q(F_a)  # Get Query, shape (batchsize, numtokens, c=d_k)
        K_s = self.W_s_K(F_s)  # Get Key, shape (batchsize, hw, c=d_k)
        
        # Transposed Key has shape (batchsize, c=d_k, hw)
        # Such that the result QK has shape (batchsize, numtokens, hw)
        # This corresponds to K x hw dimensions of M in eqn.4, see page 8 first sentence
        QK = torch.matmul(Q_a, K_s.transpose(1,2)) / math.sqrt(self.d_k)
        S = torch.nn.Softmax(QK + M)
        
        return S

class AgentLearningDecoder(nn.Module):
    
    def __init__(self,  c, num_layers, num_heads=1):
        super().__init__()
        
        self.d_k = c // num_heads
        self.attn = Attention_Eqn3(self.d_k)
    
    
    def forward(self, F_a, F_s, M_s):
        
        # Step 1: Masked Cross Attention between (F_a, F_s_hat)
        # This part is the implementation of eqn.3 and eqn.4
        # Return Part Mask S (see Fig.3 (a))
        # --------------------------------------------------------
        
        # Flatten M_s  from shape (batchsize, 1, h, w)
        # to shape (batchsize, 1, h*w)
        M_s = torch.flatten(M_s, start_dim=2)
        
        # See page 8, first sentence, N is the duplication of M
        # for each token of the agent tokens.
        # N has shape (batchsize, numtokens, hw)
        num_tokens = F_a.shape[1]
        N = M_s.repeat(1,num_tokens,1) 
                
        M = torch.where(N == 1, 0, float('-inf'))
        # Debug: Check M has zeros in it
        # print ((M == 0).nonzero(as_tuple=True)[0])
        
        # Get the "masked attention weight matrix"
        S = self.attn(F_a, F_s, M)
        
        
        # Step 2: TODO
        # This part is the implementation of eqn.5 and eqn.6
        # --------------------------------------------------------
        # Step 3: TODO
        # --------------------------------------------------------
        # Step 4: TODO
        # --------------------------------------------------------
