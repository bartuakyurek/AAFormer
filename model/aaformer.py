"""

This file is created under the CENG 502 Project.


"""
import math
import numpy as np
import torch
import torch.nn as nn

from model.tokens import init_agent_tokens
from model.featureextractor import FeatureExtractor
from model.representationencoder import RepresentationEncoder
from model.agentlearningdecoder import AgentLearningDecoder
from model.agentmatchingdecoder import AgentMatchingDecoder


class AAFormer(nn.Module):
    # cuda: bool 
    # num_tokens: int, number of agent tokens
    def __init__(self, cuda, c, hw, N, heads, num_tokens, im_res, reduce_dim, bypass_ot=False, sinkhorn_reg=1e-1):
        super().__init__()

        # Some values to store
        self.bypass_ot = bypass_ot
        self.num_tokens = num_tokens 
        self.feat_res = int(math.sqrt(hw))
        self.output_res = im_res

        # Models of AAFormer 
        self.feature_extractor = FeatureExtractor(layers=50, reduce_dim=reduce_dim, c=c)
        self.representation_encoder = RepresentationEncoder(c, hw, N, heads)
        self.agent_learning_decoder = AgentLearningDecoder(cuda, c, N, num_tokens, sinkhorn_reg = sinkhorn_reg)
        self.agent_matching_decoder = AgentMatchingDecoder(heads, c, feat_res=self.feat_res)


        # Last layers before prediction
        self.reshapers = [nn.ConvTranspose2d(c, c, kernel_size=2, stride=2) for i in range(int(math.log2(im_res/self.feat_res)))]
        self.reshaper = nn.Sequential(*self.reshapers)

        self.conv3 = nn.Conv2d(c, c//8, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(c//8, 3, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self, query_img, supp_imgs, supp_masks):

        # STEP 1: Extract Features from the backbone model (ResNet)
        # -------------------------------------------------------------------------------------------------
        F_Q, F_S, s_mask_list, f_s, M_s = self.feature_extractor(query_img, supp_imgs, supp_masks)

        # STEP 2.1: Pass the features from encoder 
        # -------------------------------------------------------------------------------------------------
        F_S_hat = self.representation_encoder(F_S)
        F_Q_hat = self.representation_encoder(F_Q)
        
        # STEP 2.2: Get Initial Agent Tokens
        # -------------------------------------------------------------------------------------------------
        # TODO: can we get rid of for loop?
        # Since every image will have different number of foreground pixels, it is not possible
        # to combine the foreground pixels of images in a single tensor. We may pad the tensors
        # since max number of foreground pixels is equal to the image area.
        X, L = [], []
        # TODO: This part assumes we are doing 1-shot learning
        for i, m in enumerate(M_s):  # M_s has shape (batchsize, 1, 16, 16)
            m = m.squeeze(0)  # Get a single mask, shape (16, 16)

            fg = np.where(m == 1.) # get foreground pixels
            bg = np.where(m == 0.) # get background pixels
            
            # Create tensor with shape [num_foreground_pix, 2] where the last dimension has
            # (x,y) locations of foreground pixels
            foreground_pix = torch.stack((torch.from_numpy(fg[0]), torch.from_numpy(fg[1])), dim=1)
            background_pix = torch.stack((torch.from_numpy(bg[0]), torch.from_numpy(bg[1])), dim=1)

            X.append(foreground_pix)
            L.append(background_pix)
        # every token has [K,c] dim for every sample in a batch
        agent_tokens = init_agent_tokens(self.num_tokens, X, L, f_s) 
        
        
        # STEP 3: Pass initial agent tokens through Agent Learning Decoder and obtain agent tokens.
        # -------------------------------------------------------------------------------------------------
        # Note: agent_tokens has shape (batchsize, num_tokens, c)
        agent_tokens = self.agent_learning_decoder(agent_tokens, F_S_hat, M_s, bypass_ot=self.bypass_ot)
        
        # STEP 4: Pass agent tokens through Agent Matching Decoder
        # -------------------------------------------------------------------------------------------------
        F_q_bar = self.agent_matching_decoder(agent_tokens, F_S_hat, F_Q_hat)
      
        # STEP 5: Reshape / Conv
        # -------------------------------------------------------------------------------------------------
        # Fig.2, reshape/conv arrow before the last prediction box:
        # Assumption: Paper doesn't mention how they reshape the output of the last decoder,
        # so we assume we can use transposed convolution to upsample the output.
        
        #print(F_q_bar.shape) ---> [batchsize, c, feat_res, feat_res]
        output = self.reshaper(F_q_bar)
        #print(output.shape) ---> [batchsize, c, im_res, im_res]
        
        output = self.conv3(output) #dec_feat_query)
        output = self.relu(output)
        output = self.conv1(output)

        return output