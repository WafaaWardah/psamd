# This file contains various neural network classes for PyTorch

import torch
import torch.nn as nn
import torchaudio


#################################################################################
# new nisqa-dim model
#################################################################################

class NiSQA_Dim(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        bundle = torchaudio.pipelines.WAV2VEC2_XLSR53
        self.feature_extractor = bundle.get_model().feature_extractor



    def forward(self, waveform):

        with torch.inference_mode():
            features, _ = self.feature_extractor.extract_features(waveform)
        
        
        y_pred = features

        print(f'length of y_pred: {len(y_pred)}')
        return y_pred





class TinyModel(nn.Module):

    def __init__(self, insize, outsize):
        # https://towardsdatascience.com/how-to-code-a-simple-neural-network-in-pytorch-for-absolute-beginners-8f5209c50fdd
        super(TinyModel, self).__init__()

        self.linear1 = torch.nn.Linear(insize, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, outsize)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x


class Linear1(nn.Module):

    def __init__(self, in_features, out_features):
        super(Linear1, self).__init__()
        self.model = nn.Sequential(nn.Linear(in_features, 10),
                                   nn.ReLU(),
                                   nn.Linear(10, out_features))
                                    # nn.Sigmoid())

    def forward(self, x):
        output = self.model(x)
        return output
    
