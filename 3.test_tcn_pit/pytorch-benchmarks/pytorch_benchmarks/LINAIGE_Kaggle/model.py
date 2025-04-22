import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
# from pytorch_model_summary import summary

from torchviz import make_dot
from contextlib import redirect_stdout
from plinio.methods import PIT
from plinio.methods import PITSuperNet
from plinio.methods.pit.nn import PITConv2d, PITLinear
from plinio.methods.pit_supernet.nn import PITSuperNetCombiner
from plinio.methods.pit_supernet import PITSuperNetModule



# ***********************************************This is specific to TNC model with new PLINIO file with window = 3 ***********************************************

# -------------------------------------------CNN+TCN+FC1-----------------------------------------------------
def build_CNN_TCN_FC1_pc(channel1:int, channel2:int, classification:bool, win_size:int, class_number:int):
    '''
    Build CNN architecture 3 model with 2 dense layers
    '''
    print("CNN3_TCN_FC1!")
    
    tcn_out = 32
    
    class CNN3_TCN_FC1_pc(nn.Module):
        def __init__(self, channel1, channel2, classification, win_size):
            
            #--------------------------------------------- feature extractor----------------------------------------
            super(CNN3_TCN_FC1_pc, self).__init__()
            # Pass (window,8,8) input to get output (channel,6,6) with kernel (3,3) -> 8-3+1=6
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=int(channel1), kernel_size=3, bias=False)
            self.bn1 = nn.BatchNorm2d(num_features=int(channel1))
            # If we use MaxPooling2d layer with kernel size of 2 (stride = kernel), the output shape would be half of the input shape->(channel,3,3)
            self.pool = nn.MaxPool2d(kernel_size=2)
            # Pass (channel1,3,3) input to get output (channel2,1,1) with kernel (3,3) -> 3-3+1=1
            self.conv2 = nn.Conv2d(in_channels=channel1, out_channels=int(channel2), kernel_size=3, bias=False)
            self.bn2 = nn.BatchNorm2d(num_features=int(channel2))
            self.sigmoid = nn.Sigmoid()
            
            #--------------------------------------------- tcn and fc layers----------------------------------------
            self.tcn = nn.Conv1d(in_channels=channel2, out_channels=tcn_out, kernel_size=3)
            self.bn3 = nn.BatchNorm1d(num_features=int(tcn_out))

            self.classification = classification
            self.fc1 = nn.Linear(tcn_out, class_number if classification else 1)

        
        def forward(self, x1, x2, x3):
            feature_list = []
            
            x = F.relu(self.bn1(self.conv1(x1)))
            x = self.pool(x)
            x = F.relu(self.bn2(self.conv2(x)))
            out_i = torch.flatten(x, 1)
            out_to_cat = out_i.unsqueeze(-1)
            
            feature_list.append(out_to_cat)
            
            x = F.relu(self.bn1(self.conv1(x2)))
            x = self.pool(x)
            x = F.relu(self.bn2(self.conv2(x)))
            out_i = torch.flatten(x, 1)
            out_to_cat = out_i.unsqueeze(-1)
            
            feature_list.append(out_to_cat)
            
            x = F.relu(self.bn1(self.conv1(x3)))
            x = self.pool(x)
            x = F.relu(self.bn2(self.conv2(x)))
            out_i = torch.flatten(x, 1)
            out_to_cat = out_i.unsqueeze(-1)
            
            feature_list.append(out_to_cat)
            
            feature_out = torch.cat(feature_list, dim=-1)
            
            x = self.bn3(self.tcn(feature_out))
            x = F.relu(x)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            
            
            if self.classification:
                x = x
            else:
                  x = self.sigmoid(x)
            return x

    model = CNN3_TCN_FC1_pc(channel1, channel2, classification, win_size)
    
    y = model(torch.zeros((win_size, 1, 8, 8)),torch.zeros((win_size, 1, 8, 8)),torch.zeros((win_size, 1, 8, 8)))

    # save the model summary
    with open('./cnn3_tcn_fc1' + str(channel1) + '_c'+ str(channel2) +'_w'+ str(win_size) +'.txt', 'w') as fi:
        with redirect_stdout(fi):
            summary(model, [(1, 8, 8), (1, 8, 8), (1, 8, 8)])
            
    print(summary(model, [(1, 8, 8), (1, 8, 8), (1, 8, 8)]))

    # plot the model architecture
    make_dot(y, params=dict(model.named_parameters())).render('cnn3_tcn_fc1_c_' + str(channel1) + '_'+ str(channel2) +'_w_'+ str(win_size), format='png')

    return model

# -------------------------------------------Model Building-----------------------------------------------------
def get_reference_model(model_name:str, channel1:int, channel2:int, classification:bool, win_size:int, class_number:int):
    '''
    Support windowing
    Select the model to run
    :param model_name: 'cnn3_tcn'
    :return:
    '''

    assert (model_name in ['c_p_c_tcn_fc'])


    if model_name == 'c_p_c_tcn_fc':
        model = build_CNN_TCN_FC1_pc(channel1, channel2, classification, win_size, class_number)

    return model