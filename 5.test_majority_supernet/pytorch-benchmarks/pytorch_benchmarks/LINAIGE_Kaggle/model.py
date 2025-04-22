import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary
from pytorch_model_summary import summary
from torchviz import make_dot
from contextlib import redirect_stdout
from plinio.methods import PIT
from plinio.methods import PITSuperNet
from plinio.methods.pit.nn import PITConv2d, PITLinear
from plinio.methods.pit_supernet.nn import PITSuperNetCombiner
from plinio.methods.pit_supernet import PITSuperNetModule

# from torchvision import models
class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, padding=1):
        super().__init__()
        self.conv1 = PITSuperNetModule([
            nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=1, bias=False),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU()),
            nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, 5, padding=2, stride=1, bias=False),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU()),
            nn.Sequential(
                torch.nn.Conv2d(in_channels, in_channels, 3, groups=in_channels, padding=1, stride=1, bias=False),
                torch.nn.BatchNorm2d(in_channels),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels, out_channels, 1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU()
            )], gumbel_softmax = True, hard_softmax = True)

    def forward(self, input):
        x = self.conv1(input)
        return x

# -------------------------------------------SUPERNET FC1-----------------------------------------------------
def build_SUPERNET_FC1_pc(channel1:int, channel2:int, classification:bool, win_size:int, class_number:int):
    '''
    Build CNN architecture 3 model with 1 dense layer
    '''
    print("supernet_FC1!")

    class SUPERNET_FC1_pc(nn.Module):
        def __init__(self, channel1, channel2, classification, win_size):
            
            super(SUPERNET_FC1_pc, self).__init__()
            # Pass (window,8,8) input to get output (channel,6,6) with kernel (3,3) -> 8-3+1=6
            self.block1 = ConvBlock(in_channels=win_size, out_channels=int(channel1))
            # If we use MaxPooling2d layer with kernel size of 2 (stride = kernel), the output shape would be half of the input shape->(channel,3,3)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            # Pass (channel1,3,3) input to get output (channel2,1,1) with kernel (3,3) -> 3-3+1=1
            self.block2 = ConvBlock(in_channels=channel1, out_channels=int(channel2))
            self.sigmoid = nn.Sigmoid()
            '''
            # Applying linear layer
            # First, converting torch.Size([1, 16, 4, 4]) to the vector with size 'channel2*4*4' as input size 
            # Second, determining the final output shape in two cases -> 
            # 1. classification: from 0 to the max people 3 in the dataset using softmax
            # 2. binary classification (regression): 0 or 1
            '''
            self.classification = classification
            self.fc = nn.Linear(channel2*4*4, class_number if classification else 1)
        
        ''' x.view()
        # Since x is the output of a convolutional layer, 
        # it has a 4-dimensional shape of (batch_size, channels, height, width). 
        # In order to feed this output to a fully connected (dense) layer, 
        # we first need to flatten the tensor to have a 2-dimensional shape of (batch_size, channels*height*width). 
        # This is achieved by calling x.view(x.size(0), -1), 
        # which reshapes the tensor so that the first dimension corresponds to the batch size 
        # and the second dimension is inferred by concatenating the remaining dimensions of the tensor.
        '''        
        def forward(self, x):
            x = self.block1(x)
            x = self.pool(x)
            x = self.block2(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            if self.classification:
                # x = F.softmax(x, dim=1)
                x = x
            else:
                  x = self.sigmoid(x)
            return x

    model = SUPERNET_FC1_pc(channel1, channel2, classification, win_size)
    
    x = torch.zeros((1, win_size, 8, 8))  # example input
    y = model(x)
    
    # save the model summary
    with open('./supernet_FC1' + '_c' + str(channel1) + '_c'+ str(channel2) +'_w'+ str(win_size) +'.txt', 'w') as fi:
        with redirect_stdout(fi):
            print(summary(model, x, show_input=False, show_hierarchical=True))
            
    print(print(summary(model, x, show_input=False, show_hierarchical=True)))

    # plot the model architecture
    make_dot(y, params=dict(model.named_parameters())).render('supernet_FC1' + '_c' + str(channel1) + '_c'+ str(channel2) +'_w'+ str(win_size), format='png')

    return model

# -------------------------------------------SUPERNET FC2-----------------------------------------------------
def build_SUPERNET_FC2_pc(channel1:int, channel2:int, classification:bool, win_size:int, class_number:int):
    '''
    Build CNN architecture 3 model with 1 dense layer
    '''
    print("supernet_FC2!")

    class SUPERNET_FC1_pc(nn.Module):
        def __init__(self, channel1, channel2, classification, win_size):
            
            super(SUPERNET_FC1_pc, self).__init__()
            # Pass (window,8,8) input to get output (channel,6,6) with kernel (3,3) -> 8-3+1=6
            self.block1 = ConvBlock(in_channels=win_size, out_channels=int(channel1))
            # If we use MaxPooling2d layer with kernel size of 2 (stride = kernel), the output shape would be half of the input shape->(channel,3,3)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            # Pass (channel1,3,3) input to get output (channel2,1,1) with kernel (3,3) -> 3-3+1=1
            self.block2 = ConvBlock(in_channels=channel1, out_channels=int(channel2))
            self.sigmoid = nn.Sigmoid()
            '''
            # Applying linear layer
            # First, converting torch.Size([1, 16, 4, 4]) to the vector with size 'channel2*4*4' as input size 
            # Second, determining the final output shape in two cases -> 
            # 1. classification: from 0 to the max people 3 in the dataset using softmax
            # 2. binary classification (regression): 0 or 1
            '''
            self.classification = classification
            self.fc1 = nn.Linear(channel2*4*4, 64)
            self.fc2 = nn.Linear(64, class_number if classification else 1)
        
        ''' x.view()
        # Since x is the output of a convolutional layer, 
        # it has a 4-dimensional shape of (batch_size, channels, height, width). 
        # In order to feed this output to a fully connected (dense) layer, 
        # we first need to flatten the tensor to have a 2-dimensional shape of (batch_size, channels*height*width). 
        # This is achieved by calling x.view(x.size(0), -1), 
        # which reshapes the tensor so that the first dimension corresponds to the batch size 
        # and the second dimension is inferred by concatenating the remaining dimensions of the tensor.
        '''        
        def forward(self, x):
            x = self.block1(x)
            x = self.pool(x)
            x = self.block2(x)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            if self.classification:
                # x = F.softmax(x, dim=1)
                x = x
            else:
                  x = self.sigmoid(x)
            return x

    model = SUPERNET_FC1_pc(channel1, channel2, classification, win_size)
    
    x = torch.zeros((1, win_size, 8, 8))  # example input
    y = model(x)
    
    # save the model summary
    with open('./supernet_FC2' + '_c' + str(channel1) + '_c'+ str(channel2) +'_w'+ str(win_size) +'.txt', 'w') as fi:
        with redirect_stdout(fi):
            print(summary(model, x, show_input=False, show_hierarchical=True))
            
    print(print(summary(model, x, show_input=False, show_hierarchical=True)))

    # plot the model architecture
    make_dot(y, params=dict(model.named_parameters())).render('supernet_FC2' + '_c' + str(channel1) + '_c'+ str(channel2) +'_w'+ str(win_size), format='png')

    return model

# -------------------------------------------CNN1-----------------------------------------------------
def build_CNN1_FC1_pc(channel:int, classification:bool, win_size:int, class_number:int):
    '''
    Build CNN architecture 1 model with 1 dense layer
    '''
    print("cnn1_FC1!")

    class CNN1_FC1_pc(nn.Module):
        def __init__(self, channel, classification, win_size):
            
            super(CNN1_FC1_pc, self).__init__()
            # pass (window,8,8) input to get output (channel,6,6) with kernel (3,3)-> 8-3+1
            self.conv = nn.Conv2d(in_channels=win_size, out_channels=int(channel), kernel_size=3, bias=False)
            self.bn = nn.BatchNorm2d(num_features=int(channel))
            self.sigmoid = nn.Sigmoid()
            '''
            # Applying linear layer
            # First, converting (channel,6,6) to the vector with size 'channel*6*6' as input size
            # Second, determining the output shape in two cases -> 
            # 1. classification: from 0 to the max people 3 in the dataset using softmax
            # 2. binary classification (regression): 0 or 1
            '''
            self.classification = classification
            self.fc = nn.Linear(channel*6*6, class_number if classification else 1)
        
        ''' x.view()
        # Since x is the output of a convolutional layer, 
        # it has a 4-dimensional shape of (batch_size, channels, height, width). 
        # In order to feed this output to a fully connected (dense) layer, 
        # we first need to flatten the tensor to have a 2-dimensional shape of (batch_size, channels*height*width). 
        # This is achieved by calling x.view(x.size(0), -1), 
        # which reshapes the tensor so that the first dimension corresponds to the batch size 
        # and the second dimension is inferred by concatenating the remaining dimensions of the tensor.
        '''    
        # def forward(self, x):
        #     x = self.bn(self.conv(x))
        #     x = F.relu(x)
        #     x = x.view(x.size(0), -1)
        #     x = self.fc(x)
        #     if self.classification:
        #         x = F.softmax(x, dim=1)
        #     return x
        
        def forward(self, x):
            x = self.bn(self.conv(x))
            x = F.relu(x)
            # x = x.view(x.size(0), -1)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            if self.classification:
            #   x = F.softmax(x, dim=1)
              x = x
            else:
              x = self.sigmoid(x)
            return x
        
    model = CNN1_FC1_pc(channel, classification, win_size)
    

    x = torch.zeros((1, win_size, 8, 8))  # example input
    y = model(x)
    # save the model summary
    with open('./cnn1_FC1' + str(channel) +'_w'+ str(win_size) +'.txt', 'w') as fi:
        with redirect_stdout(fi):
            summary(model,input_size=(win_size, 8, 8))
    
    print(summary(model,input_size=(win_size, 8, 8)))

    # plot the model architecture
    make_dot(y, params=dict(model.named_parameters())).render('./cnn1_FC1' + str(channel) +'_w'+ str(win_size) +'.png', format='png')

    
    return model

# -------------------------------------------CNN1+FC2-----------------------------------------------------
def build_CNN1_FC2_pc(channel:int, classification:bool, win_size:int, class_number:int):
    '''
    Build CNN architecture 1 model with 2 dense layers
    '''
    print("cnn1_FC2!")

    class CNN1_FC2_pc(nn.Module):
        def __init__(self, channel, classification, win_size):
            
            super(CNN1_FC2_pc, self).__init__()
            # pass (window,8,8) input to get output (channel,6,6) with kernel (3,3)-> 8-3+1
            self.conv = nn.Conv2d(in_channels=win_size, out_channels=int(channel), kernel_size=3, bias=False)
            self.bn = nn.BatchNorm2d(num_features=int(channel))
            self.sigmoid = nn.Sigmoid()
            '''
            # Applying linear layer
            # First, converting (channel,6,6) to the vector with size 'channel*6*6' and then the output of 64 as input size for fc2
            # Second, determining the final output shape in two cases -> 
            # 1. classification: from 0 to the max people 3 in the dataset using softmax
            # 2. binary classification (regression): 0 or 1
            '''
            self.classification = classification
            self.fc1 = nn.Linear(channel*6*6, 64)
            self.fc2 = nn.Linear(64, class_number if classification else 1)
        
        ''' x.view()
        # Since x is the output of a convolutional layer, 
        # it has a 4-dimensional shape of (batch_size, channels, height, width). 
        # In order to feed this output to a fully connected (dense) layer, 
        # we first need to flatten the tensor to have a 2-dimensional shape of (batch_size, channels*height*width). 
        # This is achieved by calling x.view(x.size(0), -1), 
        # which reshapes the tensor so that the first dimension corresponds to the batch size 
        # and the second dimension is inferred by concatenating the remaining dimensions of the tensor.
        '''    
        def forward(self, x):
            x = self.bn(self.conv(x))
            x = F.relu(x)
            # x = x.view(x.size(0), -1)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            if self.classification:
                # x = F.softmax(x, dim=1)
                x = x
            else:
                  x = self.sigmoid(x)
            return x

    model = CNN1_FC2_pc(channel, classification, win_size)
    
    
    x = torch.zeros((1, win_size, 8, 8))  # example input
    y = model(x)

    # save the model summary
    with open('./cnn1_FC2' + str(channel) +'_w'+ str(win_size) +'.txt', 'w') as fi:
        with redirect_stdout(fi):
            summary(model,input_size=(win_size, 8, 8))

    print(summary(model,input_size=(win_size, 8, 8)))
    
    # plot the model architecture
    make_dot(y, params=dict(model.named_parameters())).render('cnn1_FC2' + str(channel) +'_w'+ str(win_size) +'.png', format='png')

    return model

# -------------------------------------------CNN2-----------------------------------------------------
def build_CNN2_FC1_pc(channel:int, classification:bool, win_size:int, class_number:int):
    '''
    Build CNN architecture 2 model with 1 dense layer
    '''
    print("cnn2_FC1!")

    class CNN2_FC1_pc(nn.Module):
        def __init__(self, channel, classification, win_size):
            
            super(CNN2_FC1_pc, self).__init__()
            # pass (window,8,8) input to get output (channel,6,6) with kernel (3,3) -> 8-3+1
            self.conv = nn.Conv2d(in_channels=win_size, out_channels=int(channel), kernel_size=3, bias=False)
            self.bn = nn.BatchNorm2d(num_features=int(channel))
            # If we use MaxPooling2d layer with kernel size of 2 (stride = kernel), the output shape would be half of the input shape->(channel,3,3)
            self.pool = nn.MaxPool2d(kernel_size=2)
            self.sigmoid = nn.Sigmoid()
            '''
            # Applying linear layer
            # First, converting (channel,6,6) to the vector with size 'channel*6*6' as input size 
            # Second, determining the final output shape in two cases -> 
            # 1. classification: from 0 to the max people 3 in the dataset using softmax
            # 2. binary classification (regression): 0 or 1
            '''
            self.classification = classification
            self.fc = nn.Linear(channel*3*3, class_number if classification else 1)
        
        ''' x.view()
        # Since x is the output of a convolutional layer, 
        # it has a 4-dimensional shape of (batch_size, channels, height, width). 
        # In order to feed this output to a fully connected (dense) layer, 
        # we first need to flatten the tensor to have a 2-dimensional shape of (batch_size, channels*height*width). 
        # This is achieved by calling x.view(x.size(0), -1), 
        # which reshapes the tensor so that the first dimension corresponds to the batch size 
        # and the second dimension is inferred by concatenating the remaining dimensions of the tensor.
        '''        
        def forward(self, x):
            x = self.bn(self.conv(x))
            x = F.relu(x)
            x = self.pool(x)
            # x = x.view(x.size(0), -1)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            if self.classification:
                # x = F.softmax(x, dim=1)
                x = x
            else:
                x = self.sigmoid(x)
            return x

    model = CNN2_FC1_pc(channel, classification, win_size)
    
    x = torch.zeros((1, win_size, 8, 8))  # example input
    y = model(x)
    
    # save the model summary
    with open('./cnn2_FC1' + str(channel) +'_w'+ str(win_size) +'.txt', 'w') as fi:
        with redirect_stdout(fi):
            summary(model,input_size=(win_size, 8, 8))

    print(summary(model,input_size=(win_size, 8, 8)))
    
    # plot the model architecture
    make_dot(y, params=dict(model.named_parameters())).render('cnn2_FC1' + str(channel) +'_w'+ str(win_size) +'.png', format='png')

    return model

# -------------------------------------------CNN2+FC2-----------------------------------------------------
def build_CNN2_FC2_pc(channel:int, classification:bool, win_size:int, class_number:int):
    '''
    Build CNN architecture 2 model with 2 dense layers
    '''
    print("cnn2_FC2!")

    class CNN2_FC2_pc(nn.Module):
        def __init__(self, channel, classification, win_size):
            
            super(CNN2_FC2_pc, self).__init__()
            # pass (window,8,8) input to get output (channel,6,6) with kernel (3,3) -> 8-3+1
            self.conv = nn.Conv2d(in_channels=win_size, out_channels=int(channel), kernel_size=3, bias=False)
            self.bn = nn.BatchNorm2d(num_features=int(channel))
            # If we use MaxPooling2d layer with kernel size of 2 (stride = kernel), the output shape would be half of the input shape->(channel,3,3)
            self.pool = nn.MaxPool2d(kernel_size=2)
            self.sigmoid = nn.Sigmoid()
            '''
            # Applying linear layer
            # First, converting (channel,6,6) to the vector with size 'channel*6*6' as input size 
            # Second, determining the final output shape in two cases -> 
            # 1. classification: from 0 to the max people 3 in the dataset using softmax
            # 2. binary classification (regression): 0 or 1
            '''
            self.classification = classification
            self.fc1 = nn.Linear(channel*3*3, 64)
            self.fc2 = nn.Linear(64, class_number if classification else 1)
        
        ''' x.view()
        # Since x is the output of a convolutional layer, 
        # it has a 4-dimensional shape of (batch_size, channels, height, width). 
        # In order to feed this output to a fully connected (dense) layer, 
        # we first need to flatten the tensor to have a 2-dimensional shape of (batch_size, channels*height*width). 
        # This is achieved by calling x.view(x.size(0), -1), 
        # which reshapes the tensor so that the first dimension corresponds to the batch size 
        # and the second dimension is inferred by concatenating the remaining dimensions of the tensor.
        '''        
        def forward(self, x):
            x = self.bn(self.conv(x))
            x = F.relu(x)
            x = self.pool(x)
            # x = x.view(x.size(0), -1)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            if self.classification:
                # x = F.softmax(x, dim=1)
                x = x
            else:
                x = self.sigmoid(x)
            return x

    model = CNN2_FC2_pc(channel, classification, win_size)
    
    x = torch.zeros((1, win_size, 8, 8))  # example input
    y = model(x)
    
    # save the model summary
    with open('./cnn2_FC2' + str(channel) +'_w'+ str(win_size) +'.txt', 'w') as fi:
        with redirect_stdout(fi):
            summary(model,input_size=(win_size, 8, 8))
    
    print(summary(model,input_size=(win_size, 8, 8)))

    # plot the model architecture
    make_dot(y, params=dict(model.named_parameters())).render('cnn2_FC2' + str(channel) +'_w'+ str(win_size) +'.png', format='png')

    return model

# -------------------------------------------CNN3-----------------------------------------------------
def build_CNN3_FC1_pc(channel1:int, channel2:int, classification:bool, win_size:int, class_number:int):
    '''
    Build CNN architecture 3 model with 1 dense layer
    '''
    print("cnn3_FC1!")

    class CNN3_FC1_pc(nn.Module):
        def __init__(self, channel1, channel2, classification, win_size):
            
            super(CNN3_FC1_pc, self).__init__()
            # Pass (window,8,8) input to get output (channel,6,6) with kernel (3,3) -> 8-3+1=6
            self.conv1 = nn.Conv2d(in_channels=win_size, out_channels=int(channel1), kernel_size=3, bias=False)
            self.bn1 = nn.BatchNorm2d(num_features=int(channel1))
            # If we use MaxPooling2d layer with kernel size of 2 (stride = kernel), the output shape would be half of the input shape->(channel,3,3)
            self.pool = nn.MaxPool2d(kernel_size=2)
            # Pass (channel1,3,3) input to get output (channel2,1,1) with kernel (3,3) -> 3-3+1=1
            self.conv2 = nn.Conv2d(in_channels=channel1, out_channels=int(channel2), kernel_size=3, bias=False)
            self.bn2 = nn.BatchNorm2d(num_features=int(channel2))
            self.sigmoid = nn.Sigmoid()
            '''
            # Applying linear layer
            # First, converting (channel2,1,1) to the vector with size 'channel2*1*1' as input size 
            # Second, determining the final output shape in two cases -> 
            # 1. classification: from 0 to the max people 3 in the dataset using softmax
            # 2. binary classification (regression): 0 or 1
            '''
            self.classification = classification
            self.fc = nn.Linear(channel2*1*1, class_number if classification else 1)
        
        ''' x.view()
        # Since x is the output of a convolutional layer, 
        # it has a 4-dimensional shape of (batch_size, channels, height, width). 
        # In order to feed this output to a fully connected (dense) layer, 
        # we first need to flatten the tensor to have a 2-dimensional shape of (batch_size, channels*height*width). 
        # This is achieved by calling x.view(x.size(0), -1), 
        # which reshapes the tensor so that the first dimension corresponds to the batch size 
        # and the second dimension is inferred by concatenating the remaining dimensions of the tensor.
        '''        
        def forward(self, x):
            x = self.bn1(self.conv1(x))
            x = F.relu(x)
            x = self.pool(x)
            x = self.bn2(self.conv2(x))
            x = F.relu(x)
            # x = x.view(x.size(0), -1)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            if self.classification:
                # x = F.softmax(x, dim=1)
                x = x
            else:
                  x = self.sigmoid(x)
            return x

    model = CNN3_FC1_pc(channel1, channel2, classification, win_size)
    
    x = torch.zeros((2, win_size, 8, 8))  # example input
    y = model(x)
    
    # save the model summary
    with open('./cnn3_FC1' + str(channel1) + '_c'+ str(channel2) +'_w'+ str(win_size) +'.txt', 'w') as fi:
        with redirect_stdout(fi):
            summary(model,input_size=(win_size, 8, 8))
            
    print(summary(model,input_size=(win_size, 8, 8)))

    # plot the model architecture
    make_dot(y, params=dict(model.named_parameters())).render('cnn3_FC1' + str(channel1) + '_c'+ str(channel2) +'_w'+ str(win_size) +'.png', format='png')

    return model

# -------------------------------------------CNN3+FC2-----------------------------------------------------
def build_CNN3_FC2_pc(channel1:int, channel2:int, classification:bool, win_size:int, class_number:int): 
    '''
    Build CNN architecture 3 model with 2 dense layers
    '''
    print("cnn3_FC2!")

    class CNN3_FC2_pc(nn.Module):
        def __init__(self, channel1, channel2, classification, win_size):
            
            super(CNN3_FC2_pc, self).__init__()
            # Pass (window,8,8) input to get output (channel,6,6) with kernel (3,3) -> 8-3+1=6
            self.conv1 = nn.Conv2d(in_channels=win_size, out_channels=int(channel1), kernel_size=3, bias=False)
            self.bn1 = nn.BatchNorm2d(num_features=int(channel1))
            # If we use MaxPooling2d layer with kernel size of 2 (stride = kernel), the output shape would be half of the input shape->(channel,3,3)
            self.pool = nn.MaxPool2d(kernel_size=2)
            # Pass (channel1,3,3) input to get output (channel2,1,1) with kernel (3,3) -> 3-3+1=1
            self.conv2 = nn.Conv2d(in_channels=channel1, out_channels=int(channel2), kernel_size=3, bias=False)
            self.bn2 = nn.BatchNorm2d(num_features=int(channel2))
            self.sigmoid = nn.Sigmoid()
            '''
            # Applying linear layer
            # First, converting (channel2,1,1) to the vector with size 'channel2*1*1' then to 64 as input size 
            # Second, determining the final output shape in two cases -> 
            # 1. classification: from 0 to the max people 3 in the dataset using softmax
            # 2. binary classification (regression): 0 or 1
            '''
            self.classification = classification
            self.fc1 = nn.Linear(channel2*1*1, 64)
            self.fc2 = nn.Linear(64, class_number if classification else 1)
        
        ''' x.view()
        # Since x is the output of a convolutional layer, 
        # it has a 4-dimensional shape of (batch_size, channels, height, width). 
        # In order to feed this output to a fully connected (dense) layer, 
        # we first need to flatten the tensor to have a 2-dimensional shape of (batch_size, channels*height*width). 
        # This is achieved by calling x.view(x.size(0), -1), 
        # which reshapes the tensor so that the first dimension corresponds to the batch size 
        # and the second dimension is inferred by concatenating the remaining dimensions of the tensor.
        '''        
        def forward(self, x):
            x = self.bn1(self.conv1(x))
            x = F.relu(x)
            x = self.pool(x)
            x = self.bn2(self.conv2(x))
            x = F.relu(x)
            # x = x.view(x.size(0), -1)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            if self.classification:
                # x = F.softmax(x, dim=1)
                x = x
            else:
                  x = self.sigmoid(x)
            return x

    model = CNN3_FC2_pc(channel1, channel2, classification, win_size)
    
    x = torch.zeros((2, win_size, 8, 8))  # example input
    y = model(x)
    
    # save the model summary
    with open('./cnn3_FC2' + str(channel1) + '_c'+ str(channel2) +'_w'+ str(win_size) +'.txt', 'w') as fi:
        with redirect_stdout(fi):
            summary(model,input_size=(win_size, 8, 8))
            
    print(summary(model,input_size=(win_size, 8, 8)))

    # plot the model architecture
    make_dot(y, params=dict(model.named_parameters())).render('cnn3_FC2' + str(channel1) + '_c'+ str(channel2) +'_w'+ str(win_size) +'.png', format='png')

    return model

# -------------------------------------------CNN+TCN+FC2-----------------------------------------------------
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
  
        def forward(self, x):
            #--------------------------------------------- feature list----------------------------------------
            # reuse the same layers to extract features on each frame in input window
            feature_list = []
            for i in range(win_size):
                # to get any channel with shape [batch_size, channel_name, 8, 8] and convert it to [batch size, 1, 8, 8]
                x_i = x[:, i, :, :].unsqueeze(1)
                x_i = self.bn1(self.conv1(x_i))
                x_i = F.relu(x_i)
                x_i = self.pool(x_i)
                x_i = self.bn2(self.conv2(x_i))
                x_i = F.relu(x_i)
                
                out_i = torch.flatten(x_i, 1)
                
                # to convet [batch_size, 16] to [batch_size, 16, 1]
                out_to_hstack = out_i.unsqueeze(-1)

                feature_list.append(out_to_hstack)
            
            # to convert [batch_size, 1, 16] to [batch_size, 16, window_size]
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
    
    x = torch.zeros((win_size, win_size, 8, 8))  # example input
    y = model(x)
    
    # save the model summary
    with open('./cnn3_tcn_fc1' + str(channel1) + '_c'+ str(channel2) +'_w'+ str(win_size) +'.txt', 'w') as fi:
        with redirect_stdout(fi):
            summary(model,input_size=(win_size, 8, 8))
            
    print(summary(model,input_size=(win_size, 8, 8)))

    # plot the model architecture
    make_dot(y, params=dict(model.named_parameters())).render('cnn3_tcn_fc1_c_' + str(channel1) + '_'+ str(channel2) +'_w_'+ str(win_size), format='png')

    return model

# -------------------------------------------Model Building-----------------------------------------------------
def get_reference_model(model_name:str, channel1:int, channel2:int, classification:bool, win_size:int, class_number:int):
    '''
    Support windowing
    Select the model to run
    :param model_name: 'cnn1', 'cnn1_dense', 'cnn2', 'cnn2_dense', 'cnn3', 'cnn3_dense','cnn3_tcn'
    :return:
    '''

    assert (model_name in ['c_fc', 'c_fc_fc', 'c_p_fc', 'c_p_fc_fc', 'c_p_c_fc', 'c_p_c_fc_fc', 'c_p_c_fc_fc', 'c_p_c_fc_supernet', 'c_p_c_fc_fc_supernet'])

    # assert (classification), "the problem should be classification here!"

    if model_name == 'c_fc':
        model = build_CNN1_FC1_pc(channel1, classification, win_size, class_number)
    elif model_name == 'c_fc_fc':
        model = build_CNN1_FC2_pc(channel1, classification, win_size, class_number)
    elif model_name == 'c_p_fc':
        model = build_CNN2_FC1_pc(channel1, classification, win_size, class_number)
    elif model_name == 'c_p_fc_fc':
        model = build_CNN2_FC2_pc(channel1, classification, win_size, class_number)
    elif model_name == 'c_p_c_fc':
        model = build_CNN3_FC1_pc(channel1, channel2, classification, win_size, class_number)
    elif model_name == 'c_p_c_fc_fc':
        model = build_CNN3_FC2_pc(channel1, channel2, classification, win_size, class_number)
    elif model_name == 'c_p_c_fc_supernet':
         model = build_SUPERNET_FC1_pc(channel1, channel2, classification, win_size, class_number)
    elif model_name == 'c_p_c_fc_fc_supernet':
        model = build_SUPERNET_FC2_pc(channel1, channel2, classification, win_size, class_number)

    return model