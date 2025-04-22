import torch
import pathlib
import argparse
from pytorch_model_summary import summary
# import pytorch_benchmarks.LINAIGE_Kaggle as lk
from  pytorch_benchmarks.LINAIGE_Kaggle import model as model_module
from  pytorch_benchmarks.LINAIGE_Kaggle import  data as data_module
from  pytorch_benchmarks.LINAIGE_Kaggle import  train as train_module
from pytorch_benchmarks.utils import seed_all, EarlyStopping, CheckPoint
from plinio.methods import PIT, PITSuperNet
from plinio.methods.pit_supernet.nn import PITSuperNetCombiner
from pathlib import Path
import numpy as np
import pandas as pd
import os.path
import copy

def main(args):
    
    CD_SIZE = torch.tensor(args.cd_size)
    LAMBDA = torch.tensor(args.strength)
    DATA_DIR = args.data_dir
    N_EPOCHS = args.epochs 
    MODEL_NAME = args.model_name
    
    print(f'cd_size is {args.cd_size}')
    print(f'strength is {args.strength}')
    
    # Check CUDA availability
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print("Training on:", device)

    # Ensure deterministic execution
    seed_all(seed=42)

    # Listing the parameters
    win_size_list = [1,3,5,8]
    confindence_list = ['easy', 'all']
    remove_frame_list = [True, False]
    classification_list = [True, False]
    channel1_list = [8, 16, 32, 64]
    channel2_list = [8, 16, 32, 64]  # for cnn3
    model_name_list = ['c_fc', 'c_fc_fc', 'c_p_fc', 'c_p_fc_fc', 'c_p_c_fc', 'c_p_c_fc_fc', 'c_p_c_fc_supernet', 'c_p_c_fc_fc_supernet']

    win_size = win_size_list[1]
    confindence = confindence_list[0]
    remove_frame = remove_frame_list[0]
    classification = classification_list[0]
    data_dir = None
    # model_name=model_name_list[-2]
    model_name = MODEL_NAME 
    channel1 = channel1_list[0]
    channel2 = channel2_list[1]

    parameters = f'''******************Parameters******************: 
    win_size={win_size} 
    confindence = {confindence} 
    remove_frame = {remove_frame} 
    classification = {classification}
    data_dir = {data_dir}
    model_name = {model_name}
    channel1 = {channel1}
    channel2 = {channel2}
    '''
    print(parameters)
    
    print('******************Performing warm_up and NAS using base_model and selected session******************')
    # Calling dataloader to creat batched datasets
    ds_train, ds_test, x_train, class_weight, class_number = data_module.get_data( win_size=win_size,
                                                                                    confindence=confindence,
                                                                                    remove_frame=remove_frame,
                                                                                    classification=classification,
                                                                                    data_dir=None,
                                                                                    session_number=1,
                                                                                    test_split=None)
                                
    # Getting cnn model
    model_base = model_module.get_reference_model(  model_name=model_name,
                                                    channel1=channel1,
                                                    channel2=channel2,
                                                    classification=classification,
                                                    win_size=win_size,
                                                    class_number=class_number)

    print('Input shape:', x_train.shape)
    
    ######################### warm up loop #########################
    
    # Passing y_train to get_default_criterion(classification, class_weight=None) funtion to get 'crossEntropy' bassed on class_weights
    criterion = train_module.get_default_criterion(classification=classification, class_weight=class_weight)
    # criterion = train_module.get_default_criterion(classification=classification)
    
    # Setting the optimizer
    optimizer = train_module.get_default_optimizer(model_base)
    
    # Setting learning_rate and early_stop
    earlystop = EarlyStopping(patience=10, mode='min') # changed EarlyStopping(patience=20, mode='max') ????
    reduce_lr = train_module.get_default_scheduler(optimizer)
    

    search = False
    for epoch in range(N_EPOCHS):
        metrics = train_module.train_one_epoch(epoch, model_base, criterion, optimizer, ds_train, device, classification, class_number, search, cd_size=None, cd_ops=None, reg_strength=None)
        # Checking if there is not any imporvement in metrics to reduce the learning rate, the patient is 5 epochs
        reduce_lr.step(metrics['loss']) # changed to scheduler.step()???
        # Checking if there is not any imporvement in metrics, the patient is 10 epochs
        if earlystop(metrics['loss']): # eliminated ???
            break
    
    print('******************Performing nas_model_supernet******************')
    ######################### Convert the model to SUPERNET #########################
    input_example = torch.unsqueeze(torch.from_numpy(x_train[0]), 0).to(device)
    input_shape = x_train[0].shape
    model_to_nas = copy.deepcopy(model_base)
    nas_model_supernet = PITSuperNet(model_to_nas, input_shape=input_shape, autoconvert_layers = False)
    nas_model_supernet = nas_model_supernet.to(device)
    
    # Passing y_train to get_default_criterion(classification, class_weight=None) funtion to get 'crossEntropy' bassed on class_weights
    criterion = train_module.get_default_criterion(classification=classification, class_weight=class_weight)
    
    param_dicts = [
        {'params': nas_model_supernet.nas_parameters(), 'weight_decay': 0},
        {'params': nas_model_supernet.net_parameters()}]
    
    optimizer = torch.optim.Adam(param_dicts, lr=0.001, weight_decay=1e-4)
    
    # Setting learning_rate and early_stop
    earlystop = EarlyStopping(patience=10, mode='min') # changed EarlyStopping(patience=20, mode='max') ????
    reduce_lr = train_module.get_default_scheduler(optimizer)
    
    cd_ops = 0
    print("Initial model size:", nas_model_supernet.get_size())
    print("Initial model MACs:", nas_model_supernet.get_macs())
    for module in nas_model_supernet.modules(): 
        if isinstance(module, PITSuperNetCombiner):
            print(torch.nn.functional.softmax(module.alpha, dim=0))

    search = True
    for epoch in range(N_EPOCHS):
        metrics = train_module.train_one_epoch(epoch, nas_model_supernet, criterion, optimizer, ds_train, device, classification, class_number, search, cd_size=CD_SIZE, cd_ops=cd_ops, reg_strength=None)
        
        reduce_lr.step(metrics['loss']) # changed to scheduler.step()???
        # Checking if there is not any imporvement in metrics, the patient is 10 epochs
        if earlystop(metrics['loss']): # eliminated ???
            break
        print("model size:", nas_model_supernet.get_size())
        print("model MACs:", nas_model_supernet.get_macs())
        for module in nas_model_supernet.modules(): 
            if isinstance(module, PITSuperNetCombiner):
                print(torch.nn.functional.softmax(module.alpha, dim=0))
                    
    # Convert Supernet model into pytorch model
    exported_model_supernet = nas_model_supernet.arch_export()
    exported_model_supernet = exported_model_supernet.to(device)
    print('******************Final SuperNet Architecture Summary******************')
    print(summary(exported_model_supernet, input_example, show_input=False, show_hierarchical=True))


    if ds_test != 0:
        '''Note that we set session as test_set from the last one to the first one except session one'''
        test_metrics = train_module.evaluate(exported_model_supernet, criterion, ds_test, device, classification, class_number)
        
        print("Test Set Loss:", test_metrics['loss'])
        print("Test Set BAS:", test_metrics['BAS'])
        print("Test Set ACC:", test_metrics['ACC'])
        print("Test Set ROC:", test_metrics['ROC'])
        print("Test Set F1:", test_metrics['F1'])
        print("Test Set MSE:", test_metrics['MSE'])
        print("Test Set MAE:", test_metrics['MAE'])

    print('******************Performing nas_pit_model******************')
    ######################### Convert the model to PIT #########################
    input_example = torch.unsqueeze(torch.from_numpy(x_train[0]), 0).to(device)
    input_shape = x_train[0].shape
    model_to_nas_supernet = copy.deepcopy(exported_model_supernet)
    pit_model = PIT(model_to_nas_supernet, input_shape=input_shape)
    pit_model = pit_model.to(device)
    pit_model.train_features = True
    pit_model.train_rf = False
    pit_model.train_dilation = False
    print(summary(pit_model, input_example, show_input=False, show_hierarchical=True))
    
    # Passing y_train to get_default_criterion(classification, class_weight=None) funtion to get 'crossEntropy' bassed on class_weights
    criterion = train_module.get_default_criterion(classification=classification, class_weight=class_weight)
    # criterion = train_module.get_default_criterion(classification=classification)
    
    # Setting the optimizer
    param_dicts = [
        {'params': pit_model.nas_parameters(), 'weight_decay': 0},
        {'params': pit_model.net_parameters()}]
    
    optimizer = torch.optim.Adam(param_dicts, lr=0.001, weight_decay=1e-4)
    
    # Setting learning_rate and early_stop
    earlystop = EarlyStopping(patience=10, mode='min') # changed EarlyStopping(patience=20, mode='max') ????
    reduce_lr = train_module.get_default_scheduler(optimizer)
    
    search_checkpoint = CheckPoint('./search_checkpoints', pit_model, optimizer, 'max')
    
    search = True
    for epoch in range(N_EPOCHS):
        metrics = metrics = train_module.train_one_epoch(epoch, pit_model, criterion, optimizer, ds_train, device, classification, class_number, search, cd_size=None, cd_ops=None, reg_strength=LAMBDA)
        # Checking if there is not any imporvement in metrics to reduce the learning rate, the patient is 5 epochs
        if epoch > 5:
            search_checkpoint(epoch, metrics['ACC'])
            if earlystop(metrics['loss']): # changed to earlystop(metrics['val_acc'])???
                break
        # Checking if there is not any imporvement in metrics to reduce the learning rate, the patient is 5 epochs
        reduce_lr.step(metrics['loss']) # changed to scheduler.step()???

        print("architectural summary:")
        print(pit_model)
        print("model size:", pit_model.get_size())
    print("architectural summary:")
    print(pit_model)
    print("model size:", pit_model.get_size())
    
    print("******************Load best model******************")
    search_checkpoint.load_best()
    print("******************Final PIT Architecture Summary******************")
    print(summary(pit_model.arch_export(), input_example, show_input=False, show_hierarchical=True))
    
    
    if ds_test != 0:
        '''Note that we set session as test_set from the last one to the first one except session one'''
        test_metrics = train_module.evaluate(pit_model, criterion, ds_test, device, classification, class_number)
        
        print("Test Set Loss:", test_metrics['loss'])
        print("Test Set BAS:", test_metrics['BAS'])
        print("Test Set ACC:", test_metrics['ACC'])
        print("Test Set ROC:", test_metrics['ROC'])
        print("Test Set F1:", test_metrics['F1'])
        print("Test Set MSE:", test_metrics['MSE'])
        print("Test Set MAE:", test_metrics['MAE'])        
               

    # Getting the linaige dataset genrator using cross validation
    print('******************Data Preparation for Fine_Tunning******************')
    ds_linaige_cv, class_number = data_module.get_data( win_size=win_size,
                                                        confindence=confindence,
                                                        remove_frame=remove_frame,
                                                        classification=classification,
                                                        data_dir=None,
                                                        session_number=None,
                                                        test_split=None)

    Loss_list = []
    BAS_list = []
    ACC_list = []
    ROC_list = []
    F1_list = []
    MSE_list = []
    MAE_list  = []


    # Exporting generated datasets from generator due to cross validation
    print('******************Starting Main Loop******************')
    for dataset in ds_linaige_cv:
        
        ######################### Convert pit model into pytorch model #########################
        exported_model = pit_model.arch_export()
        exported_model = exported_model.to(device)
        
        # exported_model = nas_model_supernet.arch_export()
        
        reset_code_flag = False
        if reset_code_flag:
            for layer in exported_model.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
                    print('Resetttttttttttttttttttt')
                    
        print(summary(exported_model, input_example, show_input=False, show_hierarchical=True))        
            
        # Extracting all datasets from generator
        _, _, _, _, class_weight = dataset
        # Passing y_train to get_default_criterion(classification, class_weight=None) funtion to get 'crossEntropy' bassed on class_weights
        criterion = train_module.get_default_criterion(classification=classification, class_weight=class_weight)
        # criterion = train_module.get_default_criterion(classification=classification)
        
        # Setting the optimizer
        optimizer = train_module.get_default_optimizer(exported_model)
        
        # Setting learning_rate and early_stop
        earlystop = EarlyStopping(patience=10, mode='min')
        reduce_lr = train_module.get_default_scheduler(optimizer)
        
        # Calling dataloader to creat batched datasets
        ds_train, ds_test = data_module.build_dataloaders(dataset)
        
        search = False
        for epoch in range(N_EPOCHS):
            metrics = train_module.train_one_epoch(epoch, exported_model, criterion, optimizer, ds_train, device, classification, class_number, search, cd_size=None, cd_ops=None, reg_strength=None)
            # Checking if there is not any imporvement in metrics to reduce the learning rate, the patient is 5 epochs
            reduce_lr.step(metrics['loss']) # changed to scheduler.step()???
            # Checking if there is not any imporvement in metrics, the patient is 10 epochs
            if earlystop(metrics['loss']): # eliminated???
                break  
        
        '''Note that we set session as test_set from the last one to the first one except session one'''
        test_metrics = train_module.evaluate(exported_model, criterion, ds_test, device, classification, class_number)
        
        Loss_list.append(test_metrics['loss'])
        BAS_list.append(test_metrics['BAS'])
        ACC_list.append(test_metrics['ACC'])
        ROC_list.append(test_metrics['ROC'])
        F1_list.append(test_metrics['F1'])
        MSE_list.append(test_metrics['MSE'])
        MAE_list.append(test_metrics['MAE'])
        
        print("Test Set Loss:", test_metrics['loss'])
        print("Test Set BAS:", test_metrics['BAS'])
        print("Test Set ACC:", test_metrics['ACC'])
        print("Test Set ROC:", test_metrics['ROC'])
        print("Test Set F1:", test_metrics['F1'])
        print("Test Set MSE:", test_metrics['MSE'])
        print("Test Set MAE:", test_metrics['MAE'])  
        


    print("Test Set Loss_list:", Loss_list, 'Mean is:', np.mean(Loss_list), 'std is:', np.std(Loss_list))
    print("Test Set BAS_list:", BAS_list, 'Mean is:', np.mean(BAS_list), 'std is:', np.std(BAS_list))
    print("Test Set ACC_list:", ACC_list, 'Mean is:', np.mean(ACC_list), 'std is:', np.std(ACC_list))
    print("Test Set ROC_list:", ROC_list, 'Mean is:', np.mean(ROC_list), 'std is:', np.std(ROC_list))
    print("Test Set F1_list:", F1_list, 'Mean is:', np.mean(F1_list), 'std is:', np.std(F1_list))
    print("Test Set MSE_list:", MSE_list, 'Mean is:', np.mean(MSE_list), 'std is:', np.std(MSE_list))
    print("Test Set MAE_list:", MAE_list, 'Mean is:', np.mean(MAE_list), 'std is:', np.std(MAE_list))
    
    # Convert the lists to strings
    Loss_str = ', '.join(str(val) for val in Loss_list)
    BAS_str = ', '.join(str(val) for val in BAS_list)
    ACC_str = ', '.join(str(val) for val in ACC_list)
    ROC_str = ', '.join(str(val) for val in ROC_list)
    F1_str = ', '.join(str(val) for val in F1_list)
    MSE_str = ', '.join(str(val) for val in MSE_list)
    MAE_str = ', '.join(str(val) for val in MAE_list)

    # Create a dictionary with the concatenated strings
    data = {
        'Model Name': [model_name],
        'win_size': [win_size],
        'confindence': [confindence],
        'remove_frame': [remove_frame],
        'classification': [classification],
        'channel1': [channel1],
        'channel2': [channel2],
        'cd_size': [args.cd_size],
        'strength': [args.strength],
        'Base Model Total Param': [sum(p.numel() for p in model_base.parameters())],
        'SUPERNET Model Total Param': [sum(p.numel() for p in exported_model_supernet.parameters())],
        'PIT Model Total Param': [sum(p.numel() for p in exported_model.parameters())],
        'Loss_mean': [np.mean(Loss_list)],
        'BAS_mean': [np.mean(BAS_list)],
        'ACC_mean': [np.mean(ACC_list)],
        'ROC_mean': [np.mean(ROC_list)],
        'F1_mean': [np.mean(F1_list)],
        'MSE_mean': [np.mean(MSE_list)],
        'MAE_mean': [np.mean(MAE_list)],
        'Loss_std': [np.std(Loss_list)],
        'BAS_std': [np.std(BAS_list)],
        'ACC_std': [np.std(ACC_list)],
        'ROC_std': [np.std(ROC_list)],
        'F1_std': [np.std(F1_list)],
        'MSE_std': [np.std(MSE_list)],
        'MAE_std': [np.std(MAE_list)],
        'Loss_list': [Loss_str],
        'BAS_list': [BAS_str],
        'ACC_list': [ACC_str],
        'ROC_list': [ROC_str],
        'F1_list': [F1_str],
        'MSE_list': [MSE_str],
        'MAE_list': [MAE_str]
    }


    # Create a DataFrame from the dictionary
    df_new = pd.DataFrame(data)

    # Check if the file exists
    file_path = f'model_{model_name}_pit_{channel1}_{channel2}_w_{win_size}_cd_size_{args.cd_size}_reset_code_{reset_code_flag}.xlsx'
    if os.path.isfile(file_path):
        # Read the existing Excel file if it exists
        df_existing = pd.read_excel(file_path)

        # Append the new data to the existing DataFrame
        df_updated = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        # If the file doesn't exist, use the new DataFrame as is
        df_updated = df_new

    # Save the updated DataFrame to the Excel file
    df_updated.to_excel(file_path, index=False)
    print("******************Excel file saved successfully!******************") 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NAS Search and Fine-Tuning')
    parser.add_argument('--cd_size', type=float, help='Regularization Parameter')
    parser.add_argument('--strength', type=float, help='Regularization Parameter')
    parser.add_argument('--epochs', type=int, help='Number of Training Epochs')
    parser.add_argument('--data-dir', type=str, default=None, help='Path to Directory with Training Data')
    parser.add_argument('--model_name', type=str, default=None, help='Model List:[c_fc, c_fc_fc, c_p_fc, c_p_fc_fc, c_p_c_fc, c_p_c_fc_fc, c_p_c_fc_supernet, c_p_c_fc_fc_supernet]')
    args = parser.parse_args()
    main(args)
    
    # python ./pytorch-benchmarks/linaige_main.py --cd_size 1e-7 --strength 1 --epochs 10 --data-dir './' --model_name 'c_p_c_fc_supernet'