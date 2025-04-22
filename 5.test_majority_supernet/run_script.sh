#!/bin/bash

num_base=8
num_power=7

# Loop through and run the command
for ((i=1; i<=$num_base; i++))
do
    for ((j=2; j<=$num_power; j++))
    do
        s=$(echo "$i*10^-$j" | bc -l)
        echo $s '******************************************is started******************************************'
        # sleep 6
        
        # nohup python pytorch-benchmarks/linaige_main.py --strength $s --epochs 500 --data-dir './' & 2>&1
        # wait
        # pid=$!  # Get the PID of the background process
        # echo "PID: $pid"
        # # ps -ef | grep $pid
        # # ps -e | grep $pid
        
        python ./pytorch-benchmarks/linaige_main.py --cd_size $s --epochs 500 --data-dir './' --model_name 'c_p_c_fc_fc_supernet'

        # rm ./final_best_warmup.ckp  
        # rm -rf ./warmup_checkpoints/ 
        # rm -rf ./search_checkpoints/ 
         
        echo $s '******************************************is done******************************************'
    done
done