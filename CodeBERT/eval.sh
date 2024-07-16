#!/bin/bash


lang=cpp #programming language
beam_size=10

batch_size=128
source_length=512
target_length=64

# train_file=../ksc/single_line_r_train.txt
# dev_file=../ksc/single_line_r_valid.txt

output_dir=model/$lang
pretrained_model=microsoft/codebert-base #Roberta: roberta-base

export CUDA_VISIBLE_DEVICES=1

test_file=("test")
test_model=("last")
for (( i=0; i<${#test_file[@]}; i++ ))
do
  for (( k=0; k<${#test_model[@]}; k++ ))
  do
    path=$output_dir/checkpoint-${test_model[$k]}
    dev_file=../ksc/single_line_r_${test_file[$i]}.txt
    python run.py --do_test --model_type roberta --model_name_or_path $pretrained_model \
    --load_model_path $path/pytorch_model.bin  --dev_filename $dev_file \
    --test_filename $dev_file --output_dir $path \
    --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size 
    
  done
done

