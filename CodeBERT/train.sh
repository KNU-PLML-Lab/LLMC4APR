#!/bin/bash
###
export CUDA_VISIBLE_DEVICES=0
# export CUDA_VISIBLE_DEVICES=0,1
###

lang=cpp #programming language
batch_size=32 
beam_size=10
lr=5e-5
source_length=512
target_length=64

output_dir=model/$lang
train_file=../data/single_line_r_train.txt
dev_file=../data/single_line_r_valid.txt

pretrained_model=microsoft/codebert-base #Roberta: roberta-base
#pretrained_model=neulab/codebert-cpp

python run.py --do_train --do_eval --model_type roberta --model_name_or_path $pretrained_model \
--train_filename $train_file --dev_filename $dev_file --output_dir $output_dir \
--max_source_length $source_length --max_target_length $target_length --beam_size $beam_size \
--train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr \

