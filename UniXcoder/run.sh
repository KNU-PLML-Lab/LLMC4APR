# pip install torch==1.6.0+cu92 torchvision==0.7.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html > log.txt 2>&1
# pip install sklearn scipy transformers tqdm > log.txt 2>&1
CUDA_VISIBLE_DEVICES=0

lang=cpp #programming language
lr=5e-5
batch_size=32
accm_steps=1
beam_size=10
source_length=512
target_length=150
output_dir=models/$lang

data_dir=../../dataset
train_file=../ksc/single_line_r_train.txt
dev_file=../ksc/single_line_r_valid.txt
epochs=10
# pretrained_model=../../../pretrained-model/UniXcoder-base/
pretrained_model=microsoft/unixcoder-base

mkdir -p $output_dir
python run.py \
--do_train \
--do_eval \
--model_name_or_path $pretrained_model \
--train_filename $train_file \
--dev_filename $dev_file \
--tokenizer_name roberta-base \
--output_dir $output_dir \
--max_source_length $source_length \
--max_target_length $target_length \
--beam_size $beam_size \
--train_batch_size $batch_size \
--eval_batch_size $batch_size \
--learning_rate $lr \
--gradient_accumulation_steps $accm_steps \
--num_train_epochs $epochs 2>&1| tee $output_dir/train.log


batch_size=64
train_file=../ksc/single_line_r_train.txt
dev_file=../ksc/single_line_r_valid.txt
test_model=$output_dir/checkpoint-last/pytorch_model.bin #checkpoint for test

python run.py \
--do_test \
--model_name_or_path $pretrained_model \
--load_model_path $test_model \
--dev_filename $dev_file \
--test_filename $test_file \
--output_dir $output_dir \
--max_source_length $source_length \
--max_target_length $target_length \
--beam_size $beam_size \
--gradient_accumulation_steps $accm_steps \
--eval_batch_size $batch_size 2>&1| tee $output_dir/test.log
