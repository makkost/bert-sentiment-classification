import os
import sys
import subprocess
from summary import summarize
from parse_args import parse_arguments

SST2 = "glue_data/SST-2"

def main(args):
    for i in range(0, args.split):
        dataset = "{}/set{}".format(*args.path, i)
        print(dataset)
        print(args.model)
        subprocess.run("python run_classifier.py \
        --task_name=SST \
        --do_train=true \
        --do_eval=true \
        --do_predict=true \
        --data_dir={data_dir}/ \
        --vocab_file={bert_dir}/vocab.txt \
        --bert_config_file={bert_dir}/bert_config.json \
        --init_checkpoint={bert_dir}/bert_model.ckpt \
        --max_seq_length=128 \
        --train_batch_size=32 \
        --learning_rate=2e-5 \
        --num_train_epochs=3.0 \
        --output_dir={data_dir}/output/"
        .format(data_dir = dataset, bert_dir = args.model), shell=True)
    
    summarize(args)

if __name__ == '__main__':
    arguments = parse_arguments()
    sys.exit(main(arguments))