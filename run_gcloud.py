import sys
import subprocess

BERT_BASE_DIR="gs://bert-302207/uncased_L-12_H-768_A-12"
GLUE_DIR="./"
TPU_NAME="bert-tpu"

def main(bert = BERT_BASE_DIR, glue = GLUE_DIR, tpu_name = TPU_NAME):
    subprocess.Popen("python3 run_classifier.py \
    --task_name=SST \
    --do_train=true \
    --do_eval=true \
    --data_dir={glue_dir}/SST-2 \
    --vocab_file={bert_dir}/vocab.txt \
    --bert_config_file={bert_dir}/bert_config.json \
    --init_checkpoint={bert_dir}/bert_model.ckpt \
    --max_seq_length=128 \
    --train_batch_size=32 \
    --learning_rate=2e-5 \
    --num_train_epochs=3.0 \
    --use_tpu=True \
    --tpu_name={tpu} \
    --output_dir=gs://bert-302207/sst_output/"
    .format(glue_dir = glue, bert_dir = bert, tpu = tpu_name), shell=True)
    

if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))