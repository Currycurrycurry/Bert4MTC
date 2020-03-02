export BERT_BASE_DIR=/root/data/BERT/chinese_L-12_H-768_A-12
export DATA_DIR=/root/data/BERT

python3 run_classifier.py \
  --task_name=iron \
  --do_train=true \
  --do_eval=true \
  --data_dir=$DATA_DIR/data_final \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=/root/data/BERT/output
