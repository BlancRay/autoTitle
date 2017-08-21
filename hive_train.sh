bazel build -c opt --config=cuda textsum/...
bazel-bin/textsum/seq2seq_attention \
--mode=train \
--article_key=article \
--abstract_key=abstract \
--data_path=data/ckxx/tokens_bin \
--vocab_path=data/ckxx/vocab_dic \
--log_root=log \
--train_dir=log/train \
--max_run_steps=50 \
--max_abstract_sentences=1 \
--use_bucketing=True \
--truncate_input=True \
--num_gpus=1 \
