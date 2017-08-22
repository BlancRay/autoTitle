bazel build -c opt --config=cuda textsum/...
bazel-bin/textsum/seq2seq_attention \
--mode=train \
--article_key=article \
--abstract_key=abstract \
--data_path=data/all/tokens_bin \
--vocab_path=data/all/vocab_dic \
--log_root=log \
--train_dir=log/train \
--max_run_steps=100000 \
--max_abstract_sentences=100 \
--use_bucketing=True \
--truncate_input=True \
--num_gpus=1 \
--checkpoint_secs=200 \
--beam_size=10 \
--batch_size=32 \
