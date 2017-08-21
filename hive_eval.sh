bazel-bin/textsum/seq2seq_attention \
--mode=eval \
--article_key=article \
--abstract_key=abstract \
--data_path=data/net/tokens_bin \
--vocab_path=data/ckxx/vocab_dic \
--log_root=log \
--eval_dir=log/eval \
--num_gpus=1 \
--use_bucketing=True 
