date +%c
date +%s 
start=`date +%s`
bazel build -c opt --config=cuda textsum/...
bazel-bin/textsum/seq2seq_attention \
--mode=train \
--article_key=article \
--abstract_key=abstract \
--data_path=data/cctv/tokens_char_bin \
--vocab_path=data/cctv/vocab_dic_char \
--log_root=log/cctv \
--train_dir=log/cctv/train \
--max_run_steps=100000 \
--max_abstract_sentences=2 \
--use_bucketing=True \
--truncate_input=True \
--num_gpus=1 \
--checkpoint_secs=200 \
--batch_size=64 \
--loss_stop=0.000001 \
--num_softmax_samples=100 \

date +%c
date +%s 
end=`date +%s`
echo $(($end-$start))