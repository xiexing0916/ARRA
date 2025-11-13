for i in {0..7}
do
  export CUDA_VISIBLE_DEVICES=${i}
  python -u lumina_mgpt/pre_tokenize/pre_tokenize_new.py \
  --splits=8 \
  --rank=${i} \
  --target_size 512 &> ${i}.log &
done