#python make_offical_result.py --epoch 90 --mode 'single_high' --max_num 30

export epoch=100
for max_num in 5 10 20 30
do
  for mode in 'single_low' 'single_high' 'sorted' 'random'
  do
    python make_offical_result.py --epoch ${epoch} --mode ${mode} --max_num ${max_num}
  done
done
