# exp 5
 python main_nlp.py configs/resnet50_lincl.py

#exp 1
#srun -p v100 -w node20 --gres=gpu:1 python main_cla.py \
#--save_path 'resner101_random' \
#--batch_size 64 --num_workers 16 \
#--model_path ''


## exp 2
#srun -p v100 -w node20 --gres=gpu:1 python main_cla.py \
#--save_path 'resnet101_ep50' \
#--batch_size 64 --num_workers 2 \
#--model_path 'test_models/baseline_lr0.001_bs256_new_encoder-50-1500.ckpt'

## exp 3 node13
#srun -p eng2080ti --gres=gpu:1 python main_cla.py \
#--save_path 'resnet101_ep200' \
#--batch_size 32 --num_workers 2 \
#--model_path 'test_models/baseline_lr0.001_bs256_new_encoder-200-1500.ckpt'

## exp 4 v100 refine lr
#srun -p v100 --gres=gpu:1 python main_cla.py \
#--save_path 'resnet101_ep200_lr0.0005' \
#--batch_size 32 --num_workers 2 --learning_rate 0.0005 \
#--model_path 'test_models/baseline_lr0.001_bs256_new_encoder-200-1500.ckpt'
