CUDA_VISIBLE_DEVICES=0 nohup python train.py --cfg_file cfgs/xmu_robosense_models/centerpoint.yaml --launcher none --workers 1 --epochs 50 --batch_size 1 > centerpoint_train_xmu_robosense.log 2>&1 &