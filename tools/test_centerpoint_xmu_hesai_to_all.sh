CUDA_VISIBLE_DEVICES=0 python test.py --cfg_file cfgs_fourclass/xmu_ouster_models/centerpoint.yaml    --ckpt /home/djh/projects/xmuda/OpenPCDet/output/xmu_hesai_models/centerpoint/truck/ckpt/checkpoint_epoch_49.pth --launcher none --workers 1 --batch_size 8 --extra_tag hs_49
CUDA_VISIBLE_DEVICES=5 python test.py --cfg_file cfgs_fourclass/xmu_robosense_models/centerpoint.yaml --ckpt /home/djh/projects/xmuda/OpenPCDet/output/xmu_hesai_models/centerpoint/truck/ckpt/checkpoint_epoch_49.pth --launcher none --workers 1 --batch_size 8 --extra_tag hs_49 
CUDA_VISIBLE_DEVICES=7 python test.py --cfg_file cfgs_fourclass/xmu_hesai_models/centerpoint.yaml     --ckpt /home/djh/projects/xmuda/OpenPCDet/output/xmu_hesai_models/centerpoint/truck/ckpt/checkpoint_epoch_49.pth --launcher none --workers 1 --batch_size 8 --extra_tag hs_49  