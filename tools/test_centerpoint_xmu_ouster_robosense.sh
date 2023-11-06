# CUDA_VISIBLE_DEVICES=5 nohup python test.py --cfg_file cfgs/xmu_robosense_models/centerpoint.yaml --ckpt /home/djh/projects/xmuda/OpenPCDet/output/xmu_ouster_models/centerpoint/default/ckpt/checkpoint_epoch_80.pth  --launcher none --workers 1 --batch_size 2 > centerpoint_test_xmu_ouster_robosense.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python test.py --cfg_file cfgs_fourclass/xmu_robosense_models/centerpoint.yaml --ckpt_dir /home/djh/projects/xmuda/SSDA3D/output/stage1_cutmix/centerpoint_100_ouster_10_robosense_frames_cutmix/truck/ckpt --eval_all  --launcher none --workers 1 --batch_size 4 --extra_tag ssda3d_os_rs > centerpoint_test_xmu_ouster_robosense.log 2>&1 &