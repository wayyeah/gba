CUDA_VISIBLE_DEVICES=7 nohup python test.py --cfg_file cfgs_fourclass/xmu_ouster_models/centerpoint.yaml    --ckpt_dir /home/djh/projects/xmuda/OpenPCDet/output/xmu_ouster_models/centerpoint/truck/ckpt    --eval_all  --launcher none --workers 1 --batch_size 8 --extra_tag truck > centerpoint_test_xmu_ouster_fourclass.log    2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python test.py --cfg_file cfgs_fourclass/xmu_robosense_models/centerpoint.yaml --ckpt_dir /home/djh/projects/xmuda/OpenPCDet/output/xmu_robosense_models/centerpoint/truck/ckpt --eval_all  --launcher none --workers 1 --batch_size 8 --extra_tag truck > centerpoint_test_xmu_robosense_fourclass.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python test.py --cfg_file cfgs_fourclass/xmu_hesai_models/centerpoint.yaml     --ckpt_dir /home/djh/projects/xmuda/OpenPCDet/output/xmu_hesai_models/centerpoint/truck/ckpt     --eval_all  --launcher none --workers 1 --batch_size 8 --extra_tag truck > centerpoint_test_xmu_hesai_fourclass.log     2>&1 &