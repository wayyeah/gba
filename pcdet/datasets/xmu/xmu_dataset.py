import numpy as np
import pickle
import os
import json
import copy
import datetime
from pathlib import Path
import torch

# import open3d as o3d
from pypcd import pypcd

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils
from ..dataset import DatasetTemplate


"""
aiming at UDA now, supporting of fusion detection coming soon
leave a sensor list for it 
make infos for all sensors in a single pkl for it
"""
def crop_sector(points, radius_range, angle_range):
    # 计算极坐标系下的半径和角度
    r = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    theta = np.arctan2(points[:, 1], points[:, 0])
    
    # 将角度转换到 [-π, π) 范围
    theta = np.mod(theta + np.pi, 2 * np.pi) - np.pi
    
    # 应用裁剪条件
    mask = (r >= radius_range[0]) & (r <= radius_range[1]) & (theta >= angle_range[0]) & (theta <= angle_range[1])
    cropped_points = points[mask]
    
    return cropped_points

# 默认任务是针对单个传感器的检测任务，只是在生成和获取pkl的时候忽略这个问题而已
class XMechanismUnmanned(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, sensor=None):
        super().__init__(dataset_cfg, class_names, training, root_path, logger)
        """XMU data tree
        - seq00
        - seq01
            - robosense
            - ouster
            - hesai
            - gpal
            - camera_left
            - camera_front
            - camera_right
        - ...
        """
         
        assert sensor or self.dataset_cfg.SENSOR, "Sensor should be point out explicitly when creating instance of XMUDataset."
        if sensor:
            self.sensor = sensor
        else:
            self.sensor = self.dataset_cfg.SENSOR
        assert self.sensor != 'camera', "coming soon."

        if 'MAP_CLASS_TO_KITTI' in self.dataset_cfg:
            self.map_class_to_kitti = self.dataset_cfg.MAP_CLASS_TO_KITTI

        # training or testing
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        
        self.xmu_infos = [] # using a list to store

        self.split_file = os.path.join(self.root_path, 'ImageSets', self.split + '.txt')
        self.sample_seq_list = [x.strip() for x in open(self.split_file).readlines()]
        self.sample_idx_list = [{'seq': seq, 'frame_id': frame_id} for seq in self.sample_seq_list for frame_id in range(200)]

        if self.mode == 'train':
            assert self.dataset_cfg.DATA_CONFIG_TAR, "no target set config exist."
            self.dataset_cfg_tar = self.dataset_cfg.DATA_CONFIG_TAR
            self.sensor_tar = self.dataset_cfg_tar.SENSOR
            self.split_file_tar = os.path.join(self.root_path, 'ImageSets', self.split + '_top_10_percent.txt')
            self.sample_seq_list_tar = [x.strip() for x in open(self.split_file_tar).readlines()]
            self.sample_idx_list_tar = [{'seq': seq, 'frame_id': frame_id} for seq in self.sample_seq_list_tar for frame_id in range(200)]

            self.src_infos = []
            self.tar_infos = []

            self.include_xmu_data_src_tar(self.mode)        
            self.len_src_ori = len(self.src_infos)
        else:
            self.include_xmu_data(self.mode)
            

    # 从已经写好的xmu 的pkl文件中获取数据
    def include_xmu_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading XMU dataset of sensor %s and mode %s'%(self.sensor, mode))
        xmu_infos = []
        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if info_path is None or not Path(info_path).exists():
                self.logger.info('Info file %s not exists.'%info_path)
                continue
            with open(info_path, 'rb') as f:
                xmu_infos.extend(pickle.load(f))
        
        self.xmu_infos.extend(xmu_infos)

        self.logger.info('Total sequences of sensor %s in mode %s is %d'%(self.sensor, self.mode, len(xmu_infos)))


    # 从已经写好的xmu 的pkl文件中获取数据
    def include_xmu_data_src_tar(self, mode):
        # include src
        if self.logger is not None:
            self.logger.info('Loading source XMU dataset of sensor %s and mode %s'%(self.sensor, mode))
        src_infos = []
        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if info_path is None or not Path(info_path).exists():
                self.logger.info('Info file %s not exists.'%info_path)
                continue
            with open(info_path, 'rb') as f:
                src_infos.extend(pickle.load(f))
        self.src_infos.extend(src_infos)
        self.logger.info('Total source frames of sensor %s in mode %s is %d'%(self.sensor, self.mode, len(src_infos)))
        
        # include tar
        if self.logger is not None:
            self.logger.info('Loading target XMU dataset of sensor %s and mode %s'%(self.sensor, mode))
        tar_infos = []
        for info_path in self.dataset_cfg_tar.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if info_path is None or not Path(info_path).exists():
                self.logger.info('Info file %s not exists.'%info_path)
                continue
            with open(info_path, 'rb') as f:
                tar_infos.extend(pickle.load(f))
        self.tar_infos.extend(tar_infos)
        self.logger.info('Total target frames of sensor %s in mode %s is %d'%(self.sensor, self.mode, len(tar_infos)))
        
        self.xmu_infos = self.src_infos + self.tar_infos
        self.logger.info('Total frames of data in mode %s is %d'%(self.mode, len(self.xmu_infos)))


    def reduce_source_infos(self, ratio=0.25):
        len_src = len(self.src_infos)
        len_tar = len(self.tar_infos)

        assert len_src > 0, "len_src is 0"
        assert len_tar > 0, "len_tar is 0"
        # shuffle src_infos
        # np.random.shuffle(self.src_infos)
        self.src_infos = self.src_infos[int(self.len_src_ori * ratio):]
        assert len(self.src_infos) >= 0, "len_src is 0"
        self.xmu_infos = self.src_infos + self.tar_infos
        if self.logger is not None:
            self.logger.info('Number of source frames are reduced. Total frames: %d; total frame for source data: %d; total frame for target data: %d.' %(len(self.xmu_infos),len(self.src_infos), len(self.tar_infos)))


    def get_label(self, idx):
        seq = idx['seq']
        frame = idx['frame']
        # label_file = os.path.join(self.root_path, 'label', 'seq'+seq, '%s.json'%frame.zfill(4))
        label_file = os.path.join(self.root_path, 'label_ego', 'seq'+seq, '%s.json'%frame.zfill(4))
        assert Path(label_file).exists(), "label file %s not exists."%label_file
        # read json
        with open(label_file, 'r') as f:
            label = json.load(f)
        label = label['items']
        '''
        9DoF box, class, track_id, confidence, occulusion, act
        '''
        boxes_7DoF = np.zeros((len(label), 7), dtype=np.float32)
        boxes_9DoF = np.zeros((len(label), 9), dtype=np.float32)
        # classes = np.zeros((len(label), 1), dtype=str) # 为什么这么操作不对，要用list
        classes = []
        track_ids = np.zeros((len(label), 1), dtype=np.int32)
        confidences = np.zeros((len(label), 1), dtype=np.float32)
        occulusions = np.zeros((len(label), 1), dtype=np.int32)
        acts = np.zeros((len(label), 1), dtype=str)
        points_num = np.zeros((len(label), 1), dtype=np.int32)

        for i, item in enumerate(label):
            boxes_xyz = np.zeros((3, ), dtype=np.float32)
            boxes_xyz[0] = item['position']['x']
            boxes_xyz[1] = item['position']['y']
            boxes_xyz[2] = item['position']['z']

            boxes_lwh = np.zeros((3, ), dtype=np.float32)
            boxes_lwh[0] = item['dimension']['x']
            boxes_lwh[1] = item['dimension']['y']
            boxes_lwh[2] = item['dimension']['z']

            boxes_pitch_roll_yaw = np.zeros((3, ), dtype=np.float32)
            boxes_pitch_roll_yaw[0] = item['rotation']['x']
            boxes_pitch_roll_yaw[1] = item['rotation']['y']
            boxes_pitch_roll_yaw[2] = item['rotation']['z']
            
            boxes_7DoF[i,:3] = boxes_xyz
            boxes_7DoF[i,3:6] = boxes_lwh
            boxes_7DoF[i,6] = boxes_pitch_roll_yaw[2]
            
            boxes_9DoF[i,:6] = boxes_7DoF[i, :6]
            boxes_9DoF[i,6:9] = boxes_pitch_roll_yaw

            # classes[i] = item['categoryId']
            classes.append(item['categoryId'])
            track_ids[i] = item['trackId']
            if 'confidence' not in item:
                confidences[i] = -1
                print('confidence error in the %d label file %s' % (i,label_file))
            # assert 'confidence' in item, 'confidence not in label file %s' % label_file
            else:
                confidences[i] = item['confidence']
            occulusions[i] = item['occ']
            acts[i] = item['act']
            points_num[i] = item['pointsNum']

            # no attribute lost 
            assert item['pointsNum']>=0 , 'attribute pointsNum lost in label file %s' % label_file
            assert item['occ'] , 'attribute occ lost in label file %s' % label_file
            assert item['act'] , 'attribute act lost in label file %s' % label_file
            assert item['categoryId'] , 'attribute categoryId lost in label file %s' % label_file
            assert item['trackId'] , 'attribute trackId lost in label file %s' % label_file

        

        # mapping 
        assert 'TRAINING_CATEGORIES_MAPPING' in self.dataset_cfg, 'TRAINING_CATEGORIES_MAPPING not in dataset_cfg'
        if self.dataset_cfg.TRAINING_CATEGORIES_MAPPING:
            # print(classes, classes.shape)
            for i in range(len(classes)):
                if classes[i] in self.dataset_cfg.TRAINING_CATEGORIES_MAPPING:
                    classes[i] = self.dataset_cfg.TRAINING_CATEGORIES_MAPPING[classes[i]]
                    # 被截断了，Pedestria，要非常注意改变numpy的str类型，长度不可变
                    # assert classes[i] in ['Car', 'Truck', 'Pedestrian', 'Cyclist'], 'class %s not in [Car, Truck, Pedestrian, Cyclist]'%classes[i]
                else:
                    classes[i] = 'DontCare'
        classes = np.array(classes)
        return boxes_7DoF, boxes_9DoF, classes, track_ids, confidences, occulusions, acts, points_num


    def get_lidar(self, idx, sensor, num_features=4):
        seq = idx['seq']
        frame = idx['frame']
        assert sensor, "sensor should be specified."
        if sensor == 'camera':
            raise NotImplementedError
        
        lidar_file_list = os.listdir(os.path.join(self.root_path, 'seq'+seq, sensor))
        lidar_file_list.sort() #一定要sort，不然顺序不对
        lidar_file = os.path.join(self.root_path, 'seq'+seq, sensor, lidar_file_list[int(frame)])
        assert Path(lidar_file).exists(), "lidar file %s not exists."%lidar_file
        # read pcd
        lidar = pypcd.PointCloud.from_path(lidar_file)

        assert num_features==4, 'only support xyzi currently'
        
        pc_x = lidar.pc_data['x']
        pc_y = lidar.pc_data['y']
        pc_z = lidar.pc_data['z']
        pc_i = lidar.pc_data['intensity']
        lidar = np.stack([pc_x, pc_y, pc_z, pc_i], axis=1)

        # filter nan
        lidar = lidar[~np.isnan(lidar).any(axis=1)]

        return lidar


    #√ 这里还需要看看问题imu
    def get_calib(self, idx, sensor):
        seq = idx['seq']
        # assert seq >=0 and seq<=49, "seq should be in [0, 49]"
        assert sensor, "sensor should be specified."
        calib_file = os.path.join(self.root_path, 'calib_to_ego', 'transformation_matrix_%s_ego.txt'%sensor)
        assert Path(calib_file).exists(), "calib file %s not exists."%calib_file
        # read calib
        calib = np.loadtxt(calib_file, dtype=np.float32)
        return calib

    def get_pose(self, idx, sensor=None):
        # assert seq >=0 and seq<=49, "seq should be in [0, 199]"
        # if sensor in ['robosense', 'ouster', 'hesai', 'gpal']:
        #     pose_file = os.path.join(self.root_path, seq , 'pose', '%s.txt'%frame.zfill(4))
        # elif sensor in ['left', 'front', 'right']:
        #     pose_file = os.path.join(self.root_path, seq , 'pose', '%s.txt'%frame.zfill(4))
        # else:
        #     raise NotImplementedError
        # assert pose_file.exists(), "pose file %s not exists."%pose_file
        # # read pose
        # pose = np.loadtxt(pose_file, dtype=np.float32)
        # return pose

        # TODO: pose
        # output with time
        # self.logger.info('pose is not implemented, return None.', 'date: %s'%datetime.datetime.now())
        seq = idx['seq']
        frame = idx['frame']

        return None

    # 获取train/val/test的序列到sample_id_list中
    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split
        split_dir = os.path.join(self.root_path, 'ImageSets', self.split+'.txt')
        self.sample_seq_list = [x.strip() for x in open(split_dir).readlines()]

    def __len__(self):
        assert self._merge_all_iters_to_one_epoch ==False, 'not support merge_all_iters_to_one_epoch'
        if self._merge_all_iters_to_one_epoch:
            return len(self.sample_idx_list) * self.total_epochs
        
        return len(self.xmu_infos)

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        if self.merge_all_iters_to_one_epoch:
            index = index % len(self.xmu_infos)
        
        if (self.mode == 'train'):
            if (index >= len(self.src_infos)):
                is_src = False
                self.sensor = self.sensor_tar
            else:
                is_src = True

        info = copy.deepcopy(self.xmu_infos[index])
        idx = info['idx']
        # get lidar
        points = self.get_lidar(idx, sensor=self.sensor)
        # calib = info['calibs'][self.sensor+'_to_ego']
        # to ego system
        pose = self.get_pose(idx, sensor=self.sensor)
        # all_info = {}
        to_ego = True
        if to_ego:
            # transform points to ego vehicle coord with calib using matrix multiplication
            calib = info['calibs'][self.sensor+"_to_ego"]
            # print('calib: ', calib)
            points_new = np.concatenate((points[:, :3], np.ones((points.shape[0], 1))), axis=1)
            # all_info['points_new'] = points_new
            # points_new1 = np.dot(points_new, calib)
            points_new1 = np.dot(calib, points_new.T).T # 这样才是对的！！！
            points = np.concatenate((points_new1[:, :3], points[:, 3:]), axis=1)
            

        assert 'annos' in info, 'no annos in info'
        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            classes = annos['name']
            boxes_7DoF = annos['boxes_7DoF']
            num_gt = boxes_7DoF.shape[0]
            if num_gt> 0:
                point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                    torch.from_numpy(points[:, 0:3]), torch.from_numpy(boxes_7DoF)
                ).numpy() # (num_gt, num_points)
                existance = np.array([False for i in range(num_gt)])
                for i in range(num_gt):
                    if points[point_indices[i, :] > 0].shape[0] > 0:
                        existance[i] = True
                boxes_7DoF = boxes_7DoF[existance]
                classes = classes[existance]

        radius_range = [0, 150]  # 半径范围
        angle_range = [-np.pi / 3, np.pi / 3]  # 60度到负60度，转换为弧度
        cropped_points = crop_sector(points, radius_range, angle_range)
        points=cropped_points
        input_dict = {
            # 'db_flag': "xmu_%s" % self.sensor,
            # 'frame_id': self.sample_idx_list[index],
            'frame_id': info['idx'],
            'points': points,
            'calib': calib,
            'gt_boxes': boxes_7DoF,
            'gt_names': classes,
        }

        if self.mode == 'train':
            data_dict = self.prepare_data(input_dict, is_src=is_src)
        else:
            data_dict = self.prepare_data(input_dict)        

        return data_dict

    
    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.xmu_infos[0].keys():
            return 'No gt boxes for eval', {}
        
        def kitti_eval(eval_det_annos, eval_gt_annos, map_name_to_kitti):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils

            # kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_name_to_kitti, 
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False))
            kitti_class_name = [map_name_to_kitti[x] for x in class_names]
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_name
            )
            return ap_result_str, ap_dict

        def once_eval(eval_det_annos, eval_gt_annos):
            from ..once.once_eval.evaluation import get_evaluation_results

            ap_result_str, ap_dict = get_evaluation_results(gt_annos=eval_gt_annos, pred_annos=eval_det_annos, classes=class_names)

            return ap_result_str, ap_dict

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.xmu_infos]

        # store for debug
        with open ('eval_det_annos.pkl', 'wb') as f:
            pickle.dump(eval_det_annos, f)
        with open ('eval_gt_annos.pkl', 'wb') as f:
            pickle.dump(eval_gt_annos, f)

        if kwargs['eval_metric'] == 'kitti':
            ap_result_str, ap_dict =  kitti_eval(eval_det_annos, eval_gt_annos, self.map_name_to_kitti)
        elif kwargs['eval_metric'] == 'once':
            ap_result_str, ap_dict = once_eval(eval_det_annos, eval_gt_annos)
        else:
            raise NotImplementedError
        
        return ap_result_str, ap_dict
    
    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
    # def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7 or 9), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        
        def get_template_prediction(num_samples):
            # box_dim = 9 if self.dataset_cfg.get('TRAIN_WITH_SPEED', False) else 7
            box_dim = 7
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, box_dim]), 'pred_labels': np.zeros(num_samples)
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            pred_dict['pred_labels'] = pred_labels

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            if 'metadata' in batch_dict:
                single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)

        return annos


    # 用来生成写入到pkl文件的数据集信息的
    def get_infos(self, class_names, num_workers=4, has_lable=True, sample_seq_list=None, num_features=4):
        import concurrent.futures as futures

        """
        - seq
        - frame
        - sensor_paths
            - robosense
            - ouster
            - hesai
            - gpal
        - image_paths
            - left
            - front
            - right
        - num_features
        - calib
            - robosense_to_ego
            - ouster_to_ego
            - hesai_to_ego
            - gpal_to_ego
            - ego_to_world
        - pose
        - label_path
        """
        def process_single_sequence(seq_idx):
            # seq_idx = str(seq_idx).zfill(2)
            print("processing sequence {} ...".format(seq_idx))

            sensors = ['robosense', 'ouster', 'hesai' ,'gpal']
            cameras = ['left', 'front', 'right']

            sensor_paths = {sensor: [] for sensor in sensors}
            for sensor in sensors:
                sensor_path = os.path.join(self.root_path, 'seq'+seq_idx, sensor)
                sensor_paths[sensor] = [os.path.join(self.root_path, 'seq'+seq_idx, sensor, path) for path in os.listdir(os.path.join(self.root_path, 'seq'+seq_idx, sensor))] 
                assert len(sensor_paths[sensor]) == 200, "sequence {} has {} {} frames, rather than 200!".format(seq_idx, len(sensor_paths[sensor]), sensor)

            image_paths = {camera: [] for camera in cameras}
            for camera in cameras:
                image_path = os.path.join(self.root_path, 'seq'+seq_idx, 'camera_'+camera)
                image_paths[camera] = [os.path.join(self.root_path, 'seq'+seq_idx, 'camera_'+camera, path) for path in os.listdir(os.path.join(self.root_path, 'seq'+seq_idx, 'camera_'+camera))]
                assert len(image_paths[camera]) == 200, "sequence {} has {} {} frames, rather than 200!".format(seq_idx, len(image_paths[camera]), camera)
            
            # robosense_paths = os.listdir(os.path.join(self.root_path, 'seq'+seq_idx, 'robosense'))
            # ouster_paths = os.listdir(os.path.join(self.root_path, 'seq'+seq_idx, 'ouster'))
            # hesai_paths = os.listdir(os.path.join(self.root_path, 'seq'+seq_idx, 'hesai'))
            # gpal_paths = os.listdir(os.path.join(self.root_path, 'seq'+seq_idx, 'gpal'))
            # camera_left_paths = os.listdir(os.path.join(self.root_path, 'seq'+seq_idx, 'camera_left'))
            # camera_front_paths = os.listdir(os.path.join(self.root_path, 'seq'+seq_idx, 'camera_front'))
            # camera_right_paths = os.listdir(os.path.join(self.root_path, 'seq'+seq_idx, 'camera_right'))
            # assert len(robosense_paths) == len(ouster_paths) == len(hesai_paths) == len(gpal_paths) == len(camera_left_paths) == len(camera_front_paths) == len(camera_right_paths), "the number of lidar and image is not equal."
            # assert len(robosense_paths) == 200, "the number of lidar and image is not equal to 200."

            calibs = {}
            idx = {'seq': seq_idx , 'frame': 0}
            for sensor in sensors:
                calibs['%s_to_ego' % sensor] = self.get_calib(idx, sensor)

            info = {}
            seq_infos = []

            for i in range(200):
                info['seq'] = str(seq_idx).zfill(2)
                info['frame'] = str(i).zfill(4)
                idx = {'seq': info['seq'], 'frame': info['frame']}
                info['idx'] = idx
                # info['lidar_paths'] = {}
                # info['lidar_paths'].update({
                #     'robosense': os.path.join(self.root_path, 'seq'+seq_idx, 'robosense', robosense_paths[i]),
                #     'ouster': os.path.join(self.root_path, 'seq'+seq_idx, 'ouster', ouster_paths[i]),
                #     'hesai': os.path.join(self.root_path, 'seq'+seq_idx, 'hesai', hesai_paths[i]),
                #     'gpal': os.path.join(self.root_path, 'seq'+seq_idx, 'gpal', gpal_paths[i]),
                # })
                # info['image_paths'] = {}
                # info['image_paths'].updat({
                #     'left': os.path.join(self.root_path, 'seq'+seq_idx, 'camera_left', camera_left_paths[i]),
                #     'front': os.path.join(self.root_path, 'seq'+seq_idx, 'camera_front', camera_front_paths[i]),
                #     'right': os.path.join(self.root_path, 'seq'+seq_idx, 'camera_right', camera_right_paths[i]),
                # })


                info['sensor_paths'] = {}
                for sensor in sensors:
                    info['sensor_paths'].update({
                        sensor: sensor_paths[sensor][i]
                    })
                info ['image_paths'] = {}
                for camera in cameras:
                    info['image_paths'].update({
                        camera: image_paths[camera][i]
                    })

                info['num_features'] = num_features
                info['calibs'] = {}
                info['calibs'].update(calibs)
                info['pose'] = {}
                for sensor in sensors:
                    info['pose'].update({
                        sensor: self.get_pose(idx, sensor)
                    })
                for camera in cameras:
                    info['pose'].update({
                        camera: self.get_pose(idx, camera)
                    })

                info['label_path'] = os.path.join(self.root_path, 'label_ego', 'seq'+seq_idx, str(i).zfill(4)+'.json')
                assert os.path.exists(info['label_path']), "the label file {} does not exist!".format(info['label_path'])
            
                if has_lable:
                    annotations = {}
                    boxes_7DoF, boxes_9DoF, classes, track_ids, confidences, occulusions, acts, points_num = self.get_label(idx=idx)
                    annotations['boxes_7DoF'] = boxes_7DoF
                    annotations['name'] = classes
                    info['annos'] = annotations

                seq_infos.append(copy.deepcopy(info)) # 复制出了问题？？
                # print(info)
            return seq_infos
        
        sample_seq_list = sample_seq_list if sample_seq_list is not None else self.sample_seq_list
        print('sample_seq_list: ', sample_seq_list)
        # with futures.ThreadPoolExecutor(num_workers) as executor:
        #     infos = executor.map(process_single_sequence, sample_seq_list)
        #     # print('infos: ', infos)

        # # print('infos type:', type(infos), len(infos))
        # infos_l = list(infos)
        # print('infos type:', type(infos_l), len(infos_l))
        # # print('infos_total: ', infos)
        # all_infos = []
        # for info in infos_l:
        #     # print('info: ', info)
        #     all_infos += info
        # print(len(all_infos), type(all_infos))
        # return all_infos
        all_infos = []      
        # 是不是变量复制出了问题？？？
        # print(sample_seq_list)
        # print(self.sample_seq_list)
        # exit()

        for seq_idx in sample_seq_list:
            single_seq_infos = process_single_sequence(seq_idx)
            all_infos += copy.deepcopy(single_seq_infos)
        print('len all_infos: ', len(all_infos))
        # print('all_infos: ', all_infos)
        return all_infos

    # 对于每一个传感器，有一个gt_database，一起生成了
    def create_groundtruth_database(self, info_path, used_classes=None, split='train'):
        print(used_classes)
        import torch
        assert split in ['train', 'val', 'test']
        # sensors = ['robosense', 'ouster', 'hesai', 'gpal']
        # sensors = ['gpal']
        sensors = ['robosense', 'ouster', 'hesai']
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
        
        print('len info when create gt_database: ', len(infos))
        # print('info when create gt_database: ', infos)
        for sensor in sensors:
            database_save_path = Path(self.root_path) / ('gt_database_%s_fourclass' % sensor if split == 'train' else 'gt_database_%s_%s_fourclass' % (self.sensor, split))
            db_info_save_path = Path(self.root_path) / ('gt_database_info_%s_fourclass.pkl' % sensor if split == 'train' else 'gt_database_info_%s_%s_fourclass.pkl' % (self.sensor, split))

            database_save_path.mkdir(parents=True, exist_ok=True)
            all_db_infos = {}
            print('generation gt_database for sensor: %s' % sensor)
            for i in range(len(infos)):
                print('%s gt_database sample: %d/%d' % (sensor, i+1, len(infos)))
                info = infos[i]
                idx = info['idx']
                points = self.get_lidar(idx=idx, sensor=sensor, num_features=4)
                annos = info['annos']
                gt_boxes_7DoF = annos['boxes_7DoF']
                gt_names = annos['name']

                num_gt = gt_boxes_7DoF.shape[0]
                point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                    torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes_7DoF)
                ).numpy() # (num_gt, num_points)

                for i in range(num_gt):
                    file_name = '%s_%s_%s_%d.bin' % (idx['seq'], idx['frame'], gt_names[i], i)
                    file_path = database_save_path / file_name
                    gt_points = points[point_indices[i, :] > 0]
                    if gt_points.shape[0] == 0: # 滤掉没有点的gt_box
                        continue
                    
                    gt_points[:, :3] -= gt_boxes_7DoF[i, :3]

                    with open(file_path, 'wb') as f:
                        gt_points.tofile(f)
                    
                    if (used_classes is None) or (gt_names[i] in used_classes):
                        db_path = str(file_path.relative_to(self.root_path))
                        db_info = {'name': gt_names[i], 'path': db_path, 'gt_idx': i,
                                   'box3d_lidar': gt_boxes_7DoF[i], 'num_points_in_gt': gt_points.shape[0]}
                        if gt_names[i] in all_db_infos.keys():
                            all_db_infos[gt_names[i]].append(db_info)
                        else:
                            all_db_infos[gt_names[i]] = [db_info]

            for k, v in all_db_infos.items():
                print('sensor--%s database of %s : %d' % (sensor, k, len(v)))
            with open(db_info_save_path, 'wb') as f:
                pickle.dump(all_db_infos, f)

    @staticmethod
    def create_lable_file_with_name_and_box(class_names, gt_names, gt_boxes, save_label_path):
        with open(save_label_path, 'w') as f:
            for idx in range(gt_boxes.shape[0]):
                boxes = gt_boxes[idx]
                name = gt_names[idx]
                if name not in class_names:
                    continue
                line = "{name} {x} {y} {z} {l} {w} {h} {yaw}\n".format(
                    name=name, x=boxes[0], y=boxes[1], z=boxes[2],
                    l=boxes[3], w=boxes[4], h=boxes[5], yaw=boxes[6]
                )
                f.write(line)

# block for creating dataset infos
def create_xmu_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = XMechanismUnmanned(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training =True, logger=common_utils.create_logger()
    ) # 这里设置training为False会不会影响到后面的数据读取？？
    train_split, val_split, test_split = 'train', 'val', 'test'

    train_filename = os.path.join(save_path, 'xmu_infos_%s_fourclass.pkl'%train_split)
    val_filename = os.path.join(save_path, 'xmu_infos_%s_fourclass.pkl'%val_split)
    test_filename = os.path.join(save_path, 'xmu_infos_%s_fourclass.pkl'%test_split)

    print ('---------------Start to create xmu infos---------------')

    dataset.set_split(train_split)
    xmu_infos_train = dataset.get_infos(class_names, num_workers=workers, has_lable=True, sample_seq_list=None, num_features=4)
    # # print('len xmu_infos_train: ', len(xmu_infos_train))
    # # print('sample xmu_infos_train: ', xmu_infos_train[0])
    # # print('sample xmu_infos_train: ', xmu_infos_train[1])
    # # print('sample xmu_infos_train: ', xmu_infos_train[390])
    # # print('sample xmu_infos_train: ', xmu_infos_train[391])
    
    with open(train_filename, 'wb') as f:
        pickle.dump(xmu_infos_train, f)
    print ('xmu %s samples are saved to %s'%(train_split, train_filename))

    dataset.set_split(val_split)
    xmu_infos_val = dataset.get_infos(class_names, num_workers=workers, has_lable=True, sample_seq_list=None, num_features=4)
    with open(val_filename, 'wb') as f:
        pickle.dump(xmu_infos_val, f)
    print ('xmu %s samples are saved to %s'%(val_split, val_filename))

    dataset.set_split(test_split)
    xmu_infos_test = dataset.get_infos(class_names, num_workers=workers, has_lable=True, sample_seq_list=None, num_features=4)
    # print('xmu_info_test: ', xmu_infos_test)
    with open(test_filename, 'wb') as f:
        pickle.dump(xmu_infos_test, f)
    print ('xmu %s samples are saved to %s'%(test_split, test_filename))

    print ('Start creating xmu groundtruth database for %s split'%train_split)
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, used_classes=class_names, split=train_split)
    print ('Info generation done for XMechanismUnmanned dataset!')



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='arg parser for xmu dataset')
    parser.add_argument('--cfg_file', type=str, default=None, help='configuration for creating xmu infos')
    parser.add_argument('--func', type=str, default='create_kitti_infos', help='function of creating xmu infos')
    args = parser.parse_args()

    if args.func == 'create_xmu_infos' :
        import yaml
        from easydict import EasyDict
        from pathlib import Path

        dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
        assert dataset_cfg.SENSOR == 'NotSelected', 'all the sensor should be generate'
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        sensors = ['robosense', 'ouster', 'hesai', 'gpal']
        create_xmu_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Truck', 'Pedestrian', 'Cyclist'],
            # sensors=sensors,
            data_path=ROOT_DIR / 'data' / 'xmu',
            save_path=ROOT_DIR / 'data' / 'xmu', 
        )

# test function