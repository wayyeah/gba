import copy
import pickle
import os

import numpy as np

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils
from ..dataset import DatasetTemplate


class LiDARCSDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]

        if 'SET_SPLIT_TAG' in self.dataset_cfg:
            self.set_split_tag = self.dataset_cfg.SET_SPLIT_TAG
            split_dir = os.path.join(self.root_path, 'ImageSets'+'_'+self.set_split_tag, (self.split + '.txt'))
        else:
            split_dir = os.path.join(self.root_path, 'ImageSets', (self.split + '.txt'))

        # print('split', split_dir)
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if os.path.exists(split_dir) else None
        # print('sample_id_list', self.sample_id_list)

        self.lidarcs_infos = []
        self.include_data(self.mode)
        if 'MAP_CLASS_TO_KITTI' in self.dataset_cfg:
            self.map_class_to_kitti = self.dataset_cfg.MAP_CLASS_TO_KITTI

        self.sensor = self.dataset_cfg.SENSOR

    def include_data(self, mode):
        # todo log for different
        self.logger.info('Loading LiDAR-CS %s dataset.' % self.dataset_cfg.SENSOR)
        lidarcs_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                self.logger.info('No such file: %s' % info_path)
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                lidarcs_infos.extend(infos)

        self.lidarcs_infos.extend(lidarcs_infos)
        self.logger.info('Total samples for lidarcs %s dataset: %d' % (self.dataset_cfg.SENSOR, (len(lidarcs_infos))))

    def get_label(self, idx):
        label_file = self.root_path / 'label' / ('%s.txt' % idx)
        assert label_file.exists()
        with open(label_file, 'r') as f:
            lines = f.readlines()

        # [N, 8]: (category_name x y z dx dy dz heading_angle)
        gt_boxes = []
        gt_names = []
        for line in lines:
            line_list = line.strip().split(' ')
            gt_boxes.append(line_list[1:])
            gt_names.append(line_list[0])

        ## not in lidarcs info generating process
        # assert 'TRAINING_CATEGORIES_MAPPING' in self.dataset_cfg , 'TRAINING_CATEGORIES_MAPPING not in dataset_cfg'
        # if self.dataset_cfg.TRAINING_CATEGORIES_MAPPING is not None:
        #     gt_names = np.array([self.dataset_cfg.TRAINING_CATEGORIES_MAPPING.get(lab, lab)
        #                        for lab in gt_names])

        return np.array(gt_boxes, dtype=np.float32), np.array(gt_names)

    def get_lidar(self, idx):
        lidar_file = self.root_path / 'bin' / ('%s.bin' % idx)
        assert lidar_file.exists()
        point_features = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4) # reshape是否对？
        return point_features
    
    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split
        if 'SET_SPLIT_TAG' in self.dataset_cfg:
            self.set_split_tag = self.dataset_cfg.SET_SPLIT_TAG
            split_dir = self.root_path / 'ImageSets'+ '_' + self.set_split_tag / (self.split + '.txt')
        else:
            split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.sample_id_list) * self.total_epochs

        return len(self.lidarcs_infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.lidarcs_infos)

        info = copy.deepcopy(self.lidarcs_infos[index])
        sample_idx = info['point_cloud']['lidar_idx']
        points = self.get_lidar(sample_idx)
        input_dict = {
            # 'db_flag': "lidarcs_%s" % self.sensor.replace('-', '').lower(),
            'frame_id': self.sample_id_list[index],
            'points': points
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            gt_names = annos['name']
            gt_boxes_lidar = annos['gt_boxes_lidar']
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict

    @staticmethod
    # def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
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

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.lidarcs_infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def kitti_eval(eval_det_annos, eval_gt_annos, map_name_to_kitti):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils

            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            kitti_class_names = [map_name_to_kitti[x] for x in class_names]
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
            )
            return ap_result_str, ap_dict

        def once_eval(eval_det_annos, eval_gt_annos):
            from ..once.once_eval.evaluation import get_evaluation_results

            # print(type(eval_det_annos))
            # print(eval_det_annos)
            # # print(eval_gt_annos)
            # exit()
            pickle.dump(eval_gt_annos, open('/home/djh/projects/xmuda/OpenPCDet/eval_gt_annos.pkl', 'wb'))
            pickle.dump(eval_det_annos, open('/home/djh/projects/xmuda/OpenPCDet/eval_det_annos.pkl', 'wb'))

            ap_result_str, ap_dict = get_evaluation_results(
                gt_annos=eval_gt_annos, pred_annos=eval_det_annos, classes=class_names
            )
            return ap_result_str, ap_dict

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.lidarcs_infos]

        # try
        with open('eval_test_det_annos.pkl', 'wb') as f:
            pickle.dump(eval_det_annos, f)
        with open('eval_test_gt_annos.pkl', 'wb') as f:
            pickle.dump(eval_gt_annos, f)

        if kwargs['eval_metric'] == 'kitti':
            ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos, self.map_class_to_kitti)
        

        
        elif kwargs['eval_metric'] == 'once':
            ap_result_str, ap_dict = once_eval(eval_det_annos, eval_gt_annos)

        else:
            raise NotImplementedError

        return ap_result_str, ap_dict

    def get_infos(self, class_names, num_workers=4, has_label=True, sample_id_list=None, num_features=4):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            # info['point_cloud': {num_features: 4, lidar_idx: 000000}, 'annos': {'name': ['Car', 'Car'], 'gt_boxes_lidar': [[x, y, z, dx, dy, dz, heading], ...]
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': num_features, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            if has_label:
                annotations = {}
                gt_boxes_lidar, name = self.get_label(sample_idx)
                annotations['name'] = name
                annotations['gt_boxes_lidar'] = gt_boxes_lidar[:, :7]
                info['annos'] = annotations

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list

        # create a thread pool to improve the velocity
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def create_groundtruth_database(self, sensor=None, info_path=None, used_classes=None, split='train'):
        import torch

        assert sensor is not None, 'sensor must be specified'
        database_save_path = Path(self.root_path) / ('gt_database_%s' % sensor if split == 'train' else ('gt_database_%s_%s' % (sensor, split)))
        db_info_save_path = Path(self.root_path) / ('lidarcs_%s_dbinfos_%s.pkl' % (sensor,split))

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]

        # Output the num of all classes in database
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def create_label_file_with_name_and_box(class_names, gt_names, gt_boxes, save_label_path):
        with open(save_label_path, 'w') as f:
            for idx in range(gt_boxes.shape[0]):
                boxes = gt_boxes[idx]
                name = gt_names[idx]
                if name not in class_names:
                    continue
                line = "{name} {x} {y} {z} {l} {w} {h} {angle} \n".format(
                    name=name,  x=boxes[0], y=boxes[1], z=(boxes[2]), 
                    l=boxes[3], w=boxes[4], h=boxes[5], angle=boxes[6]
                )
                f.write(line)


def create_lidarcs_infos(sensor, dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = LiDARCSDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )
    train_split, val_split, test_split = 'train', 'val', 'test'
    num_features = len(dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)

    train_filename = save_path / ('lidarcs_%s_infos_%s.pkl' % (sensor, train_split))
    val_filename = save_path / ('lidarcs_%s_infos_%s.pkl' % (sensor, val_split))
    test_filename = save_path / ('lidarcs_%s_infos_%s.pkl' % (sensor, test_split))

    print('------------------------Start to generate data infos------------------------')

    dataset.set_split(train_split)
    lidarcs_infos_train = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, num_features=num_features
    )
    with open(train_filename, 'wb') as f:
        pickle.dump(lidarcs_infos_train, f)
    print('lidarcs %s info train file is saved to %s' % (sensor, train_filename))

    dataset.set_split(val_split)
    lidarcs_infos_val = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, num_features=num_features
    )
    with open(val_filename, 'wb') as f:
        pickle.dump(lidarcs_infos_val, f)
    print('lidarcs %s info val file is saved to %s' % (sensor, val_filename))

    dataset.set_split(test_split)
    lidarcs_infos_test = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, num_features=num_features
    )
    with open(test_filename, 'wb') as f:
        pickle.dump(lidarcs_infos_test, f)
    print('lidarcs %s info test file is saved to %s' % (sensor, test_filename))

    print('------------------------Start create groundtruth database for data augmentation------------------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(sensor, train_filename, split=train_split)
    print('------------------------Data preparation done------------------------')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_kitti_infos', help='')
    args = parser.parse_args()

    if args.func == 'create_lidarcs_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict

        dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        # sensors = ['Livox', 'VLD-128', 'VLD-32']
        # sensors = ['Livox', 'VLD-128', 'VLD-64', 'VLD-32']
        sensors = ['VLD-64', 'VLD-32']
        for sensor in sensors:
            print('Processing data of sensor: ', sensor)
            create_lidarcs_infos(
                sensor = sensor,
                dataset_cfg=dataset_cfg,
                class_names=['Car', 'Pedestrian', 'Cyclist'],
                data_path=ROOT_DIR / 'data' / 'lidarcs' / sensor,
                save_path=ROOT_DIR / 'data' / 'lidarcs' /sensor,
            )
    
    
    # import sys

    # print('top')
    # print(sys.argv.__len__(), sys.argv[0], sys.argv[1])

    # if sys.argv.__len__() > 1 and sys.argv[1] == 'create_lidarcs_infos':
    #     import yaml
    #     from pathlib import Path
    #     from easydict import EasyDict
    #     print('enter')

    #     dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
    #     print('done load')
    #     ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
    #     sensors = ['Livox', 'VLD-128']
    #     for sensor in sensors:
    #         print('sensor here')
    #         create_lidarcs_infos(
    #             dataset_cfg=dataset_cfg,
    #             class_names=['Car', 'Pedestrian', 'Cyclist'],
    #             data_path=ROOT_DIR / 'data' / 'lidarcs' / sensor,
    #             save_path=ROOT_DIR / 'data' / 'lidarcs' /sensor,
    #         )
