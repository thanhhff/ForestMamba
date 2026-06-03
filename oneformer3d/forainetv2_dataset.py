from os import path as osp
import numpy as np
import random

from mmdet3d.datasets.scannet_dataset import ScanNetSegDataset, ScanNetDataset
from mmdet3d.registry import DATASETS


@DATASETS.register_module()
class ForAINetV2SegDataset_(ScanNetDataset):
    """We just add super_pts_path."""
    METAINFO = {
        'classes':
        ('ground','wood','leaf'),
        'palette': [[0, 255, 0],[0, 0, 255], [0, 255, 255]],
        'seg_valid_class_ids':
        (0, 1, 2),
        'seg_all_class_ids':
        (0, 1, 2)  # possibly with 'stair' class
    }

    def get_scene_idxs(self, *args, **kwargs):
        """Compute scene_idxs with dataset-balanced sampling.

        Groups samples by dataset source (prefix of lidar_path) and
        oversamples minority datasets so each dataset contributes equally
        per epoch. The number of samples per dataset equals the size of
        the largest dataset group.
        """
        groups = {}
        for i, info in enumerate(self.data_list):
            lidar_path = info['lidar_points']['lidar_path']
            prefix = lidar_path.split('_')[0]  # e.g. BlueCat, NIBIO, NIBIO2, SCION
            # Merge NIBIO and NIBIO2 as they belong to the same dataset family
            if prefix == 'NIBIO2':
                prefix = 'NIBIO'
            groups.setdefault(prefix, []).append(i)

        balanced_idxs = []
        for prefix, idxs in groups.items():
            n = 5 if len(idxs) < 5 else len(idxs)
            if len(idxs) >= n:
                sampled = random.sample(idxs, n)
            else:
                sampled = (idxs * (n // len(idxs) + 1))[:n]
            balanced_idxs.extend(sampled)

        return np.array(balanced_idxs, dtype=np.int32)

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        #info['super_pts_path'] = osp.join(
        #    self.data_prefix.get('sp_pts_mask', ''), info['super_pts_path'])

        info = super().parse_data_info(info)

        return info