import torch
import torch.nn.functional as F
import spconv.pytorch as spconv
from torch_scatter import scatter_mean, scatter_add
import MinkowskiEngine as ME

from mmdet3d.registry import MODELS
from mmdet3d.structures import PointData
from mmdet3d.models import Base3DDetector
from .mask_matrix_nms import mask_matrix_nms
import open3d as o3d
import os
import numpy as np
from tools.base_modules import Seq, MLP, FastBatchNorm1d
from .panoptic_losses import offset_loss, discriminative_loss, FastFocalLoss
from torch_cluster import fps, knn as cluster_knn
import re
import math
import collections 
import sys, datetime

# This is the full refactored version using ThreadPoolExecutor for parallel region inference
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from plyfile import PlyData, PlyElement

import contextlib, time

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1

class ScanNetOneFormer3DMixin:
    """Class contains common methods for ScanNet and ScanNet200."""

    def predict_by_feat(self, out, superpoints):
        """Predict instance, semantic, and panoptic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `sem_preds` of shape (n_queries, n_semantic_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
        
        Returns:
            List[PointData]: of len 1 with `pts_semantic_mask`,
                `pts_instance_mask`, `instance_labels`, `instance_scores`.
        """
        inst_res = self.predict_by_feat_instance(
            out, superpoints, self.test_cfg.inst_score_thr)
        sem_res = self.predict_by_feat_semantic(out, superpoints)
        pan_res = self.predict_by_feat_panoptic(out, superpoints)

        pts_semantic_mask = [sem_res.cpu().numpy(), pan_res[0].cpu().numpy()]
        pts_instance_mask = [inst_res[0].cpu().bool().numpy(),
                             pan_res[1].cpu().numpy()]
      
        return [
            PointData(
                pts_semantic_mask=pts_semantic_mask,
                pts_instance_mask=pts_instance_mask,
                instance_labels=inst_res[1].cpu().numpy(),
                instance_scores=inst_res[2].cpu().numpy())]
    
    def predict_by_feat_instance(self, out, superpoints, score_threshold):
        """Predict instance masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.
        
        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        cls_preds = out['cls_preds'][0]
        pred_masks = out['masks'][0]

        scores = F.softmax(cls_preds, dim=-1)[:, :-1]
        if out['scores'][0] is not None:
            scores *= out['scores'][0]
        labels = torch.arange(
            self.num_classes,
            device=scores.device).unsqueeze(0).repeat(
                len(cls_preds), 1).flatten(0, 1)
        scores, topk_idx = scores.flatten(0, 1).topk(
            self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, self.num_classes, rounding_mode='floor')
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()

        if self.test_cfg.get('obj_normalization', None):
            mask_scores = (mask_pred_sigmoid * (mask_pred > 0)).sum(1) / \
                ((mask_pred > 0).sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel=kernel)

        mask_pred_sigmoid = mask_pred_sigmoid[:, superpoints]
        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr

        # score_thr
        score_mask = scores > score_threshold
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return mask_pred, labels, scores

    def predict_by_feat_semantic(self, out, superpoints, classes=None):
        """Predict semantic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `sem_preds` of shape (n_queries, n_semantic_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
            classes (List[int] or None): semantic (stuff) class ids.
        
        Returns:
            Tensor: semantic preds of shape
                (n_raw_points, n_semantic_classe + 1),
        """
        if classes is None:
            classes = list(range(out['sem_preds'][0].shape[1] - 1))
        return out['sem_preds'][0][:, classes].argmax(dim=1)[superpoints]

    def predict_by_feat_panoptic(self, out, superpoints):
        """Predict panoptic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `sem_preds` of shape (n_queries, n_semantic_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
        
        Returns:
            Tuple:
                Tensor: semantic mask of shape (n_raw_points,),
                Tensor: instance mask of shape (n_raw_points,).
        """
        sem_map = self.predict_by_feat_semantic(
            out, superpoints, self.test_cfg.stuff_classes)
        mask_pred, labels, scores  = self.predict_by_feat_instance(
            out, superpoints, self.test_cfg.pan_score_thr)
        if mask_pred.shape[0] == 0:
            return sem_map, sem_map

        scores, idxs = scores.sort()
        labels = labels[idxs]
        mask_pred = mask_pred[idxs]

        n_stuff_classes = len(self.test_cfg.stuff_classes)
        inst_idxs = torch.arange(
            n_stuff_classes, 
            mask_pred.shape[0] + n_stuff_classes, 
            device=mask_pred.device).view(-1, 1)
        insts = inst_idxs * mask_pred
        things_inst_mask, idxs = insts.max(axis=0)
        things_sem_mask = labels[idxs] + n_stuff_classes

        inst_idxs, num_pts = things_inst_mask.unique(return_counts=True)
        for inst, pts in zip(inst_idxs, num_pts):
            if pts <= self.test_cfg.npoint_thr and inst != 0:
                things_inst_mask[things_inst_mask == inst] = 0

        things_sem_mask[things_inst_mask == 0] = 0
      
        sem_map[things_inst_mask != 0] = 0
        inst_map = sem_map.clone()
        inst_map += things_inst_mask
        sem_map += things_sem_mask
        return sem_map, inst_map
    
    def _select_queries(self, x, gt_instances):
        """Select queries for train pass.

        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, n_channels).
            gt_instances (List[InstanceData_]): of len batch_size.
                Ground truth which can contain `labels` of shape (n_gts_i,),
                `sp_masks` of shape (n_gts_i, n_points_i).

        Returns:
            Tuple:
                List[Tensor]: Queries of len batch_size, each queries of shape
                    (n_queries_i, n_channels).
                List[InstanceData_]: of len batch_size, each updated
                    with `query_masks` of shape (n_gts_i, n_queries_i).
        """
        queries = []
        for i in range(len(x)):
            if self.query_thr < 1:
                n = (1 - self.query_thr) * torch.rand(1) + self.query_thr
                n = (n * len(x[i])).int()
                ids = torch.randperm(len(x[i]))[:n].to(x[i].device)
                queries.append(x[i][ids])
                gt_instances[i].query_masks = gt_instances[i].sp_masks[:, ids]
            else:
                queries.append(x[i])
                gt_instances[i].query_masks = gt_instances[i].sp_masks
        return queries, gt_instances


@MODELS.register_module()
class ScanNetOneFormer3D(ScanNetOneFormer3DMixin, Base3DDetector):
    r"""OneFormer3D for ScanNet dataset.

    Args:
        in_channels (int): Number of input channels.
        num_channels (int): NUmber of output channels.
        voxel_size (float): Voxel size.
        num_classes (int): Number of classes.
        min_spatial_shape (int): Minimal shape for spconv tensor.
        query_thr (float): We select >= query_thr * n_queries queries
            for training and all n_queries for testing.
        backbone (ConfigDict): Config dict of the backbone.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 num_channels,
                 voxel_size,
                 num_classes,
                 min_spatial_shape,
                 query_thr,
                 backbone=None,
                 decoder=None,
                 criterion=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super(Base3DDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.unet = MODELS.build(backbone)
        self.decoder = MODELS.build(decoder)
        self.criterion = MODELS.build(criterion)
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.min_spatial_shape = min_spatial_shape
        self.query_thr = query_thr
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers(in_channels, num_channels)
    
    def _init_layers(self, in_channels, num_channels):
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                num_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='subm1'))
        self.output_layer = spconv.SparseSequential(
            torch.nn.BatchNorm1d(num_channels, eps=1e-4, momentum=0.1),
            torch.nn.ReLU(inplace=True))

    def extract_feat(self, x, superpoints, inverse_mapping, batch_offsets):
        """Extract features from sparse tensor.

        Args:
            x (SparseTensor): Input sparse tensor of shape
                (n_points, in_channels).
            superpoints (Tensor): of shape (n_points,).
            inverse_mapping (Tesnor): of shape (n_points,).
            batch_offsets (List[int]): of len batch_size + 1.

        Returns:
            List[Tensor]: of len batch_size,
                each of shape (n_points_i, n_channels).
        """
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        x = scatter_mean(x.features[inverse_mapping], superpoints, dim=0)
        out = []
        for i in range(len(batch_offsets) - 1):
            out.append(x[batch_offsets[i]: batch_offsets[i + 1]])
        return out

    def collate(self, points, elastic_points=None):
        """Collate batch of points to sparse tensor.

        Args:
            points (List[Tensor]): Batch of points.
            quantization_mode (SparseTensorQuantizationMode): Minkowski
                quantization mode. We use random sample for training
                and unweighted average for inference.

        Returns:
            TensorField: Containing features and coordinates of a
                sparse tensor.
        """
        if elastic_points is None:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((p[:, :3] - p[:, :3].min(0)[0]) / self.voxel_size,
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                 for p in points])
        else:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((el_p - el_p.min(0)[0]),
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                 for el_p, p in zip(elastic_points, points)])
        
        spatial_shape = torch.clip(
            coordinates.max(0)[0][1:] + 1, self.min_spatial_shape)
        field = ME.TensorField(features=features, coordinates=coordinates)
        tensor = field.sparse()
        coordinates = tensor.coordinates
        features = tensor.features
        inverse_mapping = field.inverse_mapping(tensor.coordinate_map_key)

        return coordinates, features, inverse_mapping, spatial_shape

    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instances_3d` and `gt_sem_seg_3d`.
        Returns:
            dict: A dictionary of loss components.
        """
        batch_offsets = [0]
        superpoint_bias = 0
        sp_gt_instances = []
        sp_pts_masks = []
        for i in range(len(batch_data_samples)):
            gt_pts_seg = batch_data_samples[i].gt_pts_seg

            gt_pts_seg.sp_pts_mask += superpoint_bias
            superpoint_bias = gt_pts_seg.sp_pts_mask.max().item() + 1
            batch_offsets.append(superpoint_bias)

            sp_gt_instances.append(batch_data_samples[i].gt_instances_3d)
            sp_pts_masks.append(gt_pts_seg.sp_pts_mask)

        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'],
            batch_inputs_dict.get('elastic_coords', None))

        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))
        sp_pts_masks = torch.hstack(sp_pts_masks)
        x = self.extract_feat(
            x, sp_pts_masks, inverse_mapping, batch_offsets)
        queries, sp_gt_instances = self._select_queries(x, sp_gt_instances)
        x = self.decoder(x, queries)
        loss = self.criterion(x, sp_gt_instances)
        return loss
    
    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instance_3d` and `gt_sem_seg_3d`.
        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample contains 'pred_pts_seg'.
            And the `pred_pts_seg` contains following keys.
                - instance_scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - instance_labels (Tensor): Labels of instances, has a shape
                    (num_instances, )
                - pts_instance_mask (Tensor): Instance mask, has a shape
                    (num_points, num_instances) of type bool.
        """
        batch_offsets = [0]
        superpoint_bias = 0
        sp_pts_masks = []
        for i in range(len(batch_data_samples)):
            gt_pts_seg = batch_data_samples[i].gt_pts_seg
            gt_pts_seg.sp_pts_mask += superpoint_bias
            superpoint_bias = gt_pts_seg.sp_pts_mask.max().item() + 1
            batch_offsets.append(superpoint_bias)
            sp_pts_masks.append(gt_pts_seg.sp_pts_mask)

        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'])

        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))
        sp_pts_masks = torch.hstack(sp_pts_masks)
        x = self.extract_feat(
            x, sp_pts_masks, inverse_mapping, batch_offsets)
        x = self.decoder(x, x)

        results_list = self.predict_by_feat(x, sp_pts_masks)
        for i, data_sample in enumerate(batch_data_samples):
            data_sample.pred_pts_seg = results_list[i]
        #return batch_data_samples
        import os
        import numpy as np
        import open3d as o3d

        pred_pts_seg = batch_data_samples[0].pred_pts_seg
        instance_labels  = pred_pts_seg.instance_labels # tensor, (num_instance,)
        instance_scores = pred_pts_seg.instance_scores # tensor, (num_instance,)
        pts_instance_mask = pred_pts_seg.pts_instance_mask[0] # tensor, (num_instances, num_points)
        input_points = batch_inputs_dict["points"][0] # tensor, (num_points, xyzrgb)
        input_point_name = batch_data_samples[0].lidar_path.split('/')[-1].split('.')[0]

        def save_point_cloud(points, file_path):
            if isinstance(points, torch.Tensor):
                points = points.cpu().numpy()  # Convert tensor to NumPy array on the CPU
            points = np.asarray(points, dtype=np.float32)  # Ensure points are in float32 format
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(points[:, :3])
            pc.colors = o3d.utility.Vector3dVector(points[:, 3:])
            o3d.io.write_point_cloud(file_path, pc)

        def filter_and_save_instances(instance_labels, instance_scores, pts_instance_mask, input_points,input_point_name, threshold=0.2):

            base_dir = f"./work_dirs/{input_point_name}"
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)
            input_pc_path = os.path.join(base_dir, f"{input_point_name}.ply")
            save_point_cloud(input_points, input_pc_path)

            instance_count = {}
            for i in range(len(instance_scores)):
                if instance_scores[i] >= threshold:
                    label = instance_labels[i].item()
                    if label not in instance_count:
                        instance_count[label] = 0
                    instance_count[label] += 1
                    #print(pts_instance_mask[i])
                    instance_mask = pts_instance_mask[i].astype(bool)
                    instance_points = input_points[instance_mask]
                    instance_pc_path = os.path.join(base_dir, f"{input_point_name}_{label}_{instance_count[label]}.ply")
                    save_point_cloud(instance_points, instance_pc_path)

        filter_and_save_instances(instance_labels, instance_scores, pts_instance_mask, input_points, input_point_name)

        return batch_data_samples

@MODELS.register_module()
class ForAINetV2OneFormer3D(Base3DDetector):
    r"""for-instance dataset.

    Args:
        in_channels (int): Number of input channels.
        num_channels (int): NUmber of output channels.
        voxel_size (float): Voxel size.
        num_classes (int): Number of classes.
        min_spatial_shape (int): Minimal shape for spconv tensor.
        query_thr (float): We select >= query_thr * n_queries queries
            for training and all n_queries for testing.
        backbone (ConfigDict): Config dict of the backbone.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 num_channels,
                 voxel_size,
                 num_classes,
                 min_spatial_shape,
                 stuff_classes,
                 thing_cls,
                 backbone=None,
                 decoder=None,
                 criterion=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None,
                 radius = 16):
        super(Base3DDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.unet = MODELS.build(backbone)
        self.decoder = MODELS.build(decoder)
        self.criterion = MODELS.build(criterion)
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.min_spatial_shape = min_spatial_shape
        self.stuff_classes = stuff_classes
        self.thing_cls = thing_cls
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.radius = radius
        self._work_dir = None  # Initialized once on first predict call
        self._init_layers(in_channels, num_channels)

    def _init_layers(self, in_channels, num_channels):
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                num_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='subm1'))
        self.output_layer = spconv.SparseSequential(
            torch.nn.BatchNorm1d(num_channels, eps=1e-4, momentum=0.1),
            torch.nn.ReLU(inplace=True))

    def extract_feat(self, x):
        """Extract features from sparse tensor.

        Args:
            x (SparseTensor): Input sparse tensor of shape
                (n_points, in_channels).

        Returns:
            List[Tensor]: of len batch_size,
                each of shape (n_points_i, n_channels).
        """
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        out = []
        for i in x.indices[:, 0].unique():
            out.append(x.features[x.indices[:, 0] == i])
        return out

    def collate(self, points, elastic_points=None):
        """Collate batch of points to sparse tensor.

        Args:
            points (List[Tensor]): Batch of points.
            quantization_mode (SparseTensorQuantizationMode): Minkowski
                quantization mode. We use random sample for training
                and unweighted average for inference.

        Returns:
            TensorField: Containing features and coordinates of a
                sparse tensor.
        """
        if elastic_points is None:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((p[:, :3] - p[:, :3].min(0)[0]) / self.voxel_size,
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                 for p in points])
        else:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((el_p - el_p.min(0)[0]),
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                 for el_p, p in zip(elastic_points, points)])

        spatial_shape = torch.clip(
            coordinates.max(0)[0][1:] + 1, self.min_spatial_shape)
        field = ME.TensorField(features=features, coordinates=coordinates)
        tensor = field.sparse()
        coordinates = tensor.coordinates
        features = tensor.features
        inverse_mapping = field.inverse_mapping(tensor.coordinate_map_key)

        return coordinates, features, inverse_mapping, spatial_shape

    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instances_3d` and `gt_sem_seg_3d`.
        Returns:
            dict: A dictionary of loss components.
        """

        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'],
            batch_inputs_dict.get('elastic_coords', None))   
        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))  

        x = self.extract_feat(x)  

        x = self.decoder(x)  

        sp_gt_instances = []
        for i in range(len(batch_data_samples)):
            voxel_superpoints = inverse_mapping[coordinates[:, 0][ \
                                                        inverse_mapping] == i] 
            voxel_superpoints = torch.unique(voxel_superpoints,  
                                             return_inverse=True)[1]
            inst_mask = batch_data_samples[i].gt_pts_seg.pts_instance_mask 
            sem_mask = batch_data_samples[i].gt_pts_seg.pts_semantic_mask 
            assert voxel_superpoints.shape == inst_mask.shape

            batch_data_samples[i].gt_instances_3d.sp_sem_masks = \
                                self.get_gt_semantic_masks(sem_mask,
                                                            voxel_superpoints,
                                                            self.num_classes)  
            batch_data_samples[i].gt_instances_3d.sp_inst_masks = \
                                self.get_gt_inst_masks(inst_mask,
                                                       voxel_superpoints)    
            
            batch_data_samples[i].gt_instances_3d.labels_3d, batch_data_samples[i].gt_instances_3d.sp_inst_masks, batch_data_samples[i].gt_instances_3d.ratio_inspoint = \
                                self.filter_stuff_masks(batch_data_samples[i].gt_instances_3d, self.stuff_classes, batch_data_samples[i].gt_pts_seg.ratio_inspoint)

            sp_gt_instances.append(batch_data_samples[i].gt_instances_3d)  

        loss = self.criterion(x, sp_gt_instances)  #unified_criterion.py __call__
        return loss

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.
        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instance_3d` and `gt_sem_seg_3d`.
        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample contains 'pred_pts_seg'.
            And the `pred_pts_seg` contains following keys.
                - instance_scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - instance_labels (Tensor): Labels of instances, has a shape
                    (num_instances, )
                - pts_instance_mask (Tensor): Instance mask, has a shape
                    (num_points, num_instances) of type bool.
        """
        lidar_path = batch_data_samples[0].lidar_path
        base_name = os.path.basename(lidar_path)
        current_filename = os.path.splitext(base_name)[0]
        #if 'val' in lidar_path:
        if 'test' in lidar_path:
            import numpy as np
            from sklearn.neighbors import NearestNeighbors
            from plyfile import PlyData, PlyElement
            from tqdm import tqdm
            step_size = self.radius
            grid_size = 0.2
            num_points = 640000
            pts_semantic_gt = batch_data_samples[0].eval_ann_info['pts_semantic_mask']
            pts_instance_gt = batch_data_samples[0].eval_ann_info['pts_instance_mask']
            original_points = batch_inputs_dict['points'][0]
            regions = self.generate_cylindrical_regions(original_points, self.radius, step_size)
            all_pre_sem = [list() for _ in range(original_points.shape[0])]
            all_pre_ins = np.full(original_points.shape[0], -1)
            max_instance = 0

            global_instance_scores = np.zeros((original_points.shape[0],), dtype=float) 

            best_masks = []

            for region_idx, region in enumerate(tqdm(regions, desc=f"[{current_filename}] Processing regions")):
                region_mask = ((original_points[:, 0] - region[0]) ** 2 + (original_points[:, 1] - region[1]) ** 2) <= self.radius ** 2
                pc1 = original_points[region_mask]
                pc1_indices = torch.where(region_mask)[0]

                if len(pc1) == 0:
                    continue

                pc2, pc2_indices = self.grid_sample(pc1, pc1_indices, grid_size)
                if len(pc2) < num_points:
                    pc3 = pc2
                    pc3_indices = pc2_indices
                elif len(pc2) > num_points:
                    pc3, pc3_indices = self.points_random_sampling(pc2, pc2_indices, num_points)

                coordinates, features, inverse_mapping2, spatial_shape = self.collate([pc3])
                x = spconv.SparseConvTensor(features, coordinates, spatial_shape, len(batch_data_samples))
                x = self.extract_feat(x)
                x = self.decoder(x)
                results_list = self.predict_by_feat_test(x, inverse_mapping2, pc3)
                
                # Collect masks and their scores, process them immediately
                masks = results_list[0].pts_instance_mask[0]
                scores = results_list[0].instance_scores
                valid_scores_mask = scores > 0.6
                masks = masks[valid_scores_mask]
                scores = scores[valid_scores_mask]

                # Nearest neighbor mapping for masks to pc1
                for mask, score in zip(masks, scores):
                    mask_pc1 = self.nearest_neighbor_mapping(pc1, pc3, mask)
                    mask_points = pc1_indices[mask_pc1].cpu().numpy()

                    # Vectorized update of global instance mask and scores
                    update_mask = score > global_instance_scores[mask_points]
                    global_instance_scores[mask_points[update_mask]] = score
                    all_pre_ins[mask_points[update_mask]] = max_instance

                    if np.any(update_mask):
                        # Add the new mask
                        best_masks.append((mask_points, max_instance, score))

                    max_instance += 1

                cylinder_current_semantic_pre = self.nearest_neighbor_mapping(pc1, pc3, results_list[0].pts_semantic_mask[0])
                 
                originids = torch.where(region_mask)[0].cpu().numpy()  # Move to CPU before using np.where
                all_pre_sem = self.vote_semantic_labels(all_pre_sem, originids, cylinder_current_semantic_pre)

                originids = pc3_indices.cpu().numpy()  # Use pc3_indices for ground truth labels
                # Get gt labels for pc3
                pc3_semantic_gt = pts_semantic_gt[originids]
                pc3_instance_gt = pts_instance_gt[originids]
                # Save each pc3 to a separate .ply file
                #region_dir = f"/workspace/work_dirs/oneformer3d_radius20_e2039_test_bm2/{current_filename}/region_{region_idx}"
                #region_ply_path = os.path.join(region_dir, "pc_ins_sem.ply")
                #self.save_ply(pc3.cpu().numpy(), results_list[0].pts_semantic_mask[0], results_list[0].pts_instance_mask[1], region_ply_path, pc3_semantic_gt, pc3_instance_gt)


                # Save each mask to a separate .ply file
                #for mask_idx, (mask, score) in enumerate(zip(results_list[0].pts_instance_mask[0], results_list[0].instance_scores)):
                #    if score > 0.4:
                #        mask_points = pc3.cpu().numpy()[mask]
                #        mask_file_path = os.path.join(region_dir, f"mask{mask_idx}_{score:.2f}.ply")
                #        self.save_ply_2(mask_points, np.full(mask_points.shape[0], mask_idx), mask_file_path)


                # Save the intermediate results
                #region_ply_path = os.path.join(region_dir, "complete_ins_pre.ply")
                #self.save_ply_2_withscore(original_points.cpu().numpy(), all_pre_ins, region_ply_path, global_instance_scores)

            # Post-processing step
            final_semantic_labels = self.finalize_semantic_labels(all_pre_sem)
            ground_mask = (final_semantic_labels == 0)
            all_pre_ins[ground_mask] = -1

            # Remove instances with fewer than 10 points
            unique_instances, instance_counts = np.unique(all_pre_ins, return_counts=True)
            small_instances = unique_instances[instance_counts < 10]
            for instance in small_instances:
                all_pre_ins[all_pre_ins == instance] = -1

            # Remove replaced old masks
            unique_best_masks = []
            for mask_points, instance_id, score in best_masks:
                if np.any(all_pre_ins[mask_points] == instance_id):
                    unique_best_masks.append((mask_points, instance_id, score))

            # Save the best masks
            #best_mask_dir = f"/workspace/work_dirs/oneformer3d_radius16_qp300_e2675_test_bm1_austrian/{current_filename}/best_masks_before_block_merge"
            #os.makedirs(best_mask_dir, exist_ok=True)
            #for mask_points, instance_id, score in unique_best_masks:
            #    mask_file_path = os.path.join(best_mask_dir, f"best_mask_{instance_id}_{score}.ply")
            #    self.save_ply_2_withscore(original_points[mask_points], np.full(mask_points.shape[0], instance_id), mask_file_path, global_instance_scores[mask_points])

            # Merge masks
            #clean_all_pre_ins, merged_masks = self.merge_overlapping_instances(all_pre_ins, unique_best_masks)
            clean_all_pre_ins, merged_masks = self.merge_overlapping_instances_by_score(all_pre_ins, unique_best_masks)

            # Save the masks after block merge
            #best_mask_after_merge_dir = f"/workspace/work_dirs/oneformer3d_radius16_qp300_e2675_test_bm1_austrian/{current_filename}/best_masks_after_block_merge"
            #os.makedirs(best_mask_after_merge_dir, exist_ok=True)
            #for mask_points, instance_id, score in merged_masks:
            #    mask_file_path = os.path.join(best_mask_after_merge_dir, f"best_mask_{instance_id}_{score}.ply")
            #    self.save_ply_2_withscore(original_points[mask_points], np.full(mask_points.shape[0], instance_id), mask_file_path, global_instance_scores[mask_points])

            # Re-label instances to ensure continuous labeling
            unique_labels = np.unique(clean_all_pre_ins)
            unique_labels = unique_labels[unique_labels >= 0]  # Exclude background label (-1)
            relabel_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
            relabel_map[-1] = -1  # Keep background as -1
            clean_all_pre_ins = np.vectorize(relabel_map.get)(clean_all_pre_ins)

            # Save the final combined results
            if self._work_dir is None:
                _cfg_name = os.path.splitext(os.path.basename(sys.argv[1]))[0] if len(sys.argv) > 1 else "unknown_config"
                _ckpt     = sys.argv[2] if len(sys.argv) > 2 else ""
                _ep_m     = re.search(r'epoch[_\-](\d+)', os.path.basename(_ckpt))
                _ep_sfx   = f"_ep{_ep_m.group(1)}" if _ep_m else ""
                self._work_dir = f"./work_dirs/inference/{_cfg_name}{_ep_sfx}"
            region_path = f"{self._work_dir}/{current_filename}_final_results.ply"
            self.save_ply_withscore(original_points.cpu().numpy(), final_semantic_labels, clean_all_pre_ins, global_instance_scores, region_path, pts_semantic_gt, pts_instance_gt)
            
            for i, data_sample in enumerate(batch_data_samples):
                data_sample.pred_pts_seg = results_list[i]
                data_sample.pred_pts_seg['originids'] = originids
            return batch_data_samples
        else:
            coordinates, features, inverse_mapping, spatial_shape = self.collate(
                batch_inputs_dict['points'])
            x = spconv.SparseConvTensor(
                features, coordinates, spatial_shape, len(batch_data_samples))

            x = self.extract_feat(x)

            x = self.decoder(x)

            results_list = self.predict_by_feat(x, inverse_mapping)

            for i, data_sample in enumerate(batch_data_samples):
                data_sample.pred_pts_seg = results_list[i]

            return batch_data_samples

    def predict_by_feat(self, out, superpoints):
        """Predict instance, semantic, and panoptic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).

        Returns:
            List[PointData]: of len 1 with `pts_semantic_mask`,
                `pts_instance_mask`, `instance_labels`, `instance_scores`.
        """
        pred_masks = out['masks'][0]
        pred_scores = out['scores'][0]

        sem_res = self.pred_sem(pred_masks[-self.test_cfg.num_sem_cls:, :],
                                superpoints)
        
        inst_res = self.pred_inst_sem(pred_masks[:-self.test_cfg.num_sem_cls, :],
                                  pred_scores[:-self.test_cfg.num_sem_cls, :],
                                  superpoints, self.test_cfg.inst_score_thr, sem_res)
        pan_res = self.pred_pan(pred_masks, pred_scores,
                                superpoints, sem_res)

        pts_semantic_mask = [sem_res.cpu().numpy(), pan_res[0].cpu().numpy()]
        pts_instance_mask = [inst_res[0].cpu().bool().numpy(),
                             pan_res[1].cpu().numpy()]

        return [
            PointData(
                pts_semantic_mask=pts_semantic_mask,
                pts_instance_mask=pts_instance_mask,
                instance_labels=inst_res[1].cpu().numpy(),
                instance_scores=inst_res[2].cpu().numpy())]

    def predict_by_feat_test(self, out, superpoints, coordinates):
        """Predict instance, semantic, and panoptic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).

        Returns:
            List[PointData]: of len 1 with `pts_semantic_mask`,
                `pts_instance_mask`, `instance_labels`, `instance_scores`.
        """
        pred_masks = out['masks'][0]
        pred_scores = out['scores'][0]

        sem_res = self.pred_sem(pred_masks[-self.test_cfg.num_sem_cls:, :],
                                superpoints)
        
        # Calculate ground_z_max from coordinates of points classified as ground
        ground_points = coordinates[sem_res == 0]
        ground_z_max = ground_points[:, 2].max().item() if ground_points.size(0) > 0 else float('inf')

        
        inst_res = self.pred_inst_sem_test(pred_masks[:-self.test_cfg.num_sem_cls, :],
                                  pred_scores[:-self.test_cfg.num_sem_cls, :],
                                  superpoints, self.test_cfg.inst_score_thr, sem_res, coordinates, ground_z_max)
        pan_res = self.pred_pan_sem(pred_masks, pred_scores, 
                                superpoints, sem_res, coordinates, ground_z_max)

        pts_semantic_mask = [sem_res.cpu().numpy(), pan_res[0].cpu().numpy()]
        pts_instance_mask = [inst_res[0].cpu().bool().numpy(),
                             pan_res[1].cpu().numpy()]

        return [
            PointData(
                pts_semantic_mask=pts_semantic_mask,
                pts_instance_mask=pts_instance_mask,
                instance_labels=inst_res[1].cpu().numpy(),
                instance_scores=inst_res[2].cpu().numpy())]

    def pred_inst(self, pred_masks, pred_scores, 
                  superpoints, score_threshold):
        """Predict instance masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_queries, n_points).
            pred_scores (Tensor): of shape (n_queris, 1).
            pred_labels (Tensor): of shape (n_queries, n_instance_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.

        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        scores = pred_scores

        labels = torch.arange(
            1,
            device=scores.device).unsqueeze(0).repeat(
                self.decoder.num_queries - self.test_cfg.num_sem_cls,
                1).flatten(0, 1)
        
        scores, topk_idx = scores.flatten(0, 1).topk(
            self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, 1, rounding_mode='floor') 
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        if self.test_cfg.get('obj_normalization', None):
            mask_pred_thr = mask_pred_sigmoid > \
                self.test_cfg.obj_normalization_thr
            mask_scores = (mask_pred_sigmoid * mask_pred_thr).sum(1) / \
                (mask_pred_thr.sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel=kernel)

        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr
        mask_pred = mask_pred[:, superpoints]
        # score_thr
        score_mask = scores > score_threshold
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return mask_pred, labels, scores
    
    def pred_inst_sem(self, pred_masks, pred_scores,
                  superpoints, score_threshold, sem_res):
        """Predict instance masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_queries, n_points).
            pred_scores (Tensor): of shape (n_queris, 1).
            pred_labels (Tensor): of shape (n_queries, n_instance_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.

        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        scores = pred_scores

        labels = torch.arange(
            1,
            device=scores.device).unsqueeze(0).repeat(
                self.decoder.num_queries - self.test_cfg.num_sem_cls,
                1).flatten(0, 1)
        
        scores, topk_idx = scores.flatten(0, 1).topk(
            self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, 1, rounding_mode='floor') 
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        if self.test_cfg.get('obj_normalization', None):
            mask_pred_thr = mask_pred_sigmoid > \
                self.test_cfg.obj_normalization_thr
            mask_scores = (mask_pred_sigmoid * mask_pred_thr).sum(1) / \
                (mask_pred_thr.sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel=kernel)

        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr
        mask_pred = mask_pred[:, superpoints]

        # Loop through each mask
        # Ensure stuff_cls is a tensor and move it to the same device as mask_sem_res
        stuff_cls_tensor = torch.tensor(self.test_cfg.stuff_cls, device=sem_res.device)

        # Compute the binary mask for stuff_cls
        is_stuff = torch.isin(sem_res, stuff_cls_tensor).float()
        # Multiply mask_pred by the binary mask and sum along the columns
        mask_scores = (mask_pred * is_stuff).sum(dim=1)
        # Calculate the number of points in each mask
        num_points_in_mask = mask_pred.sum(dim=1)
        # Set scores to 0 where the majority of points are stuff_cls
        scores[mask_scores > (num_points_in_mask / 2)] = 0

        # score_thr
        score_mask = scores > score_threshold
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return mask_pred, labels, scores
    
    def pred_inst_sem_test(self, pred_masks, pred_scores,
                  superpoints, score_threshold, sem_res, coordinates, ground_z_max):
        """Predict instance masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_queries, n_points).
            pred_scores (Tensor): of shape (n_queris, 1).
            pred_labels (Tensor): of shape (n_queries, n_instance_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.

        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        scores = pred_scores

        labels = torch.arange(
            1,
            device=scores.device).unsqueeze(0).repeat(
                self.decoder.num_queries - self.test_cfg.num_sem_cls,
                1).flatten(0, 1)
        
        scores, topk_idx = scores.flatten(0, 1).topk(
            self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, 1, rounding_mode='floor') 
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        if self.test_cfg.get('obj_normalization', None):
            mask_pred_thr = mask_pred_sigmoid > \
                self.test_cfg.obj_normalization_thr
            mask_scores = (mask_pred_sigmoid * mask_pred_thr).sum(1) / \
                (mask_pred_thr.sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel=kernel)

        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr
        mask_pred = mask_pred[:, superpoints]

        # Loop through each mask
        # Ensure stuff_cls is a tensor and move it to the same device as mask_sem_res
        stuff_cls_tensor = torch.tensor(self.test_cfg.stuff_cls, device=sem_res.device)

        # Compute the binary mask for stuff_cls
        is_stuff = torch.isin(sem_res, stuff_cls_tensor).float()
        # Multiply mask_pred by the binary mask and sum along the columns
        mask_scores = (mask_pred * is_stuff).sum(dim=1)
        # Calculate the number of points in each mask
        num_points_in_mask = mask_pred.sum(dim=1)
        # Set scores to 0 where the majority of points are stuff_cls
        scores[mask_scores > (num_points_in_mask / 2)] = 0

        # Filter instances whose lowest voxel floats too far above the ground.
        # The margin is configurable: a large value (default 5 m) allows
        # mid-canopy suppressed trees while still discarding sky artefacts.
        z_margin = self.test_cfg.get('z_filter_margin', 5.0)
        for i in range(mask_pred.size(0)):
            mask = mask_pred[i]
            if mask.sum().item() == 0:
                scores[i] = 0
                continue
            z_values = coordinates[mask, 2]
            if z_values.numel() > 0 and z_values.min().item() > ground_z_max + z_margin:
                scores[i] = 0

        # score_thr
        score_mask = scores > score_threshold
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return mask_pred, labels, scores
   
    def pred_sem(self, pred_masks, superpoints):
        """Predict semantic masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_points, n_semantic_classes).
            superpoints (Tensor): of shape (n_raw_points,).        

        Returns:
            Tensor: semantic preds of shape
                (n_raw_points, 1).
        """
        mask_pred = pred_masks.sigmoid()
        mask_pred = mask_pred[:, superpoints]
        seg_map = mask_pred.argmax(0)
        return seg_map

    def pred_pan(self, pred_masks, pred_scores,
                 superpoints, sem_res):
        """Predict panoptic masks for a single scene.
        
        Args:
            pred_masks (Tensor): of shape (n_queries, n_points).
            pred_scores (Tensor): of shape (n_queris, 1).
            pred_labels (Tensor): of shape (n_queries, n_instance_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
        
        Returns:
            Tuple:
                Tensor: semantic mask of shape (n_raw_points,),
                Tensor: instance mask of shape (n_raw_points,).
        """
        stuff_cls = pred_masks.new_tensor(self.test_cfg.stuff_cls).long()
        sem_map = self.pred_sem(
            pred_masks[-self.test_cfg.num_sem_cls + stuff_cls, :], superpoints)
        sem_map_src_mapping = stuff_cls[sem_map]

        n_cls = self.test_cfg.num_sem_cls
        thr = self.test_cfg.pan_score_thr
        mask_pred, labels, scores = self.pred_inst_sem(
            pred_masks[:-n_cls, :], pred_scores[:-n_cls, :],
            superpoints, thr, sem_res)
        
        thing_idxs = torch.zeros_like(labels)
        for thing_cls in self.test_cfg.thing_cls:
            thing_idxs = thing_idxs.logical_or(labels == thing_cls)
        
        mask_pred = mask_pred[thing_idxs]
        scores = scores[thing_idxs]
        labels = labels[thing_idxs]

        if mask_pred.shape[0] == 0:
            return sem_map_src_mapping, sem_map

        scores, idxs = scores.sort()
        labels = labels[idxs]
        mask_pred = mask_pred[idxs]

        inst_idxs = torch.arange(
            1, mask_pred.shape[0]+1, device=mask_pred.device).view(-1, 1)
        insts = inst_idxs * mask_pred
        things_inst_mask, idxs = insts.max(axis=0)
        things_sem_mask = labels[idxs]+1

        inst_idxs, num_pts = things_inst_mask.unique(return_counts=True)
        for inst, pts in zip(inst_idxs, num_pts):
            if pts <= self.test_cfg.npoint_thr and inst != 0:
                things_inst_mask[things_inst_mask == inst] = 0

        things_inst_mask = torch.unique(
            things_inst_mask, return_inverse=True)[1]
        things_inst_mask[things_inst_mask != 0] += len(stuff_cls) - 1
        things_sem_mask[things_inst_mask == 0] = 0
      
        sem_map_src_mapping[things_inst_mask != 0] = 0
        sem_map[things_inst_mask != 0] = 0
        sem_map += things_inst_mask
        sem_map_src_mapping += things_sem_mask
        return sem_map_src_mapping, sem_map
    
    def pred_pan_sem(self, pred_masks, pred_scores,
                 superpoints, sem_res, coordinates, ground_z_max):
        """Predict panoptic masks for a single scene.
        
        Args:
            pred_masks (Tensor): of shape (n_queries, n_points).
            pred_scores (Tensor): of shape (n_queris, 1).
            pred_labels (Tensor): of shape (n_queries, n_instance_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
        
        Returns:
            Tuple:
                Tensor: semantic mask of shape (n_raw_points,),
                Tensor: instance mask of shape (n_raw_points,).
        """
        stuff_cls = pred_masks.new_tensor(self.test_cfg.stuff_cls).long()
        sem_map = self.pred_sem(
            pred_masks[-self.test_cfg.num_sem_cls + stuff_cls, :], superpoints)
        sem_map_src_mapping = stuff_cls[sem_map]

        n_cls = self.test_cfg.num_sem_cls
        thr = self.test_cfg.pan_score_thr
        mask_pred, labels, scores = self.pred_inst_sem_test(
            pred_masks[:-n_cls, :], pred_scores[:-n_cls, :],
            superpoints, thr, sem_res, coordinates, ground_z_max)
        
        thing_idxs = torch.zeros_like(labels)
        for thing_cls in self.test_cfg.thing_cls:
            thing_idxs = thing_idxs.logical_or(labels == thing_cls)
        
        mask_pred = mask_pred[thing_idxs]
        scores = scores[thing_idxs]
        labels = labels[thing_idxs]

        if mask_pred.shape[0] == 0:
            return sem_map_src_mapping, sem_map

        scores, idxs = scores.sort()
        labels = labels[idxs]
        mask_pred = mask_pred[idxs]

        inst_idxs = torch.arange(
            1, mask_pred.shape[0]+1, device=mask_pred.device).view(-1, 1)
        insts = inst_idxs * mask_pred
        things_inst_mask, idxs = insts.max(axis=0)
        things_sem_mask = labels[idxs]+1

        inst_idxs, num_pts = things_inst_mask.unique(return_counts=True)
        for inst, pts in zip(inst_idxs, num_pts):
            if pts <= self.test_cfg.npoint_thr and inst != 0:
                things_inst_mask[things_inst_mask == inst] = 0

        things_inst_mask = torch.unique(
            things_inst_mask, return_inverse=True)[1]
        things_inst_mask[things_inst_mask != 0] += len(stuff_cls) - 1
        things_sem_mask[things_inst_mask == 0] = 0
      
        sem_map_src_mapping[things_inst_mask != 0] = 0
        sem_map[things_inst_mask != 0] = 0
        sem_map += things_inst_mask
        sem_map_src_mapping += things_sem_mask
        return sem_map_src_mapping, sem_map

    @staticmethod
    def get_gt_semantic_masks(mask_src, sp_pts_mask, num_classes):    
        """Create ground truth semantic masks.
        
        Args:
            mask_src (Tensor): of shape (n_raw_points, 1).
            sp_pts_mask (Tensor): of shape (n_raw_points, 1).
            num_classes (Int): number of classes.
        
        Returns:
            sp_masks (Tensor): semantic mask of shape (num_classes, n_points).
        """

        # Convert mask_src to one-hot encoding
        mask = torch.nn.functional.one_hot(mask_src, num_classes=num_classes).float()

        # Aggregate class counts for each voxel
        sp_masks = scatter_add(mask, sp_pts_mask, dim=0)

        # Determine the class with the maximum count in each voxel
        sp_masks = sp_masks.argmax(dim=-1)

        # Convert the result back to one-hot encoding
        sp_masks = torch.nn.functional.one_hot(sp_masks, num_classes=num_classes).float()

        # Transpose to get the shape (num_classes, n_points)
        sp_masks = sp_masks.T

        # Ensure the output dimensions match the expected shape
        assert sp_masks.shape == (num_classes, sp_pts_mask.max().item() + 1)

        return sp_masks

    @staticmethod
    def get_gt_inst_masks(mask_src, sp_pts_mask):
        """Create ground truth instance masks.
        
        Args:
            mask_src (Tensor): of shape (n_raw_points, 1).
            sp_pts_mask (Tensor): of shape (n_raw_points, 1).
        
        Returns:
            sp_masks (Tensor): semantic mask of shape (n_points, num_inst_obj).
        """
        mask = mask_src.clone()
        if torch.sum(mask == -1) != 0:
            mask[mask == -1] = torch.max(mask) + 1
            mask = torch.nn.functional.one_hot(mask)[:, :-1]
        else:
            mask = torch.nn.functional.one_hot(mask)

        mask = mask.T
        sp_masks = scatter_mean(mask, sp_pts_mask, dim=-1)
        sp_masks = sp_masks > 0.5

        return sp_masks
    
    @staticmethod
    def filter_stuff_masks(batch_data_samples_i, stuff_classes, ratio_inspoint):
        labels_3d = batch_data_samples_i.labels_3d
        sp_inst_masks = batch_data_samples_i.sp_inst_masks

        stuff_classes_tensor = torch.tensor(stuff_classes, device=labels_3d.device)

        mask = torch.isin(labels_3d, stuff_classes_tensor)
        indices_to_keep = ~mask

        filtered_labels_3d = labels_3d[indices_to_keep]

        filtered_sp_inst_masks = sp_inst_masks[indices_to_keep]

        ratio_tensor = torch.zeros(len(labels_3d), device=labels_3d.device)
        for i, idx in enumerate(labels_3d):
            ratio_tensor[i] = ratio_inspoint[i]
        ratio_subset = ratio_tensor[indices_to_keep]

        return filtered_labels_3d, filtered_sp_inst_masks, ratio_subset
    
    @staticmethod
    def generate_cylindrical_regions(points, radius, step_size):
        x_coords = points[:, 0].cpu()  
        y_coords = points[:, 1].cpu() 

        x_min, x_max = x_coords.min().item(), x_coords.max().item()
        y_min, y_max = y_coords.min().item(), y_coords.max().item()

        regions = []
        x = x_min
        while x <= x_max:
            y = y_min
            while y <= y_max:
                regions.append((x, y))
                y += step_size
            x += step_size

        return regions

    @staticmethod
    def grid_sample(points, indices, grid_size):
        scaled_points = points / grid_size
        grid_points = torch.floor(scaled_points).int()
        
        # Use unique to find indices of each voxel
        unique_grid_points, inverse_indices = torch.unique(grid_points, return_inverse=True, dim=0)

        # Calculate the mean coordinates for each voxel
        unique_points = torch.zeros((len(unique_grid_points), points.size(1)), dtype=points.dtype, device=points.device)
        unique_indices = torch.zeros(len(unique_grid_points), dtype=indices.dtype, device=indices.device)
        for i in range(len(unique_grid_points)):
            mask = (inverse_indices == i)
            unique_points[i] = points[mask].mean(dim=0)
            unique_indices[i] = indices[mask][0]  # Just pick one of the indices in the voxel

        return unique_points, unique_indices

    @staticmethod
    def points_random_sampling(points, indices, num_points):
        choices = np.random.choice(len(points), num_points, replace=False)
        sampled_points = points[choices]
        sampled_indices = indices[choices]
        return sampled_points, sampled_indices

    @staticmethod
    def nearest_neighbor_mapping(pc1, pc3, predictions):
        from sklearn.neighbors import NearestNeighbors

        # Ensure pc3 is on CPU
        pc3_cpu = pc3.cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(pc3_cpu)
        distances, indices = nbrs.kneighbors(pc1.cpu().numpy())

        mapped_predictions = predictions[indices.squeeze()]

        return mapped_predictions

    @staticmethod
    def vote_semantic_labels(all_pre_sem, originids, cylinder_current_semantic_pre):
        for idx, originid in zip(originids, cylinder_current_semantic_pre):
            all_pre_sem[idx].append(originid.item())

        return all_pre_sem

    @staticmethod
    def region_merging(all_pre_ins, max_instance, pre_ins, originids):
        idx = np.argwhere(all_pre_ins[originids] != -1)  # has label
        idx2 = np.argwhere(all_pre_ins[originids] == -1)  # no label

        if len(idx) == 0:
            mask_valid = pre_ins != -1
            all_pre_ins[originids[mask_valid]] = pre_ins[mask_valid] + max_instance
            max_instance = max_instance + len(np.unique(pre_ins[mask_valid]))
        elif len(idx2) == 0:
            return all_pre_ins, max_instance
        else:
            new_label = pre_ins.reshape(-1)
            unique_labels = np.unique(new_label)
        
            # Ignore the background label (-1)
            unique_labels = unique_labels[unique_labels != -1]
            
            for ii_idx in unique_labels:
                new_label_ii_idx = originids[np.argwhere(new_label == ii_idx).reshape(-1)]
                
                new_has_old_idx = new_label_ii_idx[all_pre_ins[new_label_ii_idx] != -1]
                new_not_old_idx = new_label_ii_idx[all_pre_ins[new_label_ii_idx] == -1]

                if len(new_has_old_idx) == 0:
                    all_pre_ins[new_not_old_idx] = max_instance
                    max_instance += 1
                elif len(new_not_old_idx) == 0:
                    continue
                else:
                    old_labels_ii = all_pre_ins[new_has_old_idx]
                    un = np.unique(old_labels_ii)
                    max_iou_ii = 0
                    max_iou_ii_oldlabel = 0
                    for g in un:
                        idx_old_all = np.argwhere(all_pre_ins == g).reshape(-1)
                        inter_label_idx = np.intersect1d(idx_old_all, new_label_ii_idx)
                        iou1 = float(inter_label_idx.size) / float(idx_old_all.size)
                        iou2 = float(inter_label_idx.size) / float(new_label_ii_idx.size)
                        iou = max(iou1, iou2)

                        if iou > max_iou_ii:
                            max_iou_ii = iou
                            max_iou_ii_oldlabel = g

                    if max_iou_ii > 0.3:
                        all_pre_ins[new_not_old_idx] = max_iou_ii_oldlabel
                    else:
                        all_pre_ins[new_not_old_idx] = max_instance
                        max_instance += 1

        return all_pre_ins, max_instance

    @staticmethod
    def save_ply(points, semantic_pred, instance_pred, filename, semantic_gt=None, instance_gt=None):
        from plyfile import PlyData, PlyElement
        output_dir = os.path.dirname(filename)
        os.makedirs(output_dir, exist_ok=True)
        
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                ('semantic_pred', 'i4'), ('instance_pred', 'i4')]
        
        if semantic_gt is not None and instance_gt is not None:
            dtype += [('semantic_gt', 'i4'), ('instance_gt', 'i4')]
            vertex = np.array([tuple(points[i]) + (semantic_pred[i], instance_pred[i], semantic_gt[i], instance_gt[i]) for i in range(points.shape[0])],
                            dtype=dtype)
        else:
            vertex = np.array([tuple(points[i]) + (semantic_pred[i], instance_pred[i]) for i in range(points.shape[0])],
                            dtype=dtype)

        el = PlyElement.describe(vertex, 'vertex')
        PlyData([el], text=False).write(filename)
    
    @staticmethod
    def save_ply_2(points, instance_pred, filename):
        from plyfile import PlyData, PlyElement
        output_dir = os.path.dirname(filename)
        os.makedirs(output_dir, exist_ok=True)
        
        # Filter out points with instance_pred == -1
        valid_mask = instance_pred != -1
        valid_points = points[valid_mask]
        valid_instance_pred = instance_pred[valid_mask]
        
        # Define the dtype for the vertex elements
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('instance_pred', 'i4')]
        
        # Create an array of vertices
        vertex = np.array([tuple(valid_points[i]) + (valid_instance_pred[i],) for i in range(valid_points.shape[0])], dtype=dtype)

        # Describe the elements and save the ply file
        el = PlyElement.describe(vertex, 'vertex')
        PlyData([el], text=False).write(filename)

    @staticmethod
    def save_ply_2_withscore(points, instance_pred, filename, scores):
        from plyfile import PlyData, PlyElement
        output_dir = os.path.dirname(filename)
        os.makedirs(output_dir, exist_ok=True)
        
        # Filter out points with instance_pred == -1
        valid_mask = instance_pred != -1
        valid_points = points[valid_mask]
        valid_instance_pred = instance_pred[valid_mask]
        valid_scores = scores[valid_mask]
        
        # Define the dtype for the vertex elements
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('instance_pred', 'i4'), ('score', 'f4')]
        
        # Create an array of vertices
        vertex = np.array([tuple(valid_points[i]) + (valid_instance_pred[i], valid_scores[i]) for i in range(valid_points.shape[0])], dtype=dtype)

        # Describe the elements and save the ply file
        el = PlyElement.describe(vertex, 'vertex')
        PlyData([el], text=False).write(filename)

    @staticmethod
    def _log_scene_metrics(scene_name,
                           sem_pred, ins_pred, score,
                           sem_gt, ins_gt,
                           iou_th=0.5, tree_classes=(1, 2)):
        """Print per-scene metrics on the full merged prediction.

        Semantic mIoU: same confusion-matrix formula as seg_eval.
        Instance Prec/Rec/F1: greedy IoU matching at iou_th (diagnostic).
        Cov (MUCov): best-match IoU for each GT tree instance.
        """
        from mmdet3d.evaluation.functional.seg_eval import fast_hist, per_class_iou
        CLASS_NAMES = ['ground', 'wood', 'leaf']
        NUM_CLS = 3

        # ── Semantic mIoU ────────────────────────────────────────────────────
        hist = fast_hist(sem_pred.astype(np.int64), sem_gt.astype(np.int64), NUM_CLS)
        iou  = per_class_iou(hist)
        miou = float(np.nanmean(iou))

        # ── Tree instance masks ───────────────────────────────────────────────
        def get_tree_masks(sem, ins, scr=None):
            items = []
            for uid in np.unique(ins):
                if uid < 0:
                    continue
                mask = ins == uid
                vals, counts = np.unique(sem[mask], return_counts=True)
                if int(vals[np.argmax(counts)]) not in tree_classes:
                    continue
                conf = float(np.mean(scr[mask])) if scr is not None else 1.0
                items.append((mask, conf))
            return items

        pred_items = sorted(get_tree_masks(sem_pred, ins_pred, score),
                            key=lambda x: x[1], reverse=True)
        gt_items   = get_tree_masks(sem_gt, ins_gt)
        gt_masks   = [m for m, _ in gt_items]

        # ── Greedy Prec / Rec / F1 ───────────────────────────────────────────
        matched_gt = set()
        tp = 0
        for pred_mask, _ in pred_items:
            best_iou, best_gi = 0.0, -1
            for gi, gm in enumerate(gt_masks):
                if gi in matched_gt:
                    continue
                inter = np.sum(pred_mask & gm)
                union = np.sum(pred_mask | gm)
                v = inter / union if union > 0 else 0.0
                if v > best_iou:
                    best_iou, best_gi = v, gi
            if best_iou >= iou_th and best_gi >= 0:
                tp += 1
                matched_gt.add(best_gi)

        fp   = len(pred_items) - tp
        fn   = len(gt_items)   - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        # ── Cov (MUCov) ───────────────────────────────────────────────────────
        if gt_masks and pred_items:
            pred_masks = [m for m, _ in pred_items]
            covs = []
            for gm in gt_masks:
                best = max(
                    (np.sum(pm & gm) / np.sum(pm | gm)
                     if np.sum(pm | gm) > 0 else 0.0)
                    for pm in pred_masks)
                covs.append(best)
            cov = float(np.mean(covs))
        else:
            cov = 0.0

        # ── Print ─────────────────────────────────────────────────────────────
        iou_strs = '  '.join(
            f'{CLASS_NAMES[c]}={iou[c]*100:.1f}' if not np.isnan(iou[c])
            else f'{CLASS_NAMES[c]}=N/A'
            for c in range(NUM_CLS))
        print(f'[Scene] {scene_name}  '
              f'mIoU={miou*100:.1f}  {iou_strs}  '
              f'Prec={prec*100:.1f}  Rec={rec*100:.1f}  F1={f1*100:.1f}  '
              f'Cov={cov*100:.1f}  '
              f'GT={len(gt_items)}  Pred={len(pred_items)}  TP={tp}')



    @staticmethod
    def save_ply_withscore(points, semantic_pred, instance_pred, scores, filename, semantic_gt=None, instance_gt=None):
        from plyfile import PlyData, PlyElement
        output_dir = os.path.dirname(filename)
        os.makedirs(output_dir, exist_ok=True)
        
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                ('semantic_pred', 'i4'), ('instance_pred', 'i4'), ('score', 'f4')]
        
        if semantic_gt is not None and instance_gt is not None:
            dtype += [('semantic_gt', 'i4'), ('instance_gt', 'i4')]
            vertex = np.array([tuple(points[i]) + (semantic_pred[i], instance_pred[i], scores[i], semantic_gt[i], instance_gt[i]) for i in range(points.shape[0])],
                            dtype=dtype)
        else:
            vertex = np.array([tuple(points[i]) + (semantic_pred[i], instance_pred[i], scores[i]) for i in range(points.shape[0])],
                            dtype=dtype)

        el = PlyElement.describe(vertex, 'vertex')
        PlyData([el], text=False).write(filename)

    @staticmethod
    def finalize_semantic_labels(all_pre_sem):
        from collections import Counter
        final_semantic_labels = np.full(len(all_pre_sem), -1)
        for i, labels in enumerate(all_pre_sem):
            if labels:
                final_semantic_labels[i] = Counter(labels).most_common(1)[0][0]
        return final_semantic_labels
    
    @staticmethod
    def merge_overlapping_instances(all_pre_ins, best_masks, iou_threshold=0.6):
        """
        Merge overlapping instances based on IoU.

        Args:
            all_pre_ins (numpy.ndarray): Array containing instance labels for each point.
            best_masks (list): List of tuples, each containing (mask_points, instance_id).
            iou_threshold (float): IoU threshold for merging instances.

        Returns:
            numpy.ndarray: Array containing the merged instance labels for each point.
        """
        from scipy.sparse import csr_matrix

        # Create a sparse matrix for each mask using csr_matrix
        num_points = all_pre_ins.shape[0]
        num_masks = len(best_masks)
        data = []
        row_indices = []
        col_indices = []

        for idx, (mask_points, instance_id, score) in enumerate(best_masks):
            data.extend([1] * len(mask_points))
            row_indices.extend(mask_points)
            col_indices.extend([idx] * len(mask_points))

        mask_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(num_points, num_masks), dtype=np.float32)

        # Compute the IoU between masks
        intersection = mask_matrix.T @ mask_matrix
        mask_sizes = mask_matrix.sum(axis=0).A1
        union = mask_sizes[:, None] + mask_sizes - intersection

        # Ensure no division by zero
        union[union == 0] = 1

        iou = intersection / union

        print(f"Computed IoU matrix:\n{iou}")

        # Use Union-Find to manage merging of masks
        uf = UnionFind(num_masks)
        for i in range(num_masks):
            for j in range(i + 1, num_masks):
                if iou[i, j] > iou_threshold:
                    print(f"Merging instances {best_masks[i][1]} and {best_masks[j][1]} with IoU {iou[i, j]}")
                    uf.union(i, j)

        # Update masks dynamically after merging
        merged_masks = []
        merged_instance_labels = np.copy(all_pre_ins)

        for i in range(num_masks):
            root = uf.find(i)
            if root == i:  # If this is the root, create a new merged mask
                merged_points = np.unique(np.concatenate([best_masks[k][0] for k in range(num_masks) if uf.find(k) == i]))
                merged_instance_id = best_masks[i][1]  # Use the instance_id of the root
                merged_score = max([best_masks[k][2] for k in range(num_masks) if uf.find(k) == i])  # Take the max score
                merged_masks.append((merged_points, merged_instance_id, merged_score))

                # Update point-wise labels
                merged_instance_labels[merged_points] = merged_instance_id

        return merged_instance_labels, merged_masks

    @staticmethod
    def merge_overlapping_instances_by_score(all_pre_ins, best_masks, overlap_threshold=0.3):
        """
        Merge overlapping instances based on score and point overlap ratio.
        
        This method compares each mask's points with the union of two masks and determines 
        which mask to keep based on its points' proportion in the union.

        Args:
            all_pre_ins (numpy.ndarray): Array containing instance labels for each point.
            best_masks (list): List of tuples, each containing (mask_points, instance_id, score).
            overlap_threshold (float): Overlap threshold for merging instances based on mask proportion in union.

        Returns:
            numpy.ndarray: Array containing the merged instance labels for each point.
            list: List of merged masks after applying the score-based merging.
        """
        num_masks = len(best_masks)

        # Initialize all points as unassigned (-1) if necessary
        all_pre_ins = np.full(all_pre_ins.shape, -1, dtype=int)
        mask_kept = np.ones(num_masks, dtype=bool)  # Track which masks are kept

        for i in range(num_masks):
            if not mask_kept[i]:
                continue
            mask_i_points = set(best_masks[i][0])
            for j in range(i + 1, num_masks):
                if not mask_kept[j]:
                    continue
                mask_j_points = set(best_masks[j][0])

                # Calculate the intersection of the point sets
                intersection_points = mask_i_points & mask_j_points  # Intersection of the point sets
                if len(intersection_points) == 0:
                    continue  # No overlap, skip

                # Calculate the proportion of intersection relative to each mask
                mask1_ratio = len(intersection_points) / len(mask_i_points)  # intersection / mask1
                mask2_ratio = len(intersection_points) / len(mask_j_points)  # intersection / mask2

                # If either mask's proportion in the intersection is greater than the threshold, merge
                if mask1_ratio > overlap_threshold or mask2_ratio > overlap_threshold:
                    if best_masks[i][2] >= best_masks[j][2]:  # Keep the one with the higher score
                        mask_kept[j] = False  # Discard mask2
                    else:
                        mask_kept[i] = False  # Discard mask1
                        break  # If mask i is discarded, no need to compare with others


        # Filter the masks to keep only those that are not discarded
        masks_after_score_merge = [best_masks[i] for i in range(num_masks) if mask_kept[i]]

        # Update point-wise instance labels
        merged_instance_labels = np.copy(all_pre_ins)
        for mask_points, instance_id, _ in masks_after_score_merge:
            merged_instance_labels[mask_points] = instance_id

        return merged_instance_labels, masks_after_score_merge
    
@MODELS.register_module()
class ForAINetV2OneFormer3D_XAwarequery(Base3DDetector):
    r"""FOR-instance dataset.

    Args:
        in_channels (int): Number of input channels.
        num_channels (int): Number of output channels.
        voxel_size (float): Voxel size.
        num_classes (int): Number of classes.
        min_spatial_shape (int): Minimal shape for spconv tensor.
        query_thr (float): We select >= query_thr * n_queries queries
            for training and all n_queries for testing.
        backbone (ConfigDict): Config dict of the backbone.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 num_channels,
                 voxel_size,
                 num_classes,
                 min_spatial_shape,
                 stuff_classes,
                 thing_cls,
                 query_point_num=200,
                 backbone=None,
                 decoder=None,
                 criterion=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None,
                 prepare_epoch=None,
                 #prepare_epoch2=None,
                 radius = 16,
                 score_th = 0.4,
                 chunk = 20_000):
        super(Base3DDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.unet = MODELS.build(backbone)
        self.decoder = MODELS.build(decoder)
        self.criterion = MODELS.build(criterion)
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.min_spatial_shape = min_spatial_shape
        self.stuff_classes = stuff_classes
        self.thing_cls = thing_cls
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.prepare_epoch = prepare_epoch
        #self.prepare_epoch2 = prepare_epoch2
        self._init_layers(in_channels, num_channels)
        self.Embed = Seq().append(MLP([num_channels, num_channels], bias=False))
        self.Embed.append(torch.nn.Linear(num_channels, 5))
        self.query_point_num = query_point_num
        self.radius = radius
        self.score_th = score_th
        self.chunk = chunk
        self._output_path = None  # set lazily on first predict call
        self.BiSemantic = (
            Seq()
            .append(MLP([num_channels, num_channels], bias=False))
            .append(torch.nn.Linear(num_channels, 2))
            .append(torch.nn.LogSoftmax(dim=-1))
        )

    def _sample_queries(self, backbone_feats, embed_feats_tree, xyz_voxels,
                        tree_indices, voxel_instance_labels, max_queries):
        """Sample content query seeds from tree voxels using FPS on embeddings.

        This is the default (C1) strategy.  Subclasses override this method to
        implement alternative query-initialisation strategies (e.g. C2 CHM).

        Args:
            backbone_feats (Tensor): (N_voxels, C) backbone features of all voxels.
            embed_feats_tree (Tensor): (N_tree, D) embedding features of tree voxels.
            xyz_voxels (Tensor): (N_voxels, 3+) mean world-XYZ of all voxels.
            tree_indices (Tensor): (N_tree,) indices of tree voxels into xyz_voxels.
            voxel_instance_labels (Tensor or None): (N_voxels,) instance label per
                voxel. None during inference (labels are not needed).
            max_queries (int): Maximum number of queries to return.

        Returns:
            Tuple[Tensor, Tensor or None]:
                query_features of shape (K, C),
                query_instance_labels of shape (K,) or None when
                voxel_instance_labels is None.
        """
        self._n_chm_queries = 0   # base class: all queries are FPS (no CHM)
        if embed_feats_tree.size(0) == 0:
            feats  = backbone_feats[tree_indices]
            labels = voxel_instance_labels[tree_indices] \
                if voxel_instance_labels is not None else None
            return feats, labels
        device  = embed_feats_tree.device
        batch_t = torch.zeros(embed_feats_tree.size(0), dtype=torch.long, device=device)
        ratio   = min(max_queries / embed_feats_tree.size(0), 1.0)
        topk    = fps(embed_feats_tree, batch_t, ratio=float(ratio))
        selected = tree_indices[topk]
        feats    = backbone_feats[selected]
        labels   = voxel_instance_labels[selected] \
            if voxel_instance_labels is not None else None
        return feats, labels

    def _init_layers(self, in_channels, num_channels):
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                num_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='subm1'))
        self.output_layer = spconv.SparseSequential(
            torch.nn.BatchNorm1d(num_channels, eps=1e-4, momentum=0.1),
            torch.nn.ReLU(inplace=True))

    def extract_feat(self, x):
        """Extract features from sparse tensor.

        Args:
            x (SparseTensor): Input sparse tensor of shape
                (n_points, in_channels).

        Returns:
            List[Tensor]: of len batch_size,
                each of shape (n_points_i, n_channels).
        """
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        out = []
        for i in x.indices[:, 0].unique():
            out.append(x.features[x.indices[:, 0] == i])
        return out

    def collate(self, points, elastic_points=None):
        """Collate batch of points to sparse tensor.

        Args:
            points (List[Tensor]): Batch of points.
            quantization_mode (SparseTensorQuantizationMode): Minkowski
                quantization mode. We use random sample for training
                and unweighted average for inference.

        Returns:
            TensorField: Containing features and coordinates of a
                sparse tensor.
        """
        if elastic_points is None:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((p[:, :3] - p[:, :3].min(0)[0]) / self.voxel_size,
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                 for p in points])
        else:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((el_p - el_p.min(0)[0]),
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                 for el_p, p in zip(elastic_points, points)])

        spatial_shape = torch.clip(
            coordinates.max(0)[0][1:] + 1, self.min_spatial_shape)
        field = ME.TensorField(features=features, coordinates=coordinates)
        tensor = field.sparse()
        coordinates = tensor.coordinates
        features = tensor.features
        inverse_mapping = field.inverse_mapping(tensor.coordinate_map_key)

        return coordinates, features, inverse_mapping, spatial_shape

    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instances_3d` and `gt_sem_seg_3d`.
        Returns:
            dict: A dictionary of loss components.
        """

        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'],
            batch_inputs_dict.get('elastic_coords', None))    
        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))  

        x = self.extract_feat(x) 

        embed_logits = [self.Embed(y) for y in x]
        bi_semantic_logits = [self.BiSemantic(y) for y in x] 
        
        # Initialize cumulative losses
        total_discriminative_loss = 0
        total_semantic_loss_bi = 0 
        batch_size = len(batch_data_samples)

        for i in range(batch_size):
            # Get voxel indices for the current sample
            voxel_superpoints = inverse_mapping[coordinates[:, 0][inverse_mapping] == i]
            voxel_superpoints = torch.unique(voxel_superpoints, return_inverse=True)[1]
            
            pts_instance_mask = batch_data_samples[i].gt_pts_seg.pts_instance_mask  # Instance labels (per-point)
            instance_mask = batch_data_samples[i].gt_pts_seg.instance_mask  # Boolean mask for foreground points
            sem_mask = batch_data_samples[i].gt_pts_seg.pts_semantic_mask  # Semantic labels (per-point)

            device = pts_instance_mask.device

            # Filter out background points using instance_mask
            valid_instance_mask = instance_mask

            # First, compute voxel_instance_labels using the previously provided method
            voxel_instance_labels = self.get_voxel_instance_labels(
                pts_instance_mask[valid_instance_mask], 
                voxel_superpoints[valid_instance_mask]
            )
            

            #Find the valid voxels containing foreground points
            valid_voxel_indices = torch.unique(voxel_superpoints[valid_instance_mask])  # Unique voxel indices with valid points
            
            # Filter embed_logits based on valid voxels
            filtered_embed_logits = embed_logits[i][valid_voxel_indices]  # Select only the valid voxels
        
            # Use voxel_instance_labels for discriminative loss
            batch = torch.full_like(voxel_instance_labels, i)  # Simulating batch index for each voxel
            discriminative_losses = discriminative_loss(
                filtered_embed_logits,  # Using the filtered embedding logits
                voxel_instance_labels,
                batch,
                5
            )

            # Accumulate discriminative loss for this sample
            for loss_name, loss in discriminative_losses.items():
                total_discriminative_loss += loss if loss_name == "ins_loss" else 0

            # Use the precomputed bi_semantic_logits for the current sample
            bi_semantic_logit = bi_semantic_logits[i]

            # Sum the foreground (instance_mask) per voxel and determine whether the voxel is background/foreground
            instance_mask_tensor = torch.from_numpy(instance_mask)
            instance_mask = instance_mask_tensor.to(device)
            voxel_point_counts = scatter_add(torch.ones_like(instance_mask.float()), voxel_superpoints, dim=0)
            foreground_voxel_counts = scatter_add(instance_mask.float(), voxel_superpoints, dim=0)
            
            # If a voxel has more than half foreground points, consider it foreground
            bi_y = (foreground_voxel_counts / voxel_point_counts) > 0.5
            bi_y = bi_y.long()  # Convert to long for loss function compatibility

            # Calculate semantic binary cross-entropy loss over all voxels (background and foreground)
            semantic_loss_bi = torch.nn.functional.nll_loss(
                bi_semantic_logit, bi_y.to(torch.int64)  # Removed ignore_index
            )
            
            # Accumulate semantic loss for the batch
            total_semantic_loss_bi += semantic_loss_bi

        # Average the accumulated losses over the batch
        total_discriminative_loss /= batch_size
        total_semantic_loss_bi /= batch_size

        # Add to total loss
        loss_final = {
            'discriminative_loss': total_discriminative_loss,
            'semantic_loss_bi': total_semantic_loss_bi
        }

        queries = []
        queries_inslabel = []
        n_chm_list = []   # number of CHM queries per batch item (rest are FPS)

        if self.prepare_epoch:
            if kwargs['epoch'] > self.prepare_epoch:
                for i in range(batch_size):
                    voxel_superpoints = inverse_mapping[coordinates[:, 0][inverse_mapping] == i]
                    voxel_superpoints = torch.unique(voxel_superpoints, return_inverse=True)[1]
                    instance_mask = batch_data_samples[i].gt_pts_seg.instance_mask
                    valid_voxel_indices = torch.unique(voxel_superpoints[instance_mask])

                    # ── Non-differentiable selection (no gradient tracking needed) ──
                    with torch.no_grad():
                        if valid_voxel_indices.numel() < 10:
                            queries.append([])
                            queries_inslabel.append([])
                            n_chm_list.append(0)
                            continue

                        wood_class = 1
                        semantic_predictions_bi = torch.argmax(bi_semantic_logits[i], dim=1)
                        tree_indices = torch.where(semantic_predictions_bi == wood_class)[0]

                        # Mean world-XYZ per voxel (needed by CHM-guided sampling)
                        current_points = batch_inputs_dict['points'][i]
                        current_points_add = scatter_add(current_points, voxel_superpoints, dim=0)
                        voxel_counts = scatter_add(
                            torch.ones_like(current_points[:, 0].float()), voxel_superpoints, dim=0)
                        avg_points = current_points_add / voxel_counts.unsqueeze(-1).clamp(min=1)

                        # Per-voxel instance labels for query supervision
                        voxel_inst_labels = self.get_voxel_instance_labels(
                            batch_data_samples[i].gt_pts_seg.pts_instance_mask,
                            voxel_superpoints)

                    # ── Query sampling — called OUTSIDE no_grad so that learnable
                    # embeddings (e.g. chm_extra_queries) stay in the autograd graph.
                    # Backbone features are detached so gradients don't flow through
                    # the selection path (CHM / FPS are non-differentiable anyway).
                    query_feats, query_labels = self._sample_queries(
                        x[i].detach(),
                        embed_logits[i][tree_indices].detach(),
                        avg_points,
                        tree_indices,
                        voxel_inst_labels,
                        self.query_point_num)

                    queries.append(query_feats)
                    queries_inslabel.append(query_labels)
                    n_chm_list.append(getattr(self, '_n_chm_queries', 0))

                if all(len(q) == 0 for q in queries):
                    pass
                else:
                    # First check if the length of x and queries are the same
                    if any(len(q) == 0 for q in queries):

                        # Use list comprehension to filter out empty queries and save original indices
                        filtered_results = [
                            (x[i], queries[i], batch_data_samples[i], queries_inslabel[i], i)  # Keep the original index i
                            for i in range(len(queries))
                            if len(queries[i]) > 0  # Only keep non-empty queries
                        ]
                        # Unpack filtered results into separate lists
                        x, queries, batch_data_samples, queries_inslabel, original_indices = zip(*filtered_results)
                        # Convert the zipped result back to list format
                        x = list(x)
                        queries = list(queries)
                        batch_data_samples = list(batch_data_samples)
                        queries_inslabel = list(queries_inslabel)
                        original_indices = list(original_indices)  # Keep track of original indices
                    else:
                        original_indices = list(range(len(batch_data_samples)))

                    # Align n_chm_list to the (possibly filtered) batch ordering
                    n_chm_filtered = [n_chm_list[j] for j in original_indices]

                    x = self.decoder(x, queries)

                    sp_gt_instances = []
                    for i in range(len(batch_data_samples)):
                        voxel_superpoints = inverse_mapping[coordinates[:, 0][ \
                                                                    inverse_mapping] == original_indices[i]] #[326894]
                        voxel_superpoints = torch.unique(voxel_superpoints,  
                                                        return_inverse=True)[1]
                        inst_mask = batch_data_samples[i].gt_pts_seg.pts_instance_mask 
                        sem_mask = batch_data_samples[i].gt_pts_seg.pts_semantic_mask 
                        assert voxel_superpoints.shape == inst_mask.shape

                        batch_data_samples[i].gt_instances_3d.sp_sem_masks = \
                                            self.get_gt_semantic_masks(sem_mask,
                                                                        voxel_superpoints,
                                                                        self.num_classes)  
                        batch_data_samples[i].gt_instances_3d.sp_inst_masks = \
                                            self.get_gt_inst_masks(inst_mask,
                                                                voxel_superpoints) 
                        
                        # Save original per-instance class labels BEFORE filtering,
                        # so we can remap query_inslabel from original instance IDs
                        # to post-filter row indices (filter removes ground instances).
                        orig_labels = batch_data_samples[i].gt_instances_3d.labels_3d.clone()

                        batch_data_samples[i].gt_instances_3d.labels_3d, batch_data_samples[i].gt_instances_3d.sp_inst_masks, batch_data_samples[i].gt_instances_3d.ratio_inspoint = \
                                            self.filter_stuff_masks(batch_data_samples[i].gt_instances_3d, self.stuff_classes, batch_data_samples[i].gt_pts_seg.ratio_inspoint)

                        # Remap query_inslabel: original instance ID → row index in
                        # filtered sp_inst_masks (filter_stuff_masks shifts rows).
                        # Example: instance 0=ground (removed), 1=tree_a, 2=tree_b
                        #   query_inslabel=1 must become row 0 (tree_a in filtered masks)
                        stuff_t  = torch.tensor(self.stuff_classes, device=orig_labels.device)
                        kept_mask = ~torch.isin(orig_labels, stuff_t)          # bool (n_orig_inst,)
                        kept_ids  = torch.where(kept_mask)[0]                  # original IDs kept
                        remap     = torch.full((len(orig_labels),), -1,
                                               dtype=torch.long, device=orig_labels.device)
                        remap[kept_ids] = torch.arange(
                            len(kept_ids), device=orig_labels.device)

                        q_lbl = queries_inslabel[i]
                        if (q_lbl is not None
                                and isinstance(q_lbl, torch.Tensor)
                                and q_lbl.numel() > 0):
                            valid    = (q_lbl >= 0) & (q_lbl < len(orig_labels))
                            remapped = torch.full_like(q_lbl, -1)
                            remapped[valid] = remap[q_lbl[valid]]
                            queries_inslabel[i] = remapped

                        batch_data_samples[i].gt_instances_3d.query_inslabel = queries_inslabel[i]
                        batch_data_samples[i].gt_instances_3d.n_chm_queries  = torch.tensor([n_chm_filtered[i]])

                        sp_gt_instances.append(batch_data_samples[i].gt_instances_3d)  

                    loss = self.criterion(x, sp_gt_instances)
                    loss_final.update(loss)

        return loss_final

    #def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
    def predict_bm1orbm2(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.
        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instance_3d` and `gt_sem_seg_3d`.
        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample contains 'pred_pts_seg'.
            And the `pred_pts_seg` contains following keys.
                - instance_scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - instance_labels (Tensor): Labels of instances, has a shape
                    (num_instances, )
                - pts_instance_mask (Tensor): Instance mask, has a shape
                    (num_points, num_instances) of type bool.
        """
        lidar_path = batch_data_samples[0].lidar_path
        base_name = os.path.basename(lidar_path)
        current_filename = os.path.splitext(base_name)[0]
        #if 'val' in lidar_path:
        if 'test' in lidar_path:
            import numpy as np
            from sklearn.neighbors import NearestNeighbors
            from plyfile import PlyData, PlyElement
            from tqdm import tqdm
            step_size = self.radius
            grid_size = 0.2
            num_points = 640000
            pts_semantic_gt = batch_data_samples[0].eval_ann_info['pts_semantic_mask']
            pts_instance_gt = batch_data_samples[0].eval_ann_info['pts_instance_mask']
            original_points = batch_inputs_dict['points'][0]
            regions = self.generate_cylindrical_regions(original_points, self.radius, step_size)
            all_pre_sem = [list() for _ in range(original_points.shape[0])]
            all_pre_ins = np.full(original_points.shape[0], -1)
            max_instance = 0

            global_instance_scores = np.zeros((original_points.shape[0],), dtype=float) 

            best_masks = []

            all_instance_labels = set(np.unique(pts_instance_gt))
            # Initialize an empty set to store the covered instance labels
            covered_instance_labels_qp = set()

            output_path = "./work_dirs/oneformer3d_outputfolder"
            for region_idx, region in enumerate(tqdm(regions, desc=f"[{current_filename}] Processing regions")):
                region_mask = ((original_points[:, 0] - region[0]) ** 2 + (original_points[:, 1] - region[1]) ** 2) <= self.radius ** 2
                pc1 = original_points[region_mask]
                pc1_indices = torch.where(region_mask)[0]

                if len(pc1) == 0:
                    continue

                pc2, pc2_indices = self.grid_sample(pc1, pc1_indices, grid_size)
                if len(pc2) < num_points:
                    pc3 = pc2
                    pc3_indices = pc2_indices
                elif len(pc2) > num_points:
                    pc3, pc3_indices = self.points_random_sampling(pc2, pc2_indices, num_points)

                coordinates, features, inverse_mapping2, spatial_shape = self.collate([pc3])
                x = spconv.SparseConvTensor(features, coordinates, spatial_shape, len(batch_data_samples))
                x = self.extract_feat(x)

                embed_logits = self.Embed(x[0])
                bi_semantic_logits = self.BiSemantic(x[0]) 

                wood_class = 1
                semantic_predictions_bi = torch.argmax(bi_semantic_logits, dim=1)
                tree_indices = torch.where(semantic_predictions_bi == wood_class)[0]  #all voxel
                    
                if tree_indices.numel() > 0:
                    
                    # FPS from all tree points
                    batch_tensor_4 = torch.zeros(embed_logits[tree_indices].size(0), dtype=torch.long).to(embed_logits.device)  # Ensure batch_tensor on same device
                    topk_indices_4 = fps(embed_logits[tree_indices], batch_tensor_4, ratio=min(self.query_point_num / embed_logits[tree_indices].size(0), torch.tensor([1.0]).to(embed_logits.device)))
                    selected_indices_case4 = tree_indices[topk_indices_4]

                    # add content queries
                    queries = []
                    queries.append(x[0][selected_indices_case4])
                    
                    x = self.decoder(x, queries)
                    results_list = self.predict_by_feat_test(x, inverse_mapping2, pc3, selected_indices_case4)
                    
                    # Collect masks and their scores, process them immediately
                    masks = results_list[0].pts_instance_mask[0]
                    scores = results_list[0].instance_scores
                    valid_scores_mask = scores > 0.6
                    masks = masks[valid_scores_mask]
                    scores = scores[valid_scores_mask]

                    # Nearest neighbor mapping for masks to pc1
                    for mask, score in zip(masks, scores):
                        mask_pc1 = self.nearest_neighbor_mapping(pc1, pc3, mask)
                        mask_points = pc1_indices[mask_pc1].cpu().numpy()

                        # Vectorized update of global instance mask and scores
                        update_mask = score > global_instance_scores[mask_points]
                        global_instance_scores[mask_points[update_mask]] = score
                        all_pre_ins[mask_points[update_mask]] = max_instance

                        if np.any(update_mask):
                            # Add the new mask
                            best_masks.append((mask_points, max_instance, score))

                        max_instance += 1

                    cylinder_current_semantic_pre = self.nearest_neighbor_mapping(pc1, pc3, results_list[0].pts_semantic_mask[0])
                     
                    originids = torch.where(region_mask)[0].cpu().numpy()  # Move to CPU before using np.where
                    all_pre_sem = self.vote_semantic_labels(all_pre_sem, originids, cylinder_current_semantic_pre)

                    originids = pc3_indices.cpu().numpy()  # Use pc3_indices for ground truth labels
                    
                else:
                    projected_semantic_logits = bi_semantic_logits[inverse_mapping2]
                    semantic_predictions_pc3 = torch.argmax(projected_semantic_logits, dim=1)
                    cylinder_current_semantic_pre = self.nearest_neighbor_mapping(pc1, pc3, semantic_predictions_pc3)
                    
                    originids = torch.where(region_mask)[0].cpu().numpy()  # Move to CPU before using np.where
                    all_pre_sem = self.vote_semantic_labels(all_pre_sem, originids, cylinder_current_semantic_pre)   
                    
            # Post-processing step
            final_semantic_labels = self.finalize_semantic_labels(all_pre_sem)
            ground_mask = (final_semantic_labels == 0)
            all_pre_ins[ground_mask] = -1

            # Remove instances with fewer than 10 points
            unique_instances, instance_counts = np.unique(all_pre_ins, return_counts=True)
            small_instances = unique_instances[instance_counts < 10]
            for instance in small_instances:
                all_pre_ins[all_pre_ins == instance] = -1

            # Remove replaced old masks
            unique_best_masks = []
            for mask_points, instance_id, score in best_masks:
                if np.any(all_pre_ins[mask_points] == instance_id):
                    unique_best_masks.append((mask_points, instance_id, score))

            # Merge masks bm1/bm2
            #clean_all_pre_ins, merged_masks = self.merge_overlapping_instances(all_pre_ins, unique_best_masks)
            clean_all_pre_ins, merged_masks = self.merge_overlapping_instances_by_score(all_pre_ins, unique_best_masks)

            # Re-label instances to ensure continuous labeling
            unique_labels = np.unique(clean_all_pre_ins)
            unique_labels = unique_labels[unique_labels >= 0]  # Exclude background label (-1)
            relabel_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
            relabel_map[-1] = -1  # Keep background as -1
            clean_all_pre_ins = np.vectorize(relabel_map.get)(clean_all_pre_ins)

            # Save the final combined results
            region_path = os.path.join(output_path, f"{current_filename}_final_results.ply")
            self.save_ply_withscore(original_points.cpu().numpy(), final_semantic_labels, clean_all_pre_ins, global_instance_scores, region_path, pts_semantic_gt, pts_instance_gt)

            # ── Per-scene metrics on the full merged prediction ──────────────
            self._log_scene_metrics(
                current_filename,
                final_semantic_labels, clean_all_pre_ins, global_instance_scores,
                pts_semantic_gt, pts_instance_gt)

            for i, data_sample in enumerate(batch_data_samples):
                data_sample.pred_pts_seg = results_list[i]
                data_sample.pred_pts_seg['originids'] = originids
            return batch_data_samples
        else:
            coordinates, features, inverse_mapping, spatial_shape = self.collate(
                batch_inputs_dict['points'])
            x = spconv.SparseConvTensor(
                features, coordinates, spatial_shape, len(batch_data_samples))

            x = self.extract_feat(x)

            queries = []
            for i in range(len(x)):
                max_len = min(self.query_point_num, len(x[i]))
                queries.append(x[i][0:max_len])
            
            x = self.decoder(x, queries)

            results_list = self.predict_by_feat(x, inverse_mapping)

            for i, data_sample in enumerate(batch_data_samples):
                data_sample.pred_pts_seg = results_list[i]

            return batch_data_samples

    def _sliding_window_decoder_forward(self, x, query_feats, voxel_xyz):
        """Decoder call inside the sliding-window test loop.

        Subclasses that need extra arguments (e.g. spatial coordinates for
        LaSSM-based decoders) can override this method without duplicating
        the entire sliding-window logic.
        """
        return self.decoder(x, [query_feats])

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Sliding-cylinder inference over a full tile, or plain validation.

        Test path (lidar_path contains 'test'):
          1. Slide cylindrical windows across the tile.
          2. For each window: voxel/random-downsample -> sparse backbone ->
             binary-semantic head -> FPS-sample tree queries -> decoder ->
             per-instance masks + scores.
          3. Merge each window back into global buffers: semantic votes and
             a "best-score wins" instance assignment with mask bookkeeping.
          4. Post-process: resolve votes, drop ground/tiny instances, merge
             overlapping instances across windows, relabel densely, save ply.

        Validation path: single-shot backbone + decoder over the whole scene.
        """
        lidar_path = batch_data_samples[0].lidar_path
        current_filename = os.path.splitext(os.path.basename(lidar_path))[0]

        # ----- Validation path -----
        if 'test' not in lidar_path:
            coordinates, features, inverse_mapping, spatial_shape = self.collate(
                batch_inputs_dict['points'])
            x = spconv.SparseConvTensor(
                features, coordinates, spatial_shape, len(batch_data_samples))
            x = self.extract_feat(x)

            wood_class = 1
            queries = []
            for i in range(len(x)):
                embed_logits_i    = self.Embed(x[i])
                bi_sem_logits_i   = self.BiSemantic(x[i])
                tree_idx_i        = torch.where(
                    bi_sem_logits_i.argmax(dim=1) == wood_class)[0]

                # Voxel world-XYZ for this batch item
                vsp = inverse_mapping[coordinates[:, 0][inverse_mapping] == i]
                vsp = torch.unique(vsp, return_inverse=True)[1]
                pts_i   = batch_inputs_dict['points'][i].float()
                vox_xyz = scatter_add(pts_i, vsp, dim=0) / \
                    scatter_add(torch.ones_like(pts_i[:, 0]), vsp, dim=0
                                ).unsqueeze(-1).clamp(min=1)

                if tree_idx_i.numel() > 1:
                    q_feats, _ = self._sample_queries(
                        x[i], embed_logits_i[tree_idx_i], vox_xyz,
                        tree_idx_i, None, self.query_point_num)
                else:
                    # No tree voxels predicted: fall back to head-slicing
                    q_feats = x[i][:min(self.query_point_num, len(x[i]))]
                queries.append(q_feats)

            x = self.decoder(x, queries)
            results_list = self.predict_by_feat(x, inverse_mapping)

            for i, data_sample in enumerate(batch_data_samples):
                data_sample.pred_pts_seg = results_list[i]
            return batch_data_samples

        # ----- Test / sliding-window path -----
        # Hyper-parameters for this path.
        step_size      = self.radius / 4
        grid_size      = 0.2
        num_points     = 640_000
        num_cls        = self.test_cfg.num_sem_cls
        score_th1      = self.score_th            # per-mask keep threshold
        score_th2      = self.test_cfg.get('merge_overlap_thr', 0.5)
        edge_margin    = 0.5                      # drop masks within this of edge
        edge_thresh_sq = (self.radius - edge_margin) ** 2
        radius_sq      = self.radius ** 2
        wood_class     = 1

        original_points = batch_inputs_dict['points'][0]
        pts_semantic_gt = batch_data_samples[0].eval_ann_info['pts_semantic_mask']
        pts_instance_gt = batch_data_samples[0].eval_ann_info['pts_instance_mask']
        regions = self.generate_cylindrical_regions(
            original_points, self.radius, step_size)

        # Global buffers accumulated over all windows.
        N_total = original_points.shape[0]
        votes_counter          = np.zeros((N_total, num_cls), dtype=np.int16)
        all_pre_ins            = np.full(N_total, -1, dtype=np.int64)
        global_instance_scores = np.zeros(N_total, dtype=float)
        best_masks             = []   # list of (pts_glob, inst_id, score)
        max_instance           = 0

        # Auto-create output dir once per run.
        if self._output_path is None:
            _cfg_name = os.path.splitext(os.path.basename(sys.argv[1]))[0] if len(sys.argv) > 1 else "unknown_config"
            _ckpt     = sys.argv[2] if len(sys.argv) > 2 else ""
            _ep_m     = re.search(r'epoch[_\-](\d+)', os.path.basename(_ckpt))
            _ep_sfx   = f"_ep{_ep_m.group(1)}" if _ep_m else ""
            self._output_path = f"./work_dirs/inference/{_cfg_name}{_ep_sfx}"
            os.makedirs(self._output_path, exist_ok=True)
        output_path = self._output_path

        last_results, last_originids = None, None

        for region in tqdm(regions, desc=f"[{current_filename}] Processing regions"):
            # (1) Grab points inside this cylinder (squared distance avoids sqrt).
            dx = original_points[:, 0] - region[0]
            dy = original_points[:, 1] - region[1]
            region_mask = (dx * dx + dy * dy) <= radius_sq
            pc1_indices = torch.where(region_mask)[0]
            pc1 = original_points[region_mask]
            if len(pc1) == 0:
                continue

            # (2) Voxel-grid downsample, then cap point count by random sampling.
            pc2, pc2_indices = self.grid_sample(pc1, pc1_indices, grid_size)
            if len(pc2) > num_points:
                pc3, pc3_indices = self.points_random_sampling(
                    pc2, pc2_indices, num_points)
            else:
                pc3, pc3_indices = pc2, pc2_indices

            # (3) Sparse backbone forward -> embed + binary semantic logits.
            coordinates, features, inverse_mapping2, spatial_shape = \
                self.collate([pc3])
            x = spconv.SparseConvTensor(
                features, coordinates, spatial_shape, len(batch_data_samples))
            x = self.extract_feat(x)
            embed_logits       = self.Embed(x[0])
            bi_semantic_logits = self.BiSemantic(x[0])
            device             = pc3.device

            # Voxels predicted as "wood" (tree) are the query candidates.
            tree_indices = torch.where(
                bi_semantic_logits.argmax(dim=1) == wood_class)[0]

            # (4) pc1 -> pc3 nearest neighbor on GPU via torch_cluster.knn.
            # cluster_knn(x, y, k) returns (row, col): col[i] is the index in x
            # nearest to y[row[i]].  With k=1 row = arange(N_pc1) so col is the
            # direct NN index — no big intermediate matrix, stays on GPU.
            _assign    = cluster_knn(pc3.float(), pc1.float(), k=1)
            nn_idx_pc1 = _assign[1]   # (N_pc1,) index into pc3

            if tree_indices.numel() > 1:
                # (5) Compute mean world-XYZ per voxel, then sample queries.
                voxel_counts_inf = scatter_add(
                    torch.ones_like(pc3[:, 0].float()), inverse_mapping2, dim=0)
                voxel_xyz = scatter_add(pc3.float(), inverse_mapping2, dim=0) / \
                    voxel_counts_inf.unsqueeze(-1).clamp(min=1)

                # inference: voxel_instance_labels=None, labels output is ignored
                query_feats, _ = self._sample_queries(
                    x[0], embed_logits[tree_indices], voxel_xyz,
                    tree_indices, None, self.query_point_num)

                # (6) Decoder pass + per-window instance prediction.
                # query_feats already contains pooled/extra features (no index needed).
                x = self._sliding_window_decoder_forward(x, query_feats, voxel_xyz)
                results_list = self.predict_by_feat_test(
                    x, inverse_mapping2, pc3, query_feats)

                masks_np  = results_list[0].pts_instance_mask[0]  # (K,N_pc3) bool np
                scores_np = results_list[0].instance_scores       # (K,) np

                masks_t  = torch.as_tensor(
                    masks_np, device=device, dtype=torch.bool)
                scores_t = torch.as_tensor(scores_np, device=device)

                # (7) Filter kept masks:
                #     - score > threshold
                #     - no mask point within edge_margin of the cylinder edge
                #       (those belong to trees better covered by a neighbor window).
                # Vectorized across all K masks at once -- replaces the old Python
                # for-loop over masks.
                pc3_dist_sq = ((pc3[:, 0] - region[0]) ** 2 +
                               (pc3[:, 1] - region[1]) ** 2)
                edge_pts = pc3_dist_sq > edge_thresh_sq                  # (N_pc3,)
                touches_edge = (masks_t & edge_pts.unsqueeze(0)).any(1)  # (K,)
                keep = torch.where(
                    (scores_t > score_th1) & ~touches_edge)[0]

                if keep.numel():
                    masks_kept  = masks_t[keep]                   # (K',N_pc3)
                    scores_kept = scores_t[keep]                  # (K',)

                    # (8) Project kept masks onto pc1 via the NN index: each pc1
                    # point inherits the masks of its nearest pc3 voxel.
                    mk_bool = masks_kept[:, nn_idx_pc1]           # (K',N_pc1)
                    rows, cols = mk_bool.nonzero(as_tuple=True)
                    score_per_hit = scores_kept[rows]             # (nnz,)

                    # (9) For each pc1 point pick the highest-scoring kept mask.
                    N1 = pc1.shape[0]
                    best_score = torch.full((N1,), -1.0, device=device)
                    best_mid   = torch.full(
                        (N1,), -1, dtype=torch.long, device=device)
                    best_score.index_reduce_(
                        0, cols, score_per_hit, reduce='amax')
                    improved = score_per_hit == best_score[cols]
                    best_mid.index_put_(
                        (cols[improved],), rows[improved], accumulate=False)

                    # (10) Merge into globals: only overwrite a point if this
                    # window's score beats whatever a previous (overlapping)
                    # window wrote for it.
                    pts_glob   = pc1_indices.cpu().numpy()
                    new_scores = best_score.cpu().numpy()
                    better_pts = new_scores > global_instance_scores[pts_glob]

                    if better_pts.any():
                        best_mid_np     = best_mid.cpu().numpy()
                        scores_kept_np  = scores_kept.cpu().numpy()
                        global_instance_scores[pts_glob[better_pts]] = \
                            new_scores[better_pts]
                        all_pre_ins[pts_glob[better_pts]] = (
                            max_instance + best_mid_np[better_pts])

                        # Record the FULL pc1 point list for each mid that
                        # contributed at least one "better" point.
                        # Using mk_bool[mid] directly avoids re-scanning rows/cols.
                        for mid in np.unique(best_mid_np[better_pts]).tolist():
                            sel_pts = mk_bool[mid].nonzero(
                                as_tuple=True)[0].cpu().numpy()
                            best_masks.append((
                                pts_glob[sel_pts],
                                max_instance + int(mid),
                                float(scores_kept_np[int(mid)])))

                    max_instance += int(masks_kept.size(0))

                # (11) Semantic voting using the decoder's semantic head.
                sem_pred_pc3 = results_list[0].pts_semantic_mask[0]
                if isinstance(sem_pred_pc3, np.ndarray):
                    sem_pred_pc3 = torch.from_numpy(sem_pred_pc3).to(device)
                cyl_sem_pc1 = sem_pred_pc3[nn_idx_pc1]

                ids_np = pc1_indices.cpu().numpy()
                sem_np = cyl_sem_pc1.cpu().numpy().astype(int)
                np.add.at(votes_counter, (ids_np, sem_np), 1)

                last_results   = results_list
                last_originids = pc3_indices.cpu().numpy()

            else:
                # (11') Too few tree voxels: skip the decoder and vote with the
                # binary-semantic head only.
                sem_pred_pc3 = torch.argmax(
                    bi_semantic_logits[inverse_mapping2], dim=1)
                cyl_sem_pc1 = sem_pred_pc3[nn_idx_pc1]

                ids_np = pc1_indices.cpu().numpy()
                sem_np = cyl_sem_pc1.cpu().numpy().astype(int)
                np.add.at(votes_counter, (ids_np, sem_np), 1)

        # ----- Global post-processing -----
        # (A) Resolve per-point semantic label from accumulated votes.
        # argmax returns 0 for all-zero rows, which we explicitly mark as -1.
        final_semantic_labels = votes_counter.argmax(1)
        final_semantic_labels[votes_counter.sum(1) == 0] = -1

        # (B) Wipe instance labels on ground points.
        all_pre_ins[final_semantic_labels == 0] = -1

        # (C) Drop tiny instances (<10 points).
        uniq, cnt = np.unique(all_pre_ins, return_counts=True)
        to_kill = np.isin(all_pre_ins, uniq[(cnt < 10) & (uniq != -1)])
        all_pre_ins[to_kill] = -1

        # (D) Drop mask records whose instance id is no longer present.
        unique_best_masks = [
            (pts, iid, sc) for (pts, iid, sc) in best_masks
            if np.any(all_pre_ins[pts] == iid)
        ]

        # (E) Merge overlapping instances coming from different windows.
        clean_all_pre_ins, merged_masks, merged_instance_scores = \
            self.merge_overlapping_instances_by_score_speedup(
                all_pre_ins, unique_best_masks, overlap_threshold=score_th2)

        # (F) Relabel instance ids to a dense 1..M range (bg stays -1).
        # IDs start from 1 to match the GT convention (0 = background) so that
        # the evaluator's `ins_arr > 0` filter correctly captures all instances.
        # Array-LUT lookup is much faster than np.vectorize(dict.get).
        # Shift by +1 so bg (-1) maps to lut[0] = -1.
        max_id = int(clean_all_pre_ins.max()) if clean_all_pre_ins.size else -1
        lut = np.full(max_id + 2, -1, dtype=np.int64)
        pos_ids = np.unique(clean_all_pre_ins[clean_all_pre_ins >= 0])
        lut[pos_ids + 1] = np.arange(1, pos_ids.size + 1)
        clean_all_pre_ins = lut[clean_all_pre_ins + 1]

        # (G) Persist combined results.
        region_path = os.path.join(output_path, f"{current_filename}.ply")
        self.save_ply_withscore(
            original_points.cpu().numpy(), final_semantic_labels,
            clean_all_pre_ins, merged_instance_scores,
            region_path, pts_semantic_gt, pts_instance_gt)

        # Attach the last window's PointData (used by downstream eval hooks).
        if last_results is not None and len(last_results) == len(batch_data_samples):
            for i, data_sample in enumerate(batch_data_samples):
                data_sample.pred_pts_seg = last_results[i]
                data_sample.pred_pts_seg['originids'] = last_originids
        else:
            for data_sample in batch_data_samples:
                data_sample.pred_pts_seg = None
        return batch_data_samples

    def predict_by_feat(self, out, superpoints):
        """Predict instance, semantic, and panoptic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).

        Returns:
            List[PointData]: of len 1 with `pts_semantic_mask`,
                `pts_instance_mask`, `instance_labels`, `instance_scores`.
        """
        #pred_labels = out['cls_preds'][0]
        pred_masks = out['masks'][0]
        pred_scores = out['scores'][0]

        #inst_res = self.pred_inst(pred_masks[:-self.test_cfg.num_sem_cls, :],
        #                          pred_scores[:-self.test_cfg.num_sem_cls, :],
        #                          #pred_labels[:-self.test_cfg.num_sem_cls, :],
        #                          superpoints, self.test_cfg.inst_score_thr)
        sem_res = self.pred_sem(pred_masks[-self.test_cfg.num_sem_cls:, :],
                                superpoints)
        
        inst_res = self.pred_inst_sem(pred_masks[:-self.test_cfg.num_sem_cls, :],
                                  pred_scores[:-self.test_cfg.num_sem_cls, :],
                                  superpoints, self.test_cfg.inst_score_thr, sem_res)
        pan_res = self.pred_pan(pred_masks, pred_scores, #pred_labels,
                                superpoints, sem_res)

        pts_semantic_mask = [sem_res.cpu().numpy(), pan_res[0].cpu().numpy()]
        pts_instance_mask = [inst_res[0].cpu().bool().numpy(),
                             pan_res[1].cpu().numpy()]

        return [
            PointData(
                pts_semantic_mask=pts_semantic_mask,
                pts_instance_mask=pts_instance_mask,
                instance_labels=inst_res[1].cpu().numpy(),
                instance_scores=inst_res[2].cpu().numpy())]

    def predict_by_feat_test(self, out, superpoints, coordinates, queries):
        """Predict instance, semantic, and panoptic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).

        Returns:
            List[PointData]: of len 1 with `pts_semantic_mask`,
                `pts_instance_mask`, `instance_labels`, `instance_scores`.
        """
        #pred_labels = out['cls_preds'][0]
        pred_masks = out['masks'][0]
        pred_scores = out['scores'][0]

        #inst_res = self.pred_inst(pred_masks[:-self.test_cfg.num_sem_cls, :],
        #                          pred_scores[:-self.test_cfg.num_sem_cls, :],
        #                          #pred_labels[:-self.test_cfg.num_sem_cls, :],
        #                          superpoints, self.test_cfg.inst_score_thr)
        sem_res = self.pred_sem(pred_masks[-self.test_cfg.num_sem_cls:, :],
                                superpoints)
        
        # Calculate ground_z_max from coordinates of points classified as ground
        ground_points = coordinates[sem_res == 0]
        ground_z_max = ground_points[:, 2].max().item() if ground_points.size(0) > 0 else float('inf')

        
        inst_res = self.pred_inst_sem_test(pred_masks[:-self.test_cfg.num_sem_cls, :],
                                  pred_scores[:-self.test_cfg.num_sem_cls, :],
                                  superpoints, self.test_cfg.inst_score_thr, sem_res, coordinates, ground_z_max, queries)
        pan_res = self.pred_pan_sem(pred_masks, pred_scores, #pred_labels,
                                superpoints, sem_res, coordinates, ground_z_max, queries)

        pts_semantic_mask = [sem_res.cpu().numpy(), pan_res[0].cpu().numpy()]
        pts_instance_mask = [inst_res[0].cpu().bool().numpy(),
                             pan_res[1].cpu().numpy()]

        return [
            PointData(
                pts_semantic_mask=pts_semantic_mask,
                pts_instance_mask=pts_instance_mask,
                instance_labels=inst_res[1].cpu().numpy(),
                instance_scores=inst_res[2].cpu().numpy(),
                query_select_voxel_idx=inst_res[3].cpu().numpy(),
                query_select_voxel_idx2=pan_res[2].cpu().numpy())]

    def pred_inst(self, pred_masks, pred_scores, #pred_labels,
                  superpoints, score_threshold):
        """Predict instance masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_queries, n_points).
            pred_scores (Tensor): of shape (n_queris, 1).
            pred_labels (Tensor): of shape (n_queries, n_instance_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.

        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        #scores = F.softmax(pred_labels, dim=-1)[:, :-1]
        #scores *= pred_scores
        scores = pred_scores

        labels = torch.arange(
            1,
            device=scores.device).unsqueeze(0).repeat(
                self.decoder.num_queries - self.test_cfg.num_sem_cls,
                1).flatten(0, 1)
        
        scores, topk_idx = scores.flatten(0, 1).topk(
            self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, 1, rounding_mode='floor') #self.num_classes, rounding_mode='floor')
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        if self.test_cfg.get('obj_normalization', None):
            mask_pred_thr = mask_pred_sigmoid > \
                self.test_cfg.obj_normalization_thr
            mask_scores = (mask_pred_sigmoid * mask_pred_thr).sum(1) / \
                (mask_pred_thr.sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel=kernel)

        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr
        mask_pred = mask_pred[:, superpoints]
        # score_thr
        score_mask = scores > score_threshold
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return mask_pred, labels, scores
    
    def pred_inst_sem(self, pred_masks, pred_scores, #pred_labels,
                  superpoints, score_threshold, sem_res):
        """Predict instance masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_queries, n_points).
            pred_scores (Tensor): of shape (n_queris, 1).
            pred_labels (Tensor): of shape (n_queries, n_instance_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.

        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        #scores = F.softmax(pred_labels, dim=-1)[:, :-1]
        #scores *= pred_scores
        scores = pred_scores

        n_inst_queries = scores.shape[0]
        labels = torch.arange(
            1,
            device=scores.device).unsqueeze(0).repeat(
                n_inst_queries,
                1).flatten(0, 1)

        flat_scores = scores.flatten(0, 1)
        k = min(self.test_cfg.topk_insts, flat_scores.shape[0])
        scores, topk_idx = flat_scores.topk(k, sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, 1, rounding_mode='floor') #self.num_classes, rounding_mode='floor')
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        if self.test_cfg.get('obj_normalization', None):
            mask_pred_thr = mask_pred_sigmoid > \
                self.test_cfg.obj_normalization_thr
            mask_scores = (mask_pred_sigmoid * mask_pred_thr).sum(1) / \
                (mask_pred_thr.sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel=kernel)

        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr
        mask_pred = mask_pred[:, superpoints]

        # Loop through each mask
        # Ensure stuff_cls is a tensor and move it to the same device as mask_sem_res
        stuff_cls_tensor = torch.tensor(self.test_cfg.stuff_cls, device=sem_res.device)
        #for i in range(mask_pred.size(0)):
        #    mask = mask_pred[i]
        #    # Get the semantic categories of the points within the mask
        #    mask_sem_res = sem_res[mask == 1]
        #    # Check if the majority of the points' semantic categories belong to stuff_cls
        #    if torch.isin(mask_sem_res, stuff_cls_tensor).sum().item() > mask_sem_res.size(0) / 2:
        #        # If true, set the corresponding score to 0
        #        scores[i] = 0

        # Compute the binary mask for stuff_cls
        is_stuff = torch.isin(sem_res, stuff_cls_tensor).float()
        # Multiply mask_pred by the binary mask and sum along the columns
        mask_scores = (mask_pred * is_stuff).sum(dim=1)
        # Calculate the number of points in each mask
        num_points_in_mask = mask_pred.sum(dim=1)
        # Set scores to 0 where the majority of points are stuff_cls
        scores[mask_scores > (num_points_in_mask / 2)] = 0

        # score_thr
        score_mask = scores > score_threshold
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return mask_pred, labels, scores

    def pred_inst_sem_test(self, pred_masks, pred_scores, #pred_labels,
                  superpoints, score_threshold, sem_res, coordinates, ground_z_max, queries):
        """Predict instance masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_queries, n_points).
            pred_scores (Tensor): of shape (n_queris, 1).
            pred_labels (Tensor): of shape (n_queries, n_instance_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.

        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        #scores = F.softmax(pred_labels, dim=-1)[:, :-1]
        #scores *= pred_scores
        scores = pred_scores

        labels = torch.arange(
            1,
            device=scores.device).unsqueeze(0).repeat(
                queries.shape[0],
                1).flatten(0, 1)
        
        scores, topk_idx = scores.flatten(0, 1).topk(
            min(self.test_cfg.topk_insts, queries.shape[0]), sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, 1, rounding_mode='floor') #self.num_classes, rounding_mode='floor')
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()

        queries_select = queries[topk_idx]
        if self.test_cfg.get('obj_normalization', None):
            mask_pred_thr = mask_pred_sigmoid > \
                self.test_cfg.obj_normalization_thr
            mask_scores = (mask_pred_sigmoid * mask_pred_thr).sum(1) / \
                (mask_pred_thr.sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, keep_inds = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel=kernel)

        queries_select = queries_select[keep_inds]
        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr
        mask_pred = mask_pred[:, superpoints]

        # Loop through each mask
        # Ensure stuff_cls is a tensor and move it to the same device as mask_sem_res
        stuff_cls_tensor = torch.tensor(self.test_cfg.stuff_cls, device=sem_res.device)

        # Compute the binary mask for stuff_cls
        is_stuff = torch.isin(sem_res, stuff_cls_tensor).float()
        # Multiply mask_pred by the binary mask and sum along the columns
        mask_scores = (mask_pred * is_stuff).sum(dim=1)
        # Calculate the number of points in each mask
        num_points_in_mask = mask_pred.sum(dim=1)
        # Set scores to 0 where the majority of points are stuff_cls
        scores[mask_scores > (num_points_in_mask / 2)] = 0

        # Filter instances whose lowest voxel floats too far above the ground.
        # The margin is configurable: a large value (default 20 m) allows
        # mid-canopy suppressed trees while still discarding sky artefacts.
        z_margin = self.test_cfg.get('z_filter_margin', 5.0)
        for i in range(mask_pred.size(0)):
            mask = mask_pred[i]
            if mask.sum().item() == 0:
                scores[i] = 0
                continue
            z_values = coordinates[mask, 2]
            if z_values.numel() > 0 and z_values.min().item() > ground_z_max + z_margin:
                scores[i] = 0

        # score_thr
        score_mask = scores > score_threshold
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]
        queries_select = queries_select[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]
        queries_select = queries_select[npoint_mask]

        return mask_pred, labels, scores, queries_select
   
    def pred_sem(self, pred_masks, superpoints):
        """Predict semantic masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_points, n_semantic_classes).
            superpoints (Tensor): of shape (n_raw_points,).        

        Returns:
            Tensor: semantic preds of shape
                (n_raw_points, 1).
        """
        mask_pred = pred_masks.sigmoid()
        mask_pred = mask_pred[:, superpoints]
        seg_map = mask_pred.argmax(0)
        return seg_map

    def pred_pan(self, pred_masks, pred_scores, #pred_labels,
                 superpoints, sem_res):
        """Predict panoptic masks for a single scene.
        
        Args:
            pred_masks (Tensor): of shape (n_queries, n_points).
            pred_scores (Tensor): of shape (n_queris, 1).
            pred_labels (Tensor): of shape (n_queries, n_instance_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
        
        Returns:
            Tuple:
                Tensor: semantic mask of shape (n_raw_points,),
                Tensor: instance mask of shape (n_raw_points,).
        """
        stuff_cls = pred_masks.new_tensor(self.test_cfg.stuff_cls).long()
        sem_map = self.pred_sem(
            pred_masks[-self.test_cfg.num_sem_cls + stuff_cls, :], superpoints)
        sem_map_src_mapping = stuff_cls[sem_map]

        n_cls = self.test_cfg.num_sem_cls
        thr = self.test_cfg.pan_score_thr
        mask_pred, labels, scores = self.pred_inst_sem(
            pred_masks[:-n_cls, :], pred_scores[:-n_cls, :],
            #pred_labels[:-n_cls, :], superpoints, thr)
            superpoints, thr, sem_res)
        
        thing_idxs = torch.zeros_like(labels)
        for thing_cls in self.test_cfg.thing_cls:
            thing_idxs = thing_idxs.logical_or(labels == thing_cls)
        
        mask_pred = mask_pred[thing_idxs]
        scores = scores[thing_idxs]
        labels = labels[thing_idxs]

        if mask_pred.shape[0] == 0:
            return sem_map_src_mapping, sem_map

        scores, idxs = scores.sort()
        labels = labels[idxs]
        mask_pred = mask_pred[idxs]

        inst_idxs = torch.arange(
            1, mask_pred.shape[0]+1, device=mask_pred.device).view(-1, 1)
        insts = inst_idxs * mask_pred
        things_inst_mask, idxs = insts.max(axis=0)
        things_sem_mask = labels[idxs]+1

        inst_idxs, num_pts = things_inst_mask.unique(return_counts=True)
        for inst, pts in zip(inst_idxs, num_pts):
            if pts <= self.test_cfg.npoint_thr and inst != 0:
                things_inst_mask[things_inst_mask == inst] = 0

        things_inst_mask = torch.unique(
            things_inst_mask, return_inverse=True)[1]
        things_inst_mask[things_inst_mask != 0] += len(stuff_cls) - 1
        things_sem_mask[things_inst_mask == 0] = 0
      
        sem_map_src_mapping[things_inst_mask != 0] = 0
        sem_map[things_inst_mask != 0] = 0
        sem_map += things_inst_mask
        sem_map_src_mapping += things_sem_mask
        return sem_map_src_mapping, sem_map
    
    def pred_pan_sem(self, pred_masks, pred_scores, #pred_labels,
                 superpoints, sem_res, coordinates, ground_z_max, queries):
        """Predict panoptic masks for a single scene.
        
        Args:
            pred_masks (Tensor): of shape (n_queries, n_points).
            pred_scores (Tensor): of shape (n_queris, 1).
            pred_labels (Tensor): of shape (n_queries, n_instance_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
        
        Returns:
            Tuple:
                Tensor: semantic mask of shape (n_raw_points,),
                Tensor: instance mask of shape (n_raw_points,).
        """
        stuff_cls = pred_masks.new_tensor(self.test_cfg.stuff_cls).long()
        sem_map = self.pred_sem(
            pred_masks[-self.test_cfg.num_sem_cls + stuff_cls, :], superpoints)
        sem_map_src_mapping = stuff_cls[sem_map]

        n_cls = self.test_cfg.num_sem_cls
        thr = self.test_cfg.pan_score_thr
        mask_pred, labels, scores, queries_select = self.pred_inst_sem_test(
            pred_masks[:-n_cls, :], pred_scores[:-n_cls, :],
            #pred_labels[:-n_cls, :], superpoints, thr)
            superpoints, thr, sem_res, coordinates, ground_z_max, queries)
        
        thing_idxs = torch.zeros_like(labels)
        for thing_cls in self.test_cfg.thing_cls:
            thing_idxs = thing_idxs.logical_or(labels == thing_cls)
        
        mask_pred = mask_pred[thing_idxs]
        scores = scores[thing_idxs]
        labels = labels[thing_idxs]
        queries_select = queries_select[thing_idxs]

        if mask_pred.shape[0] == 0:
            return sem_map_src_mapping, sem_map, queries_select

        scores, idxs = scores.sort()
        labels = labels[idxs]
        mask_pred = mask_pred[idxs]
        queries_select = queries_select[idxs]

        inst_idxs = torch.arange(
            1, mask_pred.shape[0]+1, device=mask_pred.device).view(-1, 1)
        insts = inst_idxs * mask_pred
        things_inst_mask, idxs = insts.max(axis=0)
        things_sem_mask = labels[idxs]+1

        inst_idxs, num_pts = things_inst_mask.unique(return_counts=True)
        # Track which queries are kept
        queries_retained = torch.ones_like(queries_select, dtype=torch.bool)
        for inst, pts in zip(inst_idxs, num_pts):
            if pts <= self.test_cfg.npoint_thr and inst != 0:
                things_inst_mask[things_inst_mask == inst] = 0
                # Mark the corresponding query as removed
                queries_retained[inst-1] = False  # Note: inst-1 is used as inst_idxs starts from 1

        things_inst_mask = torch.unique(
            things_inst_mask, return_inverse=True)[1]
        things_inst_mask[things_inst_mask != 0] += len(stuff_cls) - 1
        things_sem_mask[things_inst_mask == 0] = 0
      
        sem_map_src_mapping[things_inst_mask != 0] = 0
        sem_map[things_inst_mask != 0] = 0
        sem_map += things_inst_mask
        sem_map_src_mapping += things_sem_mask

        # Return queries that were retained (those not deleted)
        queries_select = queries_select[queries_retained]
        return sem_map_src_mapping, sem_map, queries_select

    @staticmethod
    def get_voxel_instance_labels(pts_instance_mask, voxel_superpoints):
        """Aggregate instance labels for each voxel.
        
        Args:
            pts_instance_mask (Tensor): Instance labels for each point (n_raw_points,).
            voxel_superpoints (Tensor): Voxel indices for each point (n_raw_points,).
        
        Returns:
            voxel_instance_labels (Tensor): Aggregated instance labels for each voxel.
        """
        # Treat the background (-1) points as a new instance, for proper aggregation
        instance_mask_clone = pts_instance_mask.clone()
        background_label = None
        if torch.any(instance_mask_clone == -1):
            background_label = torch.max(instance_mask_clone) + 1
            instance_mask_clone[instance_mask_clone == -1] = background_label
        
        _, instance_mask_clone = torch.unique(instance_mask_clone, return_inverse=True, sorted=True)
        # Convert the instance labels to one-hot encoding
        one_hot_inst_mask = torch.nn.functional.one_hot(instance_mask_clone)

        _, inverse_indices = torch.unique(voxel_superpoints, return_inverse=True, sorted=True)

        # Use scatter_add to aggregate one-hot labels for each voxel
        voxel_instance_counts = scatter_add(one_hot_inst_mask.float(), inverse_indices, dim=0)
        
        # Determine the most frequent instance label in each voxel (argmax along the label dimension)
        voxel_instance_labels = torch.argmax(voxel_instance_counts, dim=-1)
        
        # Convert background label back to -1, if background_label exists
        if background_label is not None:
            voxel_instance_labels[voxel_instance_labels == background_label] = -1
        
        return voxel_instance_labels
    
    @staticmethod
    def get_voxel_vote_labels(vote_label, pts_instance_mask, voxel_superpoints, voxel_main_instance):
        """Aggregate vote labels (offsets) for each voxel based on the main instance.
        
        Args:
            vote_label (Tensor): Offset vectors for each point (n_raw_points, 3).
            pts_instance_mask (Tensor): Instance labels for each point (n_raw_points,).
            voxel_superpoints (Tensor): Voxel indices for each point (n_raw_points,).
            voxel_main_instance (Tensor): The main instance (majority) for each voxel.
        
        Returns:
            voxel_vote_labels (Tensor): Aggregated vote labels (offsets) for the main instance in each voxel.
        """
        _, voxel_superpoints = torch.unique(voxel_superpoints, return_inverse=True, sorted=True)
        _, pts_instance_mask = torch.unique(pts_instance_mask, return_inverse=True, sorted=True)

        
        # Create a unique key for (voxel_superpoints, pts_instance_mask) to distinguish different instances within each voxel
        combined_idx = voxel_superpoints * (pts_instance_mask.max() + 1) + pts_instance_mask
        
        # Sum vote_label offsets for each instance in each voxel
        vote_label_sum = scatter_add(vote_label, combined_idx, dim=0)
        
        # Count the number of points for each instance in each voxel
        instance_counts = scatter_add(torch.ones_like(pts_instance_mask.float()), combined_idx, dim=0)

        # Calculate the average vote_label for each instance in each voxel
        avg_vote_label = vote_label_sum / instance_counts.unsqueeze(-1).clamp(min=1)

        # Now map voxel_main_instance to the correct instance in each voxel
        # For each voxel, create a unique key for (voxel_superpoints, voxel_main_instance)
        voxel_main_idx = torch.arange(voxel_superpoints.max() + 1).to(voxel_main_instance.device) * (pts_instance_mask.max() + 1) + voxel_main_instance

        # Select the average vote label for the main instance in each voxel
        voxel_vote_labels = avg_vote_label[voxel_main_idx]

        return voxel_vote_labels

    @staticmethod
    def get_gt_semantic_masks(mask_src, sp_pts_mask, num_classes):    
        """Create ground truth semantic masks.
        
        Args:
            mask_src (Tensor): of shape (n_raw_points, 1).
            sp_pts_mask (Tensor): of shape (n_raw_points, 1).
            num_classes (Int): number of classes.
        
        Returns:
            sp_masks (Tensor): semantic mask of shape (num_classes, n_points).
        """

        # Convert mask_src to one-hot encoding
        mask = torch.nn.functional.one_hot(mask_src, num_classes=num_classes).float()

        # Aggregate class counts for each voxel
        sp_masks = scatter_add(mask, sp_pts_mask, dim=0)

        # Determine the class with the maximum count in each voxel
        sp_masks = sp_masks.argmax(dim=-1)

        # Convert the result back to one-hot encoding
        sp_masks = torch.nn.functional.one_hot(sp_masks, num_classes=num_classes).float()

        # Transpose to get the shape (num_classes, n_points)
        sp_masks = sp_masks.T

        # Ensure the output dimensions match the expected shape
        assert sp_masks.shape == (num_classes, sp_pts_mask.max().item() + 1)

        return sp_masks

    @staticmethod
    def get_gt_inst_masks(mask_src, sp_pts_mask):
        """Create ground truth instance masks.
        
        Args:
            mask_src (Tensor): of shape (n_raw_points, 1).
            sp_pts_mask (Tensor): of shape (n_raw_points, 1).
        
        Returns:
            sp_masks (Tensor): semantic mask of shape (n_points, num_inst_obj).
        """
        mask = mask_src.clone()
        if torch.sum(mask == -1) != 0:
            mask[mask == -1] = torch.max(mask) + 1
            mask = torch.nn.functional.one_hot(mask)[:, :-1]
        else:
            mask = torch.nn.functional.one_hot(mask)

        mask = mask.T
        sp_masks = scatter_mean(mask, sp_pts_mask, dim=-1)
        sp_masks = sp_masks > 0.5

        return sp_masks
    
    @staticmethod
    def filter_stuff_masks(batch_data_samples_i, stuff_classes, ratio_inspoint):
        labels_3d = batch_data_samples_i.labels_3d
        sp_inst_masks = batch_data_samples_i.sp_inst_masks

        stuff_classes_tensor = torch.tensor(stuff_classes, device=labels_3d.device)

        mask = torch.isin(labels_3d, stuff_classes_tensor)
        indices_to_keep = ~mask

        filtered_labels_3d = labels_3d[indices_to_keep]

        filtered_sp_inst_masks = sp_inst_masks[indices_to_keep]

        ratio_tensor = torch.zeros(len(labels_3d), device=labels_3d.device)
        for i, idx in enumerate(labels_3d):
            ratio_tensor[i] = ratio_inspoint[i]
        ratio_subset = ratio_tensor[indices_to_keep]

        return filtered_labels_3d, filtered_sp_inst_masks, ratio_subset
    
    @staticmethod
    def generate_cylindrical_regions(points, radius, step_size):
        x_coords = points[:, 0].cpu()  
        y_coords = points[:, 1].cpu() 

        x_min, x_max = x_coords.min().item(), x_coords.max().item()
        y_min, y_max = y_coords.min().item(), y_coords.max().item()

        regions = []
        x = x_min
        while x <= x_max:
            y = y_min
            while y <= y_max:
                regions.append((x, y))
                y += step_size
            x += step_size

        return regions

    @staticmethod
    def grid_sample_old(points, indices, grid_size):
        scaled_points = points / grid_size
        grid_points = torch.floor(scaled_points).int()
        
        # Use unique to find indices of each voxel
        unique_grid_points, inverse_indices = torch.unique(grid_points, return_inverse=True, dim=0)

        # Calculate the mean coordinates for each voxel
        unique_points = torch.zeros((len(unique_grid_points), points.size(1)), dtype=points.dtype, device=points.device)
        unique_indices = torch.zeros(len(unique_grid_points), dtype=indices.dtype, device=indices.device)
        for i in range(len(unique_grid_points)):
            mask = (inverse_indices == i)
            unique_points[i] = points[mask].mean(dim=0)
            unique_indices[i] = indices[mask][0]  # Just pick one of the indices in the voxel

        return unique_points, unique_indices

    @staticmethod
    def grid_sample(points: torch.Tensor,
                    indices: torch.Tensor,
                    grid_size: float):
        """
        Voxel‑downsample point cloud by averaging points in each voxel.

        Args
        ----
        points  : (N, 3)  xyz or xyzf  GPU tensor
        indices : (N,)    original indices (int64 / int32) GPU tensor
        grid_size : float voxel size

        Returns
        -------
        vox_points  : (M, 3)   averaged coords per voxel (same dtype/device)
        vox_indices : (M,)     one representative original index per voxel
        """
        # 1. voxel coordinate (int32)
        voxel = torch.floor(points / grid_size).to(torch.int32)          # (N,3)

        # 2. find unique voxels
        uniq, inverse = torch.unique(voxel, return_inverse=True, dim=0)   # (N,) inverse ∈ [0,M)

        M = uniq.size(0)

        # 3. scatter‑add coords  (sum / count → mean)
        ones   = torch.ones_like(inverse, dtype=points.dtype)             # (N,)

        sum_xyz = torch.zeros((M, points.size(1)), device=points.device, dtype=points.dtype)
        cnt_xyz = torch.zeros(M, device=points.device, dtype=points.dtype)

        sum_xyz.index_add_(0, inverse, points)        # Σ xyz
        cnt_xyz.index_add_(0, inverse, ones)          # Σ 1

        vox_points = sum_xyz / cnt_xyz.unsqueeze(1)   # mean

        # 4. pick a representative original index  
        vox_indices = torch.full((M,), -1, device=indices.device, dtype=indices.dtype)
        vox_indices.index_copy_(0, inverse, indices)

        return vox_points, vox_indices


    @staticmethod
    def points_random_sampling(points, indices, num_points):
        choices = np.random.choice(len(points), num_points, replace=False)
        sampled_points = points[choices]
        sampled_indices = indices[choices]
        return sampled_points, sampled_indices

    @staticmethod
    def nearest_neighbor_mapping(pc1 : torch.Tensor,
                             pc3 : torch.Tensor,
                             preds: torch.Tensor | np.ndarray,
                             chunk: int = 20_000) -> torch.Tensor:
        # GPU Tensor
        if isinstance(preds, np.ndarray):
            preds = torch.from_numpy(preds).to(pc3.device, non_blocking=True)

        if preds.ndim == 1:
            preds = preds.unsqueeze(0)            # (1,N3)

        K, N3  = preds.shape
        N1     = pc1.size(0)
        out    = torch.empty((K, N1), dtype=preds.dtype, device=preds.device)

        for s in range(0, N1, chunk):
            e   = min(s + chunk, N1)
            nn  = torch.cdist(pc1[s:e].float(), pc3.float()).argmin(dim=1)  # (c,)
            out[:, s:e] = preds[:, nn]     # (K,c)

        return out.squeeze(0) if K == 1 else out

    @staticmethod
    def nearest_neighbor_mapping_2(pc1: torch.Tensor,
                               pc3: torch.Tensor,
                               preds: torch.Tensor,
                               chunk: int = 20_000) -> torch.Tensor:

        if isinstance(preds, np.ndarray):                     # keep downstream happy
            preds = torch.as_tensor(preds, device=pc3.device)

        if preds.dim() == 1:                                  # → (1, N3)
            preds = preds.unsqueeze(0)

        K, N3 = preds.shape
        N1    = pc1.shape[0]
        out   = torch.empty((K, N1), dtype=preds.dtype, device=preds.device)

        for s in range(0, N1, chunk):
            e   = min(s + chunk, N1)
            # (c, 3) × (N3, 3) → (c, N3)
            d   = torch.cdist(pc1[s:e].float(), pc3.float())   # fits in GPU
            nn  = d.argmin(dim=1)                              # (c,)
            out[:, s:e] = preds[:, nn]                         # gather once

        return out.squeeze(0) if K == 1 else out


    @staticmethod
    def vote_semantic_labels(all_votes: list,
                              ids:    torch.Tensor | np.ndarray,
                              sem:    torch.Tensor | np.ndarray):
        """
        all_votes : List[List[int]] len == N_total
        ids, sem  : (N_pts,)
        """
        if torch.is_tensor(ids):
            ids = ids.cpu().numpy()
        if torch.is_tensor(sem):
            sem = sem.cpu().numpy()

        for idx, lab in zip(ids, sem):
            all_votes[int(idx)].append(int(lab))
        return all_votes

    @staticmethod
    def region_merging(all_pre_ins, max_instance, pre_ins, originids):
        idx = np.argwhere(all_pre_ins[originids] != -1)  # has label
        idx2 = np.argwhere(all_pre_ins[originids] == -1)  # no label

        if len(idx) == 0:
            mask_valid = pre_ins != -1
            all_pre_ins[originids[mask_valid]] = pre_ins[mask_valid] + max_instance
            max_instance = max_instance + len(np.unique(pre_ins[mask_valid]))
        elif len(idx2) == 0:
            return all_pre_ins, max_instance
        else:
            new_label = pre_ins.reshape(-1)
            unique_labels = np.unique(new_label)
        
            # Ignore the background label (-1)
            unique_labels = unique_labels[unique_labels != -1]
            
            for ii_idx in unique_labels:
            #for ii_idx in np.unique(new_label):
                new_label_ii_idx = originids[np.argwhere(new_label == ii_idx).reshape(-1)]
                #new_has_old_idx = new_label_ii_idx[np.argwhere(all_pre_ins[new_label_ii_idx] != -1)]
                #new_not_old_idx = new_label_ii_idx[np.argwhere(all_pre_ins[new_label_ii_idx] == -1)]

                new_has_old_idx = new_label_ii_idx[all_pre_ins[new_label_ii_idx] != -1]
                new_not_old_idx = new_label_ii_idx[all_pre_ins[new_label_ii_idx] == -1]

                #has_old_idx_mask = all_pre_ins[new_label_ii_idx] != -1
                #new_has_old_idx = new_label_ii_idx[has_old_idx_mask]
                #new_not_old_idx = new_label_ii_idx[~has_old_idx_mask]

                if len(new_has_old_idx) == 0:
                    all_pre_ins[new_not_old_idx] = max_instance
                    max_instance += 1
                elif len(new_not_old_idx) == 0:
                    continue
                else:
                    old_labels_ii = all_pre_ins[new_has_old_idx]
                    un = np.unique(old_labels_ii)
                    max_iou_ii = 0
                    max_iou_ii_oldlabel = 0
                    for g in un:
                        #idx_old_all = originids[np.argwhere(all_pre_ins[originids] == g).reshape(-1)]
                        idx_old_all = np.argwhere(all_pre_ins == g).reshape(-1)
                        inter_label_idx = np.intersect1d(idx_old_all, new_label_ii_idx)
                        #union_label_idx = np.union1d(idx_old_all, new_label_ii_idx)
                        iou1 = float(inter_label_idx.size) / float(idx_old_all.size)
                        iou2 = float(inter_label_idx.size) / float(new_label_ii_idx.size)
                        iou = max(iou1, iou2)
                        #iou = float(inter_label_idx.size) / float(union_label_idx.size)

                        if iou > max_iou_ii:
                            max_iou_ii = iou
                            max_iou_ii_oldlabel = g

                    if max_iou_ii > 0.3:
                        all_pre_ins[new_not_old_idx] = max_iou_ii_oldlabel
                    else:
                        all_pre_ins[new_not_old_idx] = max_instance
                        max_instance += 1

        return all_pre_ins, max_instance

    @staticmethod
    def save_ply(points, semantic_pred, instance_pred, filename, semantic_gt=None, instance_gt=None):
        from plyfile import PlyData, PlyElement
        output_dir = os.path.dirname(filename)
        os.makedirs(output_dir, exist_ok=True)
        
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                ('semantic_pred', 'i4'), ('instance_pred', 'i4')]
        
        if semantic_gt is not None and instance_gt is not None:
            dtype += [('semantic_gt', 'i4'), ('instance_gt', 'i4')]
            vertex = np.array([tuple(points[i]) + (semantic_pred[i], instance_pred[i], semantic_gt[i], instance_gt[i]) for i in range(points.shape[0])],
                            dtype=dtype)
        else:
            vertex = np.array([tuple(points[i]) + (semantic_pred[i], instance_pred[i]) for i in range(points.shape[0])],
                            dtype=dtype)

        el = PlyElement.describe(vertex, 'vertex')
        PlyData([el], text=False).write(filename)
    
    @staticmethod
    def save_ply_2(points, instance_pred, filename):
        from plyfile import PlyData, PlyElement
        output_dir = os.path.dirname(filename)
        os.makedirs(output_dir, exist_ok=True)
        
        # Filter out points with instance_pred == -1
        valid_mask = instance_pred != -1
        valid_points = points[valid_mask]
        valid_instance_pred = instance_pred[valid_mask]
        
        # Define the dtype for the vertex elements
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('instance_pred', 'i4')]
        
        # Create an array of vertices
        vertex = np.array([tuple(valid_points[i]) + (valid_instance_pred[i],) for i in range(valid_points.shape[0])], dtype=dtype)

        # Describe the elements and save the ply file
        el = PlyElement.describe(vertex, 'vertex')
        PlyData([el], text=False).write(filename)

    @staticmethod
    def save_ply_2_withscore(points, instance_pred, filename, scores):
        from plyfile import PlyData, PlyElement
        output_dir = os.path.dirname(filename)
        os.makedirs(output_dir, exist_ok=True)
        
        # Filter out points with instance_pred == -1
        valid_mask = instance_pred != -1
        valid_points = points[valid_mask]
        valid_instance_pred = instance_pred[valid_mask]
        valid_scores = scores[valid_mask]
        
        # Define the dtype for the vertex elements
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('instance_pred', 'i4'), ('score', 'f4')]
        
        # Create an array of vertices
        vertex = np.array([tuple(valid_points[i]) + (valid_instance_pred[i], valid_scores[i]) for i in range(valid_points.shape[0])], dtype=dtype)

        # Describe the elements and save the ply file
        el = PlyElement.describe(vertex, 'vertex')
        PlyData([el], text=False).write(filename)
    
    @staticmethod
    def compute_mean_cov(features, labels):
        # features: NxD tensor, where N is the number of points, D is the embedding dimension
        # labels: N tensor, the instance label for each point
        unique_labels = torch.unique(labels)
        mean_cov_dict = {}
        
        for label in unique_labels:
            # Get the points belonging to this instance
            instance_points = features[labels == label]
            
            if instance_points.shape[0] > 1:
                # Calculate the mean vector (mu)
                mu = instance_points.mean(dim=0)
                
                # Calculate the covariance matrix (Sigma)
                cov = torch.cov(instance_points.T)
            else:
                # If only one point, set cov as a zero matrix
                mu = instance_points[0]
                cov = torch.zeros((features.shape[1], features.shape[1]), device=features.device)
            mean_cov_dict[label.item()] = (mu, cov)

        return mean_cov_dict

    
    @staticmethod
    def compute_mahalanobis_distances(features, labels, mean_cov_dict):
        num_points = features.shape[0]
        distances = torch.zeros(num_points, device=features.device)
        
        # Retrieve the mean and covariance matrix for each label and batch them
        unique_labels = torch.unique(labels)
        num_labels = unique_labels.size(0)
        
        mu_list = []
        cov_inv_list = []
        
        # Preprocessing: compute the inverse of all covariance matrices
        for label in unique_labels:
            mu, cov = mean_cov_dict[label.item()]
            
            # Add a small diagonal matrix to prevent singular matrix issues
            cov += torch.eye(cov.shape[0]).to(cov.device) * 1e-6
            
            # Compute the inverse matrix
            cov_inv = torch.inverse(cov)
            
            mu_list.append(mu)
            cov_inv_list.append(cov_inv)
        
        # Convert lists to tensors
        mu_tensor = torch.stack(mu_list)  # (num_labels, D)
        cov_inv_tensor = torch.stack(cov_inv_list)  # (num_labels, D, D)
        
        # Find the corresponding mean and inverse covariance matrix for each point based on its label
        label_indices = torch.searchsorted(unique_labels, labels)
        
        # Get the mean and inverse covariance matrix for each point
        mu_for_points = mu_tensor[label_indices]  # (N, D)
        cov_inv_for_points = cov_inv_tensor[label_indices]  # (N, D, D)
        
        # Compute (q_i - mu)
        delta = features - mu_for_points  # (N, D)
        
        # Compute Mahalanobis distance using batch operations
        distances = torch.einsum('nd,ndd,nd->n', delta, cov_inv_for_points, delta)  # (N,)
        
        # Ensure distances are non-negative
        distances = torch.clamp(distances, min=0)
        
        return torch.sqrt(distances)

    @staticmethod
    def compute_scores(distances):
        # Compute the scores based on Mahalanobis distance
        scores = torch.exp(-distances**2)
        return scores
    
    @staticmethod
    def compute_mean(embed_logits, labels):
        """
        Compute the mean vector (mu) for each instance.
        :param embed_logits: N x D tensor, where N is the number of points and D is the feature dimension
        :param labels: N tensor, where each value is the instance label for each point
        :return: Mean vector for each instance
        """
        num_classes = labels.max().item() + 1
        one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()  # (N, num_labels)

        # One-hot encode the labels
        #one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=num_labels).float()  # (N, num_labels)
        
        # Count the number of points per label
        count_per_label = one_hot_labels.sum(dim=0).unsqueeze(-1)  # (num_labels, 1)
        
        # Sum the feature vectors for each label
        sum_per_label = torch.matmul(one_hot_labels.T, embed_logits)  # Equivalent to einsum

        #sum_per_label = torch.einsum('nd,nl->ld', embed_logits, one_hot_labels)  # (num_labels, D)
        
        # Calculate the mean vector for each instance
        mean_per_label = sum_per_label / count_per_label  # (num_labels, D)
        
        return mean_per_label, torch.unique(labels)

    @staticmethod
    def compute_euclidean_distances(embed_logits, labels, mean_per_label, unique_labels):
        """
        Compute the Euclidean distance between each point's feature vector and its corresponding instance mean vector.
        :param embed_logits: N x D tensor of feature vectors
        :param labels: N tensor of instance labels
        :param mean_per_label: Mean vector for each instance
        :param unique_labels: Unique labels for each instance
        :return: Euclidean distance for each point
        """
        # Convert labels to indices corresponding to the unique labels
        label_indices = torch.searchsorted(unique_labels, labels)
        
        # Retrieve the mean vector for each point based on its label
        mu_for_points = mean_per_label[label_indices]  # (N, D)
        
        # Compute the Euclidean distance between each point and the corresponding mean vector
        distances = torch.norm(embed_logits - mu_for_points, dim=-1)  # (N,)
        
        return distances

    @staticmethod
    def compute_scores_2(distances):
        """
        Convert Euclidean distances to scores, where a smaller distance results in a higher score.
        :param distances: Euclidean distance for each point
        :return: Scores in the range [0, 1]
        """
        return torch.exp(-distances)  # The smaller the distance, the closer the score is to 1

    @staticmethod
    def normalize_scores_per_instance(scores, labels):
        """
        Normalize the scores for each instance separately to the range [0, 1].
        :param scores: N tensor of scores for each point.
        :param labels: N tensor of instance labels for each point.
        :return: Normalized scores in the range [0, 1].
        """
        unique_labels = torch.unique(labels)
        
        for label in unique_labels:
            # Get the mask for points belonging to the current instance
            mask = labels == label
            
            # Get the scores for the current instance
            instance_scores = scores[mask]
            
            # Min-max normalization: (score - min) / (max - min)
            min_score = instance_scores.min()
            max_score = instance_scores.max()
            
            if max_score > min_score:
                # Normalize if there is a range
                scores[mask] = (instance_scores - min_score) / (max_score - min_score)
            else:
                # If all scores are the same, set them to 1 (since all distances are the same)
                scores[mask] = 1.0
        
        return scores
    
    @staticmethod
    def save_ply_with_logits(points, offset_logits, embed_logits, bi_semantic_logits, semantic_logits, voxel_instance_labels, voxel_semantic_labels, scores_embed, qscore_logits, filename):
        from plyfile import PlyData, PlyElement
        output_dir = os.path.dirname(filename)
        os.makedirs(output_dir, exist_ok=True)
        
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),  
                ('offset_x', 'f4'), ('offset_y', 'f4'), ('offset_z', 'f4'),  
                ('embed_0', 'f4'), ('embed_1', 'f4'), ('embed_2', 'f4'), ('embed_3', 'f4'), ('embed_4', 'f4'),
                ('bi_semantic_0', 'f4'), ('bi_semantic_1', 'f4'),  
                ('semantic_0', 'f4'), ('semantic_1', 'f4'), ('semantic_2', 'f4'),
                ('ins_label', 'f4'), ('sem_label', 'f4'), ('score_embed', 'f4'), ('score_pre', 'f4')] 

        vertex = np.array([
            tuple(points[i]) + 
            tuple(offset_logits[i]) + 
            tuple(embed_logits[i]) + 
            tuple(bi_semantic_logits[i]) + 
            tuple(semantic_logits[i]) +
            (voxel_instance_labels[i].item(),) +
            (voxel_semantic_labels[i].item(),) +
            (scores_embed[i].item(),) +
            (qscore_logits[i].item(),)
            for i in range(points.shape[0])
        ], dtype=dtype)

        el = PlyElement.describe(vertex, 'vertex')
        PlyData([el], text=False).write(filename)

    @staticmethod
    def save_ply_withscore(points, semantic_pred, instance_pred, scores, filename, semantic_gt=None, instance_gt=None):
        from plyfile import PlyData, PlyElement
        output_dir = os.path.dirname(filename)
        os.makedirs(output_dir, exist_ok=True)
        
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                ('semantic_pred', 'i4'), ('instance_pred', 'i4'), ('score', 'f4')]
        
        if semantic_gt is not None and instance_gt is not None:
            dtype += [('semantic_gt', 'i4'), ('instance_gt', 'i4')]
            vertex = np.array([tuple(points[i]) + (semantic_pred[i], instance_pred[i], scores[i], semantic_gt[i], instance_gt[i]) for i in range(points.shape[0])],
                            dtype=dtype)
        else:
            vertex = np.array([tuple(points[i]) + (semantic_pred[i], instance_pred[i], scores[i]) for i in range(points.shape[0])],
                            dtype=dtype)

        el = PlyElement.describe(vertex, 'vertex')
        PlyData([el], text=False).write(filename)

    @staticmethod
    def save_bluepoints(points, semantic_pred, instance_pred, scores, filename, semantic_gt=None, instance_gt=None):
        from plyfile import PlyData, PlyElement

        output_dir = os.path.dirname(filename)
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract base filename and index
        match = re.search(r'_(\d+)\.ply$', filename)
        if match:
            num = int(match.group(1)) + 1
            base_name = filename[:match.start(1)]
        else:
            num = 1
            base_name = filename.replace('.ply', '')
            #if not base_name.endswith('_'):
            #    base_name += '_'

        # Ensure "bluepoints" does not get repeated in filename
        if "bluepoints" in base_name:
            base_name = re.sub(r'_bluepoints_', '', base_name)  # Remove trailing "_bluepoints" if it exists

        new_filename = f"{base_name}_{num}.ply"
        new_filename_filtered = f"{base_name}_bluepoints_{num}.ply"

        # Determine previous bluepoints file
        prev_bluepoints_filename = f"{base_name}_bluepoints_{num-1}.ply"

        # If previous bluepoints file exists, load its semantic_pred
        if os.path.exists(prev_bluepoints_filename):
            print(f"Loading semantic_pred from {prev_bluepoints_filename}")
            plydata = PlyData.read(prev_bluepoints_filename)
            prev_semantic_pred = np.array(plydata['vertex']['semantic_pred'])  # 读取存储的 semantic_pred
            semantic_pred = prev_semantic_pred  # Override input semantic_pred

        # Filter points that meet the condition
        mask = (semantic_pred != 0) & (instance_pred == -1)
        points_filtered = points[mask]
        semantic_pred_filtered = semantic_pred[mask]  # 也保存 semantic_pred
        semantic_gt_filtered = semantic_gt[mask] if semantic_gt is not None else None
        instance_gt_filtered = instance_gt[mask] if instance_gt is not None else None

        # Keep points that do not meet the condition
        points_remain = points[~mask]
        semantic_pred_remain = semantic_pred[~mask]
        instance_pred_remain = instance_pred[~mask]
        scores_remain = scores[~mask]
        semantic_gt_remain = semantic_gt[~mask] if semantic_gt is not None else None
        instance_gt_remain = instance_gt[~mask] if instance_gt is not None else None

        # Save the unfiltered point cloud only if it contains points
        if points_remain.shape[0] > 0:
            dtype_remain = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                            ('semantic_pred', 'i4'), ('instance_pred', 'i4'), ('score', 'f4')]
            if semantic_gt is not None and instance_gt is not None:
                dtype_remain += [('semantic_gt', 'i4'), ('instance_gt', 'i4')]
                vertex_remain = np.array(
                    [tuple(points_remain[i]) + (semantic_pred_remain[i], instance_pred_remain[i], scores_remain[i], semantic_gt_remain[i], instance_gt_remain[i])
                    for i in range(points_remain.shape[0])], dtype=dtype_remain)
            else:
                vertex_remain = np.array(
                    [tuple(points_remain[i]) + (semantic_pred_remain[i], instance_pred_remain[i], scores_remain[i])
                    for i in range(points_remain.shape[0])], dtype=dtype_remain)

            el_remain = PlyElement.describe(vertex_remain, 'vertex')
            PlyData([el_remain], text=False).write(new_filename)

        # Save the filtered point cloud (with semantic_pred)
        if points_filtered.shape[0] > 0:
            dtype_filtered = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                            ('semantic_pred', 'i4'), ('semantic_seg', 'i4'), ('treeID', 'i4')]  # 添加 semantic_pred
            vertex_filtered = np.array(
                [tuple(points_filtered[i]) + (semantic_pred_filtered[i], semantic_gt_filtered[i]+1, instance_gt_filtered[i])
                for i in range(points_filtered.shape[0])], dtype=dtype_filtered)

            el_filtered = PlyElement.describe(vertex_filtered, 'vertex')
            PlyData([el_filtered], text=False).write(new_filename_filtered)


    @staticmethod
    def finalize_semantic_labels_old(all_pre_sem):
        from collections import Counter
        final_semantic_labels = np.full(len(all_pre_sem), -1)
        for i, labels in enumerate(all_pre_sem):
            if labels:
                final_semantic_labels[i] = Counter(labels).most_common(1)[0][0]
        return final_semantic_labels

    @staticmethod
    def finalize_semantic_labels(all_pre_sem: list[list[int]]) -> np.ndarray:

        lens = np.fromiter((len(v) for v in all_pre_sem), dtype=np.int32,
                        count=len(all_pre_sem))                 # (N,)

        if lens.max() == 0:                                        
            return np.full(len(all_pre_sem), -1, dtype=np.int32)

        offsets = np.cumsum(lens, dtype=np.int64)                  # (N,)
        flat    = np.fromiter((lab for sub in all_pre_sem for lab in sub),
                            dtype=np.int32,
                            count=lens.sum())                    # (ΣL,)

        starts = np.empty_like(offsets)
        starts[0] = 0
        starts[1:] = offsets[:-1]

        out = np.full(len(all_pre_sem), -1, dtype=np.int32)
        for idx, (s, e) in enumerate(zip(starts, offsets)):
            if s == e:         
                continue
            slice_vals = flat[s:e]
            binc = np.bincount(slice_vals)
            out[idx] = binc.argmax()

        return out

    
    @staticmethod
    def merge_overlapping_instances(all_pre_ins, best_masks, iou_threshold=0.6):
        """
        Merge overlapping instances based on IoU.

        Args:
            all_pre_ins (numpy.ndarray): Array containing instance labels for each point.
            best_masks (list): List of tuples, each containing (mask_points, instance_id).
            iou_threshold (float): IoU threshold for merging instances.

        Returns:
            numpy.ndarray: Array containing the merged instance labels for each point.
        """
        from scipy.sparse import csr_matrix

        # Create a sparse matrix for each mask using csr_matrix
        num_points = all_pre_ins.shape[0]
        num_masks = len(best_masks)
        data = []
        row_indices = []
        col_indices = []

        for idx, (mask_points, instance_id, score) in enumerate(best_masks):
            data.extend([1] * len(mask_points))
            row_indices.extend(mask_points)
            col_indices.extend([idx] * len(mask_points))

        mask_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(num_points, num_masks), dtype=np.float32)

        # Compute the IoU between masks
        intersection = mask_matrix.T @ mask_matrix
        mask_sizes = mask_matrix.sum(axis=0).A1
        union = mask_sizes[:, None] + mask_sizes - intersection

        # Ensure no division by zero
        union[union == 0] = 1

        iou = intersection / union

        print(f"Computed IoU matrix:\n{iou}")

        # Use Union-Find to manage merging of masks
        uf = UnionFind(num_masks)
        for i in range(num_masks):
            for j in range(i + 1, num_masks):
                if iou[i, j] > iou_threshold:
                    print(f"Merging instances {best_masks[i][1]} and {best_masks[j][1]} with IoU {iou[i, j]}")
                    uf.union(i, j)

        # Map each mask to its root
        #merged_instance_labels = np.copy(all_pre_ins)
        #for i in range(num_masks):
        #    root = uf.find(i)
        #    instance_id_root = best_masks[root][1]
        #    instance_id_i = best_masks[i][1]
        #    merged_instance_labels[merged_instance_labels == instance_id_i] = instance_id_root
        #return merged_instance_labels

        # Update masks dynamically after merging
        merged_masks = []
        merged_instance_labels = np.copy(all_pre_ins)

        for i in range(num_masks):
            root = uf.find(i)
            if root == i:  # If this is the root, create a new merged mask
                merged_points = np.unique(np.concatenate([best_masks[k][0] for k in range(num_masks) if uf.find(k) == i]))
                merged_instance_id = best_masks[i][1]  # Use the instance_id of the root
                merged_score = max([best_masks[k][2] for k in range(num_masks) if uf.find(k) == i])  # Take the max score
                merged_masks.append((merged_points, merged_instance_id, merged_score))

                # Update point-wise labels
                merged_instance_labels[merged_points] = merged_instance_id

        return merged_instance_labels, merged_masks

    @staticmethod
    def merge_overlapping_instances_by_score(all_pre_ins, best_masks, overlap_threshold=0.3):
        """
        Merge overlapping instances based on score and point overlap ratio.
        
        This method compares each mask's points with the union of two masks and determines 
        which mask to keep based on its points' proportion in the union.

        Args:
            all_pre_ins (numpy.ndarray): Array containing instance labels for each point.
            best_masks (list): List of tuples, each containing (mask_points, instance_id, score).
            overlap_threshold (float): Overlap threshold for merging instances based on mask proportion in union.

        Returns:
            numpy.ndarray: Array containing the merged instance labels for each point.
            list: List of merged masks after applying the score-based merging.
        """
        num_masks = len(best_masks)

        # Initialize all points as unassigned (-1) if necessary
        all_pre_ins = np.full(all_pre_ins.shape, -1, dtype=int)
        mask_kept = np.ones(num_masks, dtype=bool)  # Track which masks are kept

        for i in range(num_masks):
            if not mask_kept[i]:
                continue
            mask_i_points = set(best_masks[i][0])
            for j in range(i + 1, num_masks):
                if not mask_kept[j]:
                    continue
                mask_j_points = set(best_masks[j][0])

                # Calculate the intersection of the point sets
                intersection_points = mask_i_points & mask_j_points  # Intersection of the point sets
                if len(intersection_points) == 0:
                    continue  # No overlap, skip

                # Calculate the proportion of intersection relative to each mask
                mask1_ratio = len(intersection_points) / len(mask_i_points)  # intersection / mask1
                mask2_ratio = len(intersection_points) / len(mask_j_points)  # intersection / mask2

                # If either mask's proportion in the intersection is greater than the threshold, merge
                if mask1_ratio > overlap_threshold or mask2_ratio > overlap_threshold:
                    if best_masks[i][2] >= best_masks[j][2]:  # Keep the one with the higher score
                        mask_kept[j] = False  # Discard mask2
                    else:
                        mask_kept[i] = False  # Discard mask1
                        break  # If mask i is discarded, no need to compare with others


        # Filter the masks to keep only those that are not discarded
        masks_after_score_merge = [best_masks[i] for i in range(num_masks) if mask_kept[i]]

        # Update point-wise instance labels
        merged_instance_labels = np.copy(all_pre_ins)
        merged_instance_scores = np.full(all_pre_ins.shape, -1, dtype=float) 
        for mask_points, instance_id, score in masks_after_score_merge:
            merged_instance_labels[mask_points] = instance_id
            merged_instance_scores[mask_points] = score

        return merged_instance_labels, masks_after_score_merge, merged_instance_scores

    @staticmethod
    def merge_overlapping_instances_by_score_speedup(all_pre_ins,
                                            best_masks,
                                            overlap_threshold=0.30):

        N = all_pre_ins.shape[0]

        merged_instance_labels  = np.full(N, -1, dtype=int)
        merged_instance_scores  = np.full(N, -1.0, dtype=float)

        if not best_masks:
            return merged_instance_labels, [], merged_instance_scores

        best_masks = sorted(best_masks, key=lambda x: -x[2])

        taken_flag = np.zeros(N, dtype=np.bool_)        
        kept_masks = []

        for pts_idx, inst_id, score in best_masks:
            pts_idx = np.asarray(pts_idx, dtype=int)   # 防止 list 进来
            overlap = taken_flag[pts_idx].mean()       # == ratio

            if overlap > overlap_threshold:
                continue          # 丢弃低分

            # 保留 & 写入
            kept_masks.append((pts_idx, inst_id, score))

            merged_instance_labels[pts_idx] = inst_id
            merged_instance_scores[pts_idx] = score
            taken_flag[pts_idx]             = True

        return merged_instance_labels, kept_masks, merged_instance_scores


@MODELS.register_module()
class ForAINetV2OneFormer3D_CHMquery(ForAINetV2OneFormer3D_XAwarequery):
    """C2: CHM-Guided Instance Query Initialization.

    Replaces FPS-on-embeddings with a three-step, physically-grounded strategy:

    1. **CHM + Allometric Local Maxima** — Build a 2-D Canopy Height Model from
       tree-class voxels.  Detect local maxima with an allometric window whose
       radius scales with tree height: w(h) = a·h^b (initialised from forestry
       literature; a and b are stored as nn.Parameter so they appear in the
       model state-dict and can be tuned).

    2. **Cylinder Feature Pooling** — For each detected treetop (i*, j*), pool
       the Mamba U-Net features of all voxels within r_pool metres of the peak:
           q_m = AvgPool( F[ voxels in cylinder(i*, j*, r_pool) ] )
       This gives a spatially grounded feature vector that integrates crown
       context rather than sampling a single arbitrary voxel.

    3. **FPS Supplement** — Fill the remaining query budget (max_queries -
       N_chm) with FPS-sampled backbone features from tree voxels, exactly
       as C1 does.  All supplementary queries are real backbone features with
       spatial context, giving the same convergence speed as C1.

    Args:
        num_channels (int): Backbone feature dimension (default 32). Must match
            the parent class *num_channels* argument.
        chm_resolution (float): XY grid-cell size in metres. Default: 0.5.
        chm_min_height (float): Minimum Z for a CHM cell to be a treetop
            candidate (filters ground / shrub returns). Default: 2.0.
        chm_allometric_a (float): Allometric scale factor initialised from
            forestry literature. Default: 0.5.
        chm_allometric_b (float): Allometric exponent initialised from
            forestry literature. Default: 0.6.
        chm_pool_radius (float): XY radius (m) for cylinder feature pooling
            around each detected peak. Default: 0.5.
        chm_num_extra_queries (int): Kept for config compatibility but unused
            (FPS supplement replaces learnable embeddings). Default: 50.
        **kwargs: Forwarded verbatim to the parent class.
    """

    def __init__(self,
                 num_channels=32,
                 chm_resolution=0.5,
                 chm_min_height=2.0,
                 chm_allometric_a=0.5,
                 chm_allometric_b=0.6,
                 chm_pool_radius=0.5,
                 chm_num_extra_queries=50,
                 **kwargs):
        super().__init__(num_channels=num_channels, **kwargs)
        self.chm_resolution        = chm_resolution
        self.chm_min_height        = chm_min_height
        self.chm_pool_radius       = chm_pool_radius
        self.chm_num_extra_queries = chm_num_extra_queries
        # Allometric window parameters: w(h) = a * h^b
        # Stored as nn.Parameter so they live in the model state-dict.
        self.chm_allometric_a = torch.nn.Parameter(
            torch.tensor(float(chm_allometric_a)))
        self.chm_allometric_b = torch.nn.Parameter(
            torch.tensor(float(chm_allometric_b)))

    # ------------------------------------------------------------------
    # Helper: allometric local-maxima detection
    # ------------------------------------------------------------------

    def _detect_chm_peaks(self, chm, occupied, res):
        """Detect CHM local maxima with an allometric variable-radius window.

        For each occupied cell (i, j) the local-maximum condition is:
            CHM(i, j) >= CHM(i', j')  for all (i', j') with
            ||(i', j') - (i, j)||  <= w( CHM(i, j) ) / res
        where w(h) = a * h^b  (allometric scaling from forestry literature).

        Args:
            chm (np.ndarray): 2-D CHM array (n_rows, n_cols), float32.
            occupied (np.ndarray): bool mask of cells with at least one voxel.
            res (float): grid-cell size in metres.

        Returns:
            peak_rows (np.ndarray): row indices of detected peaks.
            peak_cols (np.ndarray): column indices of detected peaks.
        """
        a = float(self.chm_allometric_a.item())
        b = float(self.chm_allometric_b.item())

        candidate_rows, candidate_cols = np.where(
            occupied & (chm >= self.chm_min_height))

        peak_rows, peak_cols = [], []
        for r, c in zip(candidate_rows, candidate_cols):
            h       = float(chm[r, c])
            w_m     = a * (h ** b)            # window radius in metres
            w_cells = max(1, int(np.ceil(w_m / res)))   # radius in grid cells
            r0 = max(0, r - w_cells);  r1 = min(chm.shape[0], r + w_cells + 1)
            c0 = max(0, c - w_cells);  c1 = min(chm.shape[1], c + w_cells + 1)
            if chm[r, c] >= chm[r0:r1, c0:c1].max():
                peak_rows.append(r)
                peak_cols.append(c)

        return np.array(peak_rows, dtype=np.int64), \
               np.array(peak_cols, dtype=np.int64)

    # ------------------------------------------------------------------
    # Helper: cylinder feature pooling
    # ------------------------------------------------------------------

    def _cylinder_pool(self, backbone_feats, xy_voxels, peak_xy_t, r_pool):
        """Average-pool backbone features from voxels within r_pool of each peak.

        q_m = AvgPool( F[ voxels in cylinder(peak_x, peak_y, r_pool) ] )

        If a peak has no voxel within r_pool, the single nearest voxel is used.

        Args:
            backbone_feats (Tensor): (N_voxels, C) Mamba U-Net features.
            xy_voxels (Tensor): (N_voxels, 2) XY positions of voxels (float).
            peak_xy_t (Tensor): (P, 2) XY positions of detected peaks (float).
            r_pool (float): pooling radius in metres.

        Returns:
            pooled (Tensor): (P, C) pooled feature vectors.
        """
        P          = peak_xy_t.shape[0]
        C          = backbone_feats.shape[1]
        device     = backbone_feats.device
        r2         = r_pool * r_pool
        pooled     = torch.zeros(P, C, device=device, dtype=backbone_feats.dtype)

        # Vectorised: (P, N) squared XY distances.
        diff    = peak_xy_t.unsqueeze(1) - xy_voxels.unsqueeze(0)   # (P, N, 2)
        dist_sq = (diff ** 2).sum(-1)                                # (P, N)
        in_cyl  = dist_sq <= r2                                      # (P, N) bool

        for m in range(P):
            mask = in_cyl[m]                        # (N,)
            if mask.any():
                pooled[m] = backbone_feats[mask].mean(0)
            else:
                # Fallback: nearest voxel
                pooled[m] = backbone_feats[dist_sq[m].argmin()]

        return pooled

    # ------------------------------------------------------------------
    # Query sampling override
    # ------------------------------------------------------------------

    def _sample_queries(self, backbone_feats, embed_feats_tree, xyz_voxels,
                        tree_indices, voxel_instance_labels, max_queries):
        """CHM-guided query sampling as per the C2 formulation.

        Steps:
          1. Build CHM from tree-voxel XY/Z positions.
          2. Detect local maxima with allometric window w(h) = a·h^b.
          3. For each peak, pool backbone features from a cylinder of radius
             r_pool  →  q_m = AvgPool(F[voxels in cylinder]).
          4. Fill remaining budget (max_queries - N_chm) with FPS backbone
             features from tree voxels, same as C1.
          5. If more peaks than max_queries, keep the tallest treetops.
          6. Fallback to parent FPS when no peaks are detected.

        Args:
            backbone_feats (Tensor): (N_voxels, C) Mamba U-Net features.
            embed_feats_tree (Tensor): (N_tree, D) embedding features (used
                only for FPS fallback).
            xyz_voxels (Tensor): (N_voxels, 3+) mean world-XYZ of all voxels.
            tree_indices (Tensor): (N_tree,) indices of tree voxels.
            voxel_instance_labels (Tensor or None): (N_voxels,) labels.
            max_queries (int): Maximum queries before extra queries are added.

        Returns:
            Tuple[Tensor, Tensor or None]:
                query_features  (K_chm + K_fps, C),
                query_inst_labels (K_chm + K_fps,) or None.
        """
        device = backbone_feats.device

        if tree_indices.numel() == 0:
            return super()._sample_queries(
                backbone_feats, embed_feats_tree, xyz_voxels,
                tree_indices, voxel_instance_labels, max_queries)

        # --- 1. Build 2-D CHM from tree-voxel positions -----------------
        tree_pts = xyz_voxels[tree_indices, :3].detach().cpu().float()
        xy_np    = tree_pts[:, :2].numpy()
        z_np     = tree_pts[:, 2].numpy()
        xy_min   = xy_np.min(axis=0)
        res      = self.chm_resolution

        col_idx = np.floor((xy_np[:, 0] - xy_min[0]) / res).astype(np.int32)
        row_idx = np.floor((xy_np[:, 1] - xy_min[1]) / res).astype(np.int32)
        n_cols  = int(col_idx.max()) + 1
        n_rows  = int(row_idx.max()) + 1

        chm            = np.full((n_rows, n_cols), -np.inf, dtype=np.float32)
        np.maximum.at(chm, (row_idx, col_idx), z_np)
        occupied       = chm > -np.inf
        chm[~occupied] = 0.0

        # --- 2. Detect local maxima with allometric window --------------
        peak_rows, peak_cols = self._detect_chm_peaks(chm, occupied, res)

        if len(peak_rows) == 0:
            return super()._sample_queries(
                backbone_feats, embed_feats_tree, xyz_voxels,
                tree_indices, voxel_instance_labels, max_queries)

        # --- 3. Cylinder feature pooling around each detected peak ------
        peak_x   = peak_cols * res + xy_min[0] + res * 0.5
        peak_y   = peak_rows * res + xy_min[1] + res * 0.5
        peak_xy_np = np.stack([peak_x, peak_y], axis=1)   # (P, 2)

        # Keep only the tallest peaks up to max_queries
        n_peaks = len(peak_rows)
        if n_peaks > max_queries:
            peak_z     = chm[peak_rows, peak_cols]
            order      = np.argsort(-peak_z)[:max_queries]
            peak_rows  = peak_rows[order]
            peak_cols  = peak_cols[order]
            peak_xy_np = peak_xy_np[order]

        peak_xy_t = torch.from_numpy(peak_xy_np).float().to(device)

        # Pool backbone features for each peak (all voxels, not just tree)
        # so that crown context (leaf points) is also captured.
        xy_voxels_t = xyz_voxels[:, :2].float()
        chm_feats   = self._cylinder_pool(
            backbone_feats, xy_voxels_t, peak_xy_t, self.chm_pool_radius)

        # --- 4. Instance labels for CHM queries -------------------------
        if voxel_instance_labels is not None:
            # Assign the most common voxel instance label inside each cylinder
            diff_all   = peak_xy_t.unsqueeze(1) - xy_voxels_t.unsqueeze(0)
            dist_sq_all = (diff_all ** 2).sum(-1)   # (P, N)
            # Nearest voxel per peak is used for the label
            nearest_vox = dist_sq_all.argmin(dim=1)   # (P,)
            chm_labels  = voxel_instance_labels[nearest_vox]
        else:
            chm_labels = None

        # --- 5. Supplement with FPS queries to fill the budget -----------
        return self._supplement_with_fps(
            chm_feats, chm_labels,
            backbone_feats, embed_feats_tree, xyz_voxels,
            tree_indices, voxel_instance_labels, max_queries, device)

    def _supplement_with_fps(self, chm_feats, chm_labels,
                             backbone_feats, embed_feats_tree, xyz_voxels,
                             tree_indices, voxel_instance_labels,
                             max_queries, device):
        """Fill remaining query budget with FPS backbone features.

        CHM gives N_chm peaks (spatially grounded, physics-based).
        If N_chm < max_queries, use FPS on tree-voxel embeddings to sample
        the remaining (max_queries - N_chm) queries — exactly as C1 does.
        All supplementary queries are real backbone features, not learnable
        embeddings, so they carry spatial context and converge like C1.

        Args:
            chm_feats (Tensor): (N_chm, C) cylinder-pooled CHM features.
            chm_labels (Tensor or None): (N_chm,) GT instance labels.
            backbone_feats / embed_feats_tree / xyz_voxels / tree_indices /
                voxel_instance_labels / max_queries: forwarded from
                _sample_queries for FPS fallback.
            device: target device.

        Returns:
            Tuple[Tensor, Tensor or None]: combined features and labels.
        """
        n_chm = chm_feats.shape[0]
        self._n_chm_queries = n_chm   # expose CHM count for the matcher
        n_fps_needed = max_queries - n_chm

        if n_fps_needed <= 0 or embed_feats_tree.size(0) == 0:
            return chm_feats, chm_labels

        fps_feats, fps_labels = super()._sample_queries(
            backbone_feats, embed_feats_tree, xyz_voxels,
            tree_indices, voxel_instance_labels, n_fps_needed)

        combined_feats = torch.cat([chm_feats, fps_feats.to(device)], dim=0)
        if chm_labels is not None and fps_labels is not None:
            combined_labels = torch.cat([chm_labels, fps_labels.to(device)], dim=0)
        elif chm_labels is not None:
            combined_labels = chm_labels
        else:
            combined_labels = fps_labels

        return combined_feats, combined_labels


@MODELS.register_module()
class ForAINetV2OneFormer3D_CHMquery_Spatial(ForAINetV2OneFormer3D_CHMquery):
    """v5: CHMquery + spatial coordinate passing to the SSM decoder.

    Identical to :class:`ForAINetV2OneFormer3D_CHMquery` except that after
    query sampling it passes two coordinate tensors to the decoder:

    * ``query_xyzs`` — 3-D world positions of each query seed (CHM peak
      centres at (peak_x, peak_y, chm_height) + FPS-selected voxel centres).
    * ``src_xyzs`` — mean world-XYZ of each superpoint, used by
      ``MambaAggregation`` for spatial k-NN (instead of feature-space k-NN).

    These coordinates activate the full LaSSM spatial operations that are
    otherwise skipped in v4 due to the feature-space fallback.
    """

    # ------------------------------------------------------------------
    # Query sampling with position tracking
    # ------------------------------------------------------------------

    def _sample_queries(self, backbone_feats, embed_feats_tree, xyz_voxels,
                        tree_indices, voxel_instance_labels, max_queries):
        """Same logic as CHMquery but also stores query XYZ in self._query_xyzs_cache."""
        device = backbone_feats.device

        # ── Fallback: no tree voxels ─────────────────────────────────────
        if tree_indices.numel() == 0:
            feats, labels = super()._sample_queries(
                backbone_feats, embed_feats_tree, xyz_voxels,
                tree_indices, voxel_instance_labels, max_queries)
            n = feats.shape[0]
            self._query_xyzs_cache = xyz_voxels[:n, :3].float()
            return feats, labels

        # ── CHM peak detection (mirrors parent logic exactly) ─────────────
        tree_pts = xyz_voxels[tree_indices, :3].detach().cpu().float()
        xy_np    = tree_pts[:, :2].numpy()
        z_np     = tree_pts[:, 2].numpy()
        xy_min   = xy_np.min(axis=0)
        res      = self.chm_resolution

        col_idx  = np.floor((xy_np[:, 0] - xy_min[0]) / res).astype(np.int32)
        row_idx  = np.floor((xy_np[:, 1] - xy_min[1]) / res).astype(np.int32)
        n_cols   = int(col_idx.max()) + 1
        n_rows   = int(row_idx.max()) + 1

        chm            = np.full((n_rows, n_cols), -np.inf, dtype=np.float32)
        np.maximum.at(chm, (row_idx, col_idx), z_np)
        occupied       = chm > -np.inf
        chm[~occupied] = 0.0

        peak_rows, peak_cols = self._detect_chm_peaks(chm, occupied, res)

        # ── Fallback: no CHM peaks → FPS with position tracking ───────────
        if len(peak_rows) == 0:
            feats, labels, xyzs = self._fps_with_xyz(
                backbone_feats, embed_feats_tree, xyz_voxels,
                tree_indices, voxel_instance_labels, max_queries, device)
            self._n_chm_queries  = 0
            self._query_xyzs_cache = xyzs
            return feats, labels

        # ── Trim to budget by tallest peaks ────────────────────────────────
        peak_x   = peak_cols * res + xy_min[0] + res * 0.5
        peak_y   = peak_rows * res + xy_min[1] + res * 0.5
        peak_xy_np = np.stack([peak_x, peak_y], axis=1)

        if len(peak_rows) > max_queries:
            peak_z_arr = chm[peak_rows, peak_cols]
            order      = np.argsort(-peak_z_arr)[:max_queries]
            peak_rows  = peak_rows[order]
            peak_cols  = peak_cols[order]
            peak_xy_np = peak_xy_np[order]

        peak_z_vals  = chm[peak_rows, peak_cols]                      # (P,)
        peak_xyz_np  = np.stack(
            [peak_xy_np[:, 0], peak_xy_np[:, 1], peak_z_vals], axis=1)  # (P, 3)
        chm_xyzs     = torch.from_numpy(peak_xyz_np).float().to(device)

        # ── Cylinder feature pooling ───────────────────────────────────────
        peak_xy_t   = torch.from_numpy(peak_xy_np).float().to(device)
        xy_voxels_t = xyz_voxels[:, :2].float()
        chm_feats   = self._cylinder_pool(
            backbone_feats, xy_voxels_t, peak_xy_t, self.chm_pool_radius)

        # ── Instance labels for CHM queries ────────────────────────────────
        if voxel_instance_labels is not None:
            diff_all    = peak_xy_t.unsqueeze(1) - xy_voxels_t.unsqueeze(0)
            dist_sq_all = (diff_all ** 2).sum(-1)
            nearest_vox = dist_sq_all.argmin(dim=1)
            chm_labels  = voxel_instance_labels[nearest_vox]
        else:
            chm_labels = None

        # ── FPS supplement ────────────────────────────────────────────────
        n_chm        = chm_feats.shape[0]
        self._n_chm_queries = n_chm
        n_fps_needed = max_queries - n_chm

        if n_fps_needed <= 0 or embed_feats_tree.size(0) == 0:
            self._query_xyzs_cache = chm_xyzs
            combined_labels = chm_labels
            return chm_feats, combined_labels

        fps_feats, fps_labels, fps_xyzs = self._fps_with_xyz(
            backbone_feats, embed_feats_tree, xyz_voxels,
            tree_indices, voxel_instance_labels, n_fps_needed, device)

        combined_feats = torch.cat([chm_feats, fps_feats.to(device)], dim=0)
        combined_xyzs  = torch.cat([chm_xyzs,  fps_xyzs.to(device)],  dim=0)

        if chm_labels is not None and fps_labels is not None:
            combined_labels = torch.cat(
                [chm_labels, fps_labels.to(device)], dim=0)
        elif chm_labels is not None:
            combined_labels = chm_labels
        else:
            combined_labels = fps_labels

        self._query_xyzs_cache = combined_xyzs
        return combined_feats, combined_labels

    def _fps_with_xyz(self, backbone_feats, embed_feats_tree, xyz_voxels,
                      tree_indices, voxel_instance_labels, max_queries, device):
        """Run FPS and return (feats, labels, xyzs) with position tracking."""
        if embed_feats_tree.size(0) == 0:
            feats  = backbone_feats[tree_indices]
            xyzs   = xyz_voxels[tree_indices, :3].float()
            labels = voxel_instance_labels[tree_indices] \
                if voxel_instance_labels is not None else None
            return feats, labels, xyzs

        batch_t  = torch.zeros(
            embed_feats_tree.size(0), dtype=torch.long, device=device)
        ratio    = min(max_queries / embed_feats_tree.size(0), 1.0)
        topk     = fps(embed_feats_tree, batch_t, ratio=float(ratio))
        selected = tree_indices[topk]
        feats    = backbone_feats[selected]
        xyzs     = xyz_voxels[selected, :3].float()
        labels   = voxel_instance_labels[selected] \
            if voxel_instance_labels is not None else None
        return feats, labels, xyzs

    # ------------------------------------------------------------------
    # Loss override: collect + pass spatial coordinates
    # ------------------------------------------------------------------

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'],
            batch_inputs_dict.get('elastic_coords', None))
        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))

        x = self.extract_feat(x)

        embed_logits      = [self.Embed(y) for y in x]
        bi_semantic_logits = [self.BiSemantic(y) for y in x]

        total_discriminative_loss = 0
        total_semantic_loss_bi    = 0
        batch_size = len(batch_data_samples)

        for i in range(batch_size):
            voxel_superpoints = inverse_mapping[coordinates[:, 0][inverse_mapping] == i]
            voxel_superpoints = torch.unique(voxel_superpoints, return_inverse=True)[1]

            pts_instance_mask = batch_data_samples[i].gt_pts_seg.pts_instance_mask
            instance_mask     = batch_data_samples[i].gt_pts_seg.instance_mask
            device            = pts_instance_mask.device

            voxel_instance_labels = self.get_voxel_instance_labels(
                pts_instance_mask[instance_mask],
                voxel_superpoints[instance_mask])

            valid_voxel_indices  = torch.unique(voxel_superpoints[instance_mask])
            filtered_embed_logits = embed_logits[i][valid_voxel_indices]
            batch_t = torch.full_like(voxel_instance_labels, i)
            discriminative_losses = discriminative_loss(
                filtered_embed_logits, voxel_instance_labels, batch_t, 5)
            for loss_name, loss in discriminative_losses.items():
                total_discriminative_loss += loss if loss_name == "ins_loss" else 0

            bi_semantic_logit   = bi_semantic_logits[i]
            instance_mask_t     = torch.from_numpy(
                batch_data_samples[i].gt_pts_seg.instance_mask).to(device)
            voxel_point_counts  = scatter_add(
                torch.ones_like(instance_mask_t.float()), voxel_superpoints, dim=0)
            foreground_voxel_counts = scatter_add(
                instance_mask_t.float(), voxel_superpoints, dim=0)
            bi_y = ((foreground_voxel_counts / voxel_point_counts) > 0.5).long()
            total_semantic_loss_bi += torch.nn.functional.nll_loss(
                bi_semantic_logit, bi_y.to(torch.int64))

        total_discriminative_loss /= batch_size
        total_semantic_loss_bi    /= batch_size
        loss_final = {
            'discriminative_loss': total_discriminative_loss,
            'semantic_loss_bi':    total_semantic_loss_bi,
        }

        queries          = []
        queries_inslabel = []
        query_xyzs_list  = []   # ← v5 addition
        src_xyzs_list    = []   # ← v5 addition
        n_chm_list       = []

        if self.prepare_epoch:
            if kwargs['epoch'] > self.prepare_epoch:
                for i in range(batch_size):
                    voxel_superpoints = inverse_mapping[
                        coordinates[:, 0][inverse_mapping] == i]
                    voxel_superpoints = torch.unique(
                        voxel_superpoints, return_inverse=True)[1]
                    instance_mask = batch_data_samples[i].gt_pts_seg.instance_mask
                    valid_voxel_indices = torch.unique(
                        voxel_superpoints[instance_mask])

                    with torch.no_grad():
                        if valid_voxel_indices.numel() < 10:
                            queries.append([])
                            queries_inslabel.append([])
                            query_xyzs_list.append(None)
                            src_xyzs_list.append(None)
                            n_chm_list.append(0)
                            continue

                        wood_class = 1
                        semantic_predictions_bi = torch.argmax(
                            bi_semantic_logits[i], dim=1)
                        tree_indices = torch.where(
                            semantic_predictions_bi == wood_class)[0]

                        current_points = batch_inputs_dict['points'][i]
                        current_points_add = scatter_add(
                            current_points, voxel_superpoints, dim=0)
                        voxel_counts = scatter_add(
                            torch.ones_like(current_points[:, 0].float()),
                            voxel_superpoints, dim=0)
                        avg_points = (current_points_add /
                                      voxel_counts.unsqueeze(-1).clamp(min=1))

                        voxel_inst_labels = self.get_voxel_instance_labels(
                            batch_data_samples[i].gt_pts_seg.pts_instance_mask,
                            voxel_superpoints)

                    query_feats, query_labels = self._sample_queries(
                        x[i].detach(),
                        embed_logits[i][tree_indices].detach(),
                        avg_points,
                        tree_indices,
                        voxel_inst_labels,
                        self.query_point_num)

                    queries.append(query_feats)
                    queries_inslabel.append(query_labels)
                    query_xyzs_list.append(             # ← v5 addition
                        self._query_xyzs_cache.to(query_feats.device)
                        if hasattr(self, '_query_xyzs_cache') else None)
                    src_xyzs_list.append(               # ← v5 addition
                        avg_points[:, :3].float())
                    n_chm_list.append(getattr(self, '_n_chm_queries', 0))

                if all(len(q) == 0 for q in queries):
                    pass
                else:
                    if any(len(q) == 0 for q in queries):
                        filtered_results = [
                            (x[i], queries[i], batch_data_samples[i],
                             queries_inslabel[i],
                             query_xyzs_list[i], src_xyzs_list[i], i)
                            for i in range(len(queries))
                            if len(queries[i]) > 0
                        ]
                        (x, queries, batch_data_samples, queries_inslabel,
                         query_xyzs_list, src_xyzs_list,
                         original_indices) = zip(*filtered_results)
                        x                  = list(x)
                        queries            = list(queries)
                        batch_data_samples = list(batch_data_samples)
                        queries_inslabel   = list(queries_inslabel)
                        query_xyzs_list    = list(query_xyzs_list)
                        src_xyzs_list      = list(src_xyzs_list)
                        original_indices   = list(original_indices)
                    else:
                        original_indices = list(range(len(batch_data_samples)))

                    n_chm_filtered = [n_chm_list[j] for j in original_indices]

                    x = self.decoder(                   # ← v5: pass coords
                        x, queries,
                        query_xyzs=query_xyzs_list,
                        src_xyzs=src_xyzs_list)

                    sp_gt_instances = []
                    for i in range(len(batch_data_samples)):
                        voxel_superpoints = inverse_mapping[
                            coordinates[:, 0][inverse_mapping] == original_indices[i]]
                        voxel_superpoints = torch.unique(
                            voxel_superpoints, return_inverse=True)[1]
                        inst_mask = batch_data_samples[i].gt_pts_seg.pts_instance_mask
                        sem_mask  = batch_data_samples[i].gt_pts_seg.pts_semantic_mask
                        assert voxel_superpoints.shape == inst_mask.shape

                        batch_data_samples[i].gt_instances_3d.sp_sem_masks = \
                            self.get_gt_semantic_masks(
                                sem_mask, voxel_superpoints, self.num_classes)
                        batch_data_samples[i].gt_instances_3d.sp_inst_masks = \
                            self.get_gt_inst_masks(inst_mask, voxel_superpoints)

                        orig_labels = \
                            batch_data_samples[i].gt_instances_3d.labels_3d.clone()
                        (batch_data_samples[i].gt_instances_3d.labels_3d,
                         batch_data_samples[i].gt_instances_3d.sp_inst_masks,
                         batch_data_samples[i].gt_instances_3d.ratio_inspoint) = \
                            self.filter_stuff_masks(
                                batch_data_samples[i].gt_instances_3d,
                                self.stuff_classes,
                                batch_data_samples[i].gt_pts_seg.ratio_inspoint)

                        stuff_t   = torch.tensor(
                            self.stuff_classes,
                            device=orig_labels.device)
                        kept_mask = ~torch.isin(orig_labels, stuff_t)
                        kept_ids  = torch.where(kept_mask)[0]
                        remap     = torch.full(
                            (len(orig_labels),), -1,
                            dtype=torch.long, device=orig_labels.device)
                        remap[kept_ids] = torch.arange(
                            len(kept_ids), device=orig_labels.device)

                        q_lbl = queries_inslabel[i]
                        if (q_lbl is not None
                                and isinstance(q_lbl, torch.Tensor)
                                and q_lbl.numel() > 0):
                            valid    = (q_lbl >= 0) & (q_lbl < len(orig_labels))
                            remapped = torch.full_like(q_lbl, -1)
                            remapped[valid] = remap[q_lbl[valid]]
                            queries_inslabel[i] = remapped

                        batch_data_samples[i].gt_instances_3d.query_inslabel = \
                            queries_inslabel[i]
                        batch_data_samples[i].gt_instances_3d.n_chm_queries = \
                            torch.tensor([n_chm_filtered[i]])
                        sp_gt_instances.append(
                            batch_data_samples[i].gt_instances_3d)

                    loss = self.criterion(x, sp_gt_instances)
                    loss_final.update(loss)

        return loss_final

    # ------------------------------------------------------------------
    # Sliding-window decoder override: pass spatial coords to LaSSM decoder
    # ------------------------------------------------------------------

    def _sliding_window_decoder_forward(self, x, query_feats, voxel_xyz):
        """Override: pass spatial coordinates so MambaAggregation uses 3-D
        k-NN instead of the feature-space fallback it was never trained on."""
        q_xyz = getattr(self, '_query_xyzs_cache', None)
        if q_xyz is not None:
            return self.decoder(
                x, [query_feats],
                query_xyzs=[q_xyz],
                src_xyzs=[voxel_xyz[:, :3].float()])
        return self.decoder(x, [query_feats])

    # ------------------------------------------------------------------
    # Predict override (validation path only)
    # ------------------------------------------------------------------

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        lidar_path = batch_data_samples[0].lidar_path

        # ----- Validation path: pass spatial coords to decoder -----
        if 'test' not in lidar_path:
            coordinates, features, inverse_mapping, spatial_shape = self.collate(
                batch_inputs_dict['points'])
            x = spconv.SparseConvTensor(
                features, coordinates, spatial_shape, len(batch_data_samples))
            x = self.extract_feat(x)

            wood_class      = 1
            queries         = []
            query_xyzs_list = []
            src_xyzs_list   = []

            for i in range(len(x)):
                embed_logits_i  = self.Embed(x[i])
                bi_sem_logits_i = self.BiSemantic(x[i])
                tree_idx_i      = torch.where(
                    bi_sem_logits_i.argmax(dim=1) == wood_class)[0]

                vsp     = inverse_mapping[coordinates[:, 0][inverse_mapping] == i]
                vsp     = torch.unique(vsp, return_inverse=True)[1]
                pts_i   = batch_inputs_dict['points'][i].float()
                vox_xyz = (scatter_add(pts_i, vsp, dim=0) /
                           scatter_add(torch.ones_like(pts_i[:, 0]),
                                       vsp, dim=0).unsqueeze(-1).clamp(min=1))

                if tree_idx_i.numel() > 1:
                    q_feats, _ = self._sample_queries(
                        x[i], embed_logits_i[tree_idx_i], vox_xyz,
                        tree_idx_i, None, self.query_point_num)
                    q_xyz = (self._query_xyzs_cache.to(q_feats.device)
                             if hasattr(self, '_query_xyzs_cache') else None)
                else:
                    q_feats = x[i][:min(self.query_point_num, len(x[i]))]
                    q_xyz   = None

                queries.append(q_feats)
                query_xyzs_list.append(q_xyz)
                src_xyzs_list.append(vox_xyz[:, :3].float())

            x = self.decoder(
                x, queries,
                query_xyzs=query_xyzs_list,
                src_xyzs=src_xyzs_list)
            results_list = self.predict_by_feat(x, inverse_mapping)

            for i, data_sample in enumerate(batch_data_samples):
                data_sample.pred_pts_seg = results_list[i]
            return batch_data_samples

        # ----- Test / sliding-window path: delegate to parent -----
        return super().predict(batch_inputs_dict, batch_data_samples, **kwargs)


@MODELS.register_module()
class ForAINetV2OneFormer3D_CHMquery_MultiScale(
        ForAINetV2OneFormer3D_CHMquery_Spatial):
    """v6: Multi-scale CHM query initialisation.

    Runs CHM peak detection at two spatial resolutions:
      * fine   (default 0.3 m): catches closely-spaced trees in dense stands.
      * coarse (default 0.7 m): catches broad-crown trees that fine resolution
        misses because the allometric window subdivides one crown into multiple
        false peaks.

    Peaks from both scales are merged with a greedy spatial NMS: peaks are
    sorted by CHM height (tallest first) and any peak whose XY distance to an
    already-kept peak is < ``chm_min_peak_dist`` metres is suppressed.
    """

    def __init__(self,
                 chm_resolution_fine=0.3,
                 chm_resolution_coarse=0.7,
                 chm_min_peak_dist=1.0,
                 **kwargs):
        # Pass fine resolution as the single-scale default so parent
        # fallback paths still use a sensible grid size.
        super().__init__(chm_resolution=chm_resolution_fine, **kwargs)
        self.chm_resolution_fine   = chm_resolution_fine
        self.chm_resolution_coarse = chm_resolution_coarse
        self.chm_min_peak_dist     = chm_min_peak_dist

    # ------------------------------------------------------------------
    # Multi-scale CHM peak collection with spatial NMS
    # ------------------------------------------------------------------

    def _collect_peaks_multiscale(self, xy_np, z_np, xy_min):
        """Detect CHM peaks at two resolutions and merge with spatial NMS.

        Args:
            xy_np  (np.ndarray): (N, 2) XY positions of tree voxels.
            z_np   (np.ndarray): (N,)  Z  heights of tree voxels.
            xy_min (np.ndarray): (2,)  minimum XY extent.

        Returns:
            peak_xy (np.ndarray): (P, 2) float32 XY centres of kept peaks.
            peak_z  (np.ndarray): (P,)  float32 CHM heights of kept peaks.
        """
        all_peak_xy, all_peak_z = [], []

        for res in (self.chm_resolution_fine, self.chm_resolution_coarse):
            col_idx = np.floor((xy_np[:, 0] - xy_min[0]) / res).astype(np.int32)
            row_idx = np.floor((xy_np[:, 1] - xy_min[1]) / res).astype(np.int32)
            n_cols  = int(col_idx.max()) + 1
            n_rows  = int(row_idx.max()) + 1

            chm            = np.full((n_rows, n_cols), -np.inf, dtype=np.float32)
            np.maximum.at(chm, (row_idx, col_idx), z_np)
            occupied       = chm > -np.inf
            chm[~occupied] = 0.0

            peak_rows, peak_cols = self._detect_chm_peaks(chm, occupied, res)
            if len(peak_rows) == 0:
                continue

            px = peak_cols * res + xy_min[0] + res * 0.5
            py = peak_rows * res + xy_min[1] + res * 0.5
            pz = chm[peak_rows, peak_cols]

            all_peak_xy.append(np.stack([px, py], axis=1).astype(np.float32))
            all_peak_z.append(pz.astype(np.float32))

        if not all_peak_xy:
            return (np.zeros((0, 2), dtype=np.float32),
                    np.zeros(0, dtype=np.float32))

        peak_xy = np.concatenate(all_peak_xy, axis=0)   # (M, 2)
        peak_z  = np.concatenate(all_peak_z,  axis=0)   # (M,)

        # Greedy spatial NMS: sort tallest first, suppress close neighbours.
        order   = np.argsort(-peak_z)
        peak_xy = peak_xy[order]
        peak_z  = peak_z[order]

        kept        = np.ones(len(peak_xy), dtype=bool)
        d_sq_thresh = self.chm_min_peak_dist ** 2
        for i in range(len(peak_xy) - 1):
            if not kept[i]:
                continue
            diffs = peak_xy[i + 1:] - peak_xy[i]       # (rest, 2)
            d_sq  = (diffs ** 2).sum(axis=1)            # (rest,)
            kept[i + 1:][d_sq < d_sq_thresh] = False

        return peak_xy[kept], peak_z[kept]

    # ------------------------------------------------------------------
    # _sample_queries override: multi-scale CHM + spatial coordinate cache
    # ------------------------------------------------------------------

    def _sample_queries(self, backbone_feats, embed_feats_tree, xyz_voxels,
                        tree_indices, voxel_instance_labels, max_queries):
        """Multi-scale CHM query sampling with position tracking."""
        device = backbone_feats.device

        # ── Fallback: no tree voxels ──────────────────────────────────────
        if tree_indices.numel() == 0:
            feats, labels = ForAINetV2OneFormer3D_CHMquery._sample_queries(
                self, backbone_feats, embed_feats_tree, xyz_voxels,
                tree_indices, voxel_instance_labels, max_queries)
            n = feats.shape[0]
            self._query_xyzs_cache = xyz_voxels[:n, :3].float()
            self._n_chm_queries    = 0
            return feats, labels

        # ── Build multi-scale peaks ───────────────────────────────────────
        tree_pts = xyz_voxels[tree_indices, :3].detach().cpu().float()
        xy_np    = tree_pts[:, :2].numpy()
        z_np     = tree_pts[:, 2].numpy()
        xy_min   = xy_np.min(axis=0)

        peak_xy_np, peak_z_vals = self._collect_peaks_multiscale(
            xy_np, z_np, xy_min)

        # ── Fallback: no peaks detected ───────────────────────────────────
        if len(peak_xy_np) == 0:
            feats, labels, xyzs = self._fps_with_xyz(
                backbone_feats, embed_feats_tree, xyz_voxels,
                tree_indices, voxel_instance_labels, max_queries, device)
            self._n_chm_queries    = 0
            self._query_xyzs_cache = xyzs
            return feats, labels

        # ── Trim to budget by tallest peaks (already NMS-sorted) ─────────
        if len(peak_xy_np) > max_queries:
            peak_xy_np  = peak_xy_np[:max_queries]
            peak_z_vals = peak_z_vals[:max_queries]

        peak_xyz_np = np.concatenate(
            [peak_xy_np, peak_z_vals[:, None]], axis=1)   # (P, 3)
        chm_xyzs    = torch.from_numpy(peak_xyz_np).float().to(device)

        # ── Cylinder feature pooling ──────────────────────────────────────
        peak_xy_t   = torch.from_numpy(peak_xy_np).float().to(device)
        xy_voxels_t = xyz_voxels[:, :2].float()
        chm_feats   = self._cylinder_pool(
            backbone_feats, xy_voxels_t, peak_xy_t, self.chm_pool_radius)

        # ── Instance labels for CHM queries ──────────────────────────────
        if voxel_instance_labels is not None:
            diff_all    = peak_xy_t.unsqueeze(1) - xy_voxels_t.unsqueeze(0)
            dist_sq_all = (diff_all ** 2).sum(-1)
            nearest_vox = dist_sq_all.argmin(dim=1)
            chm_labels  = voxel_instance_labels[nearest_vox]
        else:
            chm_labels = None

        # ── FPS supplement ────────────────────────────────────────────────
        n_chm        = chm_feats.shape[0]
        self._n_chm_queries = n_chm
        n_fps_needed = max_queries - n_chm

        if n_fps_needed <= 0 or embed_feats_tree.size(0) == 0:
            self._query_xyzs_cache = chm_xyzs
            return chm_feats, chm_labels

        fps_feats, fps_labels, fps_xyzs = self._fps_with_xyz(
            backbone_feats, embed_feats_tree, xyz_voxels,
            tree_indices, voxel_instance_labels, n_fps_needed, device)

        combined_feats = torch.cat([chm_feats, fps_feats.to(device)], dim=0)
        combined_xyzs  = torch.cat([chm_xyzs,  fps_xyzs.to(device)],  dim=0)

        if chm_labels is not None and fps_labels is not None:
            combined_labels = torch.cat(
                [chm_labels, fps_labels.to(device)], dim=0)
        elif chm_labels is not None:
            combined_labels = chm_labels
        else:
            combined_labels = fps_labels

        self._query_xyzs_cache = combined_xyzs
        return combined_feats, combined_labels



@MODELS.register_module()
class ScanNet200OneFormer3D(ScanNetOneFormer3DMixin, Base3DDetector):
    """OneFormer3D for ScanNet200 dataset.
    
    Args:
        voxel_size (float): Voxel size.
        num_classes (int): Number of classes.
        query_thr (float): Min percent of queries.
        backbone (ConfigDict): Config dict of the backbone.
        neck (ConfigDict, optional): Config dict of the neck.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        matcher (ConfigDict): To match superpoints to objects.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(self,
                 voxel_size,
                 num_classes,
                 query_thr,
                 backbone=None,
                 neck=None,
                 decoder=None,
                 criterion=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super(Base3DDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.decoder = MODELS.build(decoder)
        self.criterion = MODELS.build(criterion)
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.query_thr = query_thr
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, batch_inputs_dict, batch_data_samples):
        """Extract features from sparse tensor.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_pts_seg.sp_pts_mask`.

        Returns:
            Tuple:
                List[Tensor]: of len batch_size,
                    each of shape (n_points_i, n_channels).
                List[Tensor]: of len batch_size,
                    each of shape (n_points_i, n_classes + 1).
        """
        # construct tensor field
        coordinates, features = [], []
        for i in range(len(batch_inputs_dict['points'])):
            if 'elastic_coords' in batch_inputs_dict:
                coordinates.append(
                    batch_inputs_dict['elastic_coords'][i] * self.voxel_size)
            else:
                coordinates.append(batch_inputs_dict['points'][i][:, :3])
            features.append(batch_inputs_dict['points'][i][:, 3:])
        
        coordinates, features = ME.utils.batch_sparse_collate(
            [(c / self.voxel_size, f) for c, f in zip(coordinates, features)],
            device=coordinates[0].device)
        field = ME.TensorField(coordinates=coordinates, features=features)

        # forward of backbone and neck
        x = self.backbone(field.sparse())
        if self.with_neck:
            x = self.neck(x)
        x = x.slice(field).features

        # apply scatter_mean
        sp_pts_masks, n_super_points = [], []
        for data_sample in batch_data_samples:
            sp_pts_mask = data_sample.gt_pts_seg.sp_pts_mask
            sp_pts_masks.append(sp_pts_mask + sum(n_super_points))
            n_super_points.append(sp_pts_mask.max() + 1)
        x = scatter_mean(x, torch.cat(sp_pts_masks), dim=0)  # todo: do we need dim?

        # apply cls_layer
        features = []
        for i in range(len(n_super_points)):
            begin = sum(n_super_points[:i])
            end = sum(n_super_points[:i + 1])
            features.append(x[begin: end])
        return features

    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instances_3d` and `gt_sem_seg_3d`.
        Returns:
            dict: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs_dict, batch_data_samples)
        gt_instances = [s.gt_instances_3d for s in batch_data_samples]
        queries, gt_instances = self._select_queries(x, gt_instances)
        x = self.decoder(x, queries)
        return self.criterion(x, gt_instances)

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_pts_seg.sp_pts_mask`.
        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample contains 'pred_pts_seg'.
            And the `pred_pts_seg` contains following keys.
                - instance_scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - instance_labels (Tensor): Labels of instances, has a shape
                    (num_instances, )
                - pts_instance_mask (Tensor): Instance mask, has a shape
                    (num_points, num_instances) of type bool.
        """
        assert len(batch_data_samples) == 1
        x = self.extract_feat(batch_inputs_dict, batch_data_samples)
        x = self.decoder(x, x)
        pred_pts_seg = self.predict_by_feat(
            x, batch_data_samples[0].gt_pts_seg.sp_pts_mask)
        batch_data_samples[0].pred_pts_seg = pred_pts_seg[0]
        return batch_data_samples


@MODELS.register_module()
class S3DISOneFormer3D(Base3DDetector):
    r"""OneFormer3D for S3DIS dataset.

    Args:
        in_channels (int): Number of input channels.
        num_channels (int): NUmber of output channels.
        voxel_size (float): Voxel size.
        num_classes (int): Number of classes.
        min_spatial_shape (int): Minimal shape for spconv tensor.
        backbone (ConfigDict): Config dict of the backbone.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 num_channels,
                 voxel_size,
                 num_classes,
                 min_spatial_shape,
                 backbone=None,
                 decoder=None,
                 criterion=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super(Base3DDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.unet = MODELS.build(backbone)
        self.decoder = MODELS.build(decoder)
        self.criterion = MODELS.build(criterion)
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.min_spatial_shape = min_spatial_shape
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers(in_channels, num_channels)

    def _init_layers(self, in_channels, num_channels):
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                num_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='subm1'))
        self.output_layer = spconv.SparseSequential(
            torch.nn.BatchNorm1d(num_channels, eps=1e-4, momentum=0.1),
            torch.nn.ReLU(inplace=True))

    def extract_feat(self, x):
        """Extract features from sparse tensor.

        Args:
            x (SparseTensor): Input sparse tensor of shape
                (n_points, in_channels).

        Returns:
            List[Tensor]: of len batch_size,
                each of shape (n_points_i, n_channels).
        """
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        out = []
        for i in x.indices[:, 0].unique():
            out.append(x.features[x.indices[:, 0] == i])
        return out

    def collate(self, points, elastic_points=None):
        """Collate batch of points to sparse tensor.

        Args:
            points (List[Tensor]): Batch of points.
            quantization_mode (SparseTensorQuantizationMode): Minkowski
                quantization mode. We use random sample for training
                and unweighted average for inference.

        Returns:
            TensorField: Containing features and coordinates of a
                sparse tensor.
        """
        if elastic_points is None:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((p[:, :3] - p[:, :3].min(0)[0]) / self.voxel_size,
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                 for p in points])
        else:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((el_p - el_p.min(0)[0]),
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                 for el_p, p in zip(elastic_points, points)])

        spatial_shape = torch.clip(
            coordinates.max(0)[0][1:] + 1, self.min_spatial_shape)
        field = ME.TensorField(features=features, coordinates=coordinates)
        tensor = field.sparse()
        coordinates = tensor.coordinates
        features = tensor.features
        inverse_mapping = field.inverse_mapping(tensor.coordinate_map_key)

        return coordinates, features, inverse_mapping, spatial_shape

    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instances_3d` and `gt_sem_seg_3d`.
        Returns:
            dict: A dictionary of loss components.
        """

        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'],
            batch_inputs_dict.get('elastic_coords', None))
        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))

        x = self.extract_feat(x)

        x = self.decoder(x)

        sp_gt_instances = []
        for i in range(len(batch_data_samples)):
            voxel_superpoints = inverse_mapping[coordinates[:, 0][ \
                                                        inverse_mapping] == i]
            voxel_superpoints = torch.unique(voxel_superpoints,
                                             return_inverse=True)[1]
            inst_mask = batch_data_samples[i].gt_pts_seg.pts_instance_mask
            sem_mask = batch_data_samples[i].gt_pts_seg.pts_semantic_mask
            assert voxel_superpoints.shape == inst_mask.shape

            batch_data_samples[i].gt_instances_3d.sp_sem_masks = \
                                self.get_gt_semantic_masks(sem_mask,
                                                            voxel_superpoints,
                                                            self.num_classes)
            batch_data_samples[i].gt_instances_3d.sp_inst_masks = \
                                self.get_gt_inst_masks(inst_mask,
                                                       voxel_superpoints)
            sp_gt_instances.append(batch_data_samples[i].gt_instances_3d)

        loss = self.criterion(x, sp_gt_instances)
        return loss

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.
        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instance_3d` and `gt_sem_seg_3d`.
        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample contains 'pred_pts_seg'.
            And the `pred_pts_seg` contains following keys.
                - instance_scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - instance_labels (Tensor): Labels of instances, has a shape
                    (num_instances, )
                - pts_instance_mask (Tensor): Instance mask, has a shape
                    (num_points, num_instances) of type bool.
        """

        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'])
        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))

        x = self.extract_feat(x)

        x = self.decoder(x)

        results_list = self.predict_by_feat(x, inverse_mapping)

        for i, data_sample in enumerate(batch_data_samples):
            data_sample.pred_pts_seg = results_list[i]
        return batch_data_samples

    def predict_by_feat(self, out, superpoints):
        """Predict instance, semantic, and panoptic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).

        Returns:
            List[PointData]: of len 1 with `pts_semantic_mask`,
                `pts_instance_mask`, `instance_labels`, `instance_scores`.
        """
        pred_labels = out['cls_preds'][0]
        pred_masks = out['masks'][0]
        pred_scores = out['scores'][0]

        inst_res = self.pred_inst(pred_masks[:-self.test_cfg.num_sem_cls, :],
                                  pred_scores[:-self.test_cfg.num_sem_cls, :],
                                  pred_labels[:-self.test_cfg.num_sem_cls, :],
                                  superpoints, self.test_cfg.inst_score_thr)
        sem_res = self.pred_sem(pred_masks[-self.test_cfg.num_sem_cls:, :],
                                superpoints)
        pan_res = self.pred_pan(pred_masks, pred_scores, pred_labels,
                                superpoints)

        pts_semantic_mask = [sem_res.cpu().numpy(), pan_res[0].cpu().numpy()]
        pts_instance_mask = [inst_res[0].cpu().bool().numpy(),
                             pan_res[1].cpu().numpy()]

        return [
            PointData(
                pts_semantic_mask=pts_semantic_mask,
                pts_instance_mask=pts_instance_mask,
                instance_labels=inst_res[1].cpu().numpy(),
                instance_scores=inst_res[2].cpu().numpy())]

    def pred_inst(self, pred_masks, pred_scores, pred_labels,
                  superpoints, score_threshold):
        """Predict instance masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_queries, n_points).
            pred_scores (Tensor): of shape (n_queris, 1).
            pred_labels (Tensor): of shape (n_queries, n_instance_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.

        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        scores = F.softmax(pred_labels, dim=-1)[:, :-1]
        scores *= pred_scores

        labels = torch.arange(
            self.num_classes,
            device=scores.device).unsqueeze(0).repeat(
                self.decoder.num_queries - self.test_cfg.num_sem_cls,
                1).flatten(0, 1)
        
        scores, topk_idx = scores.flatten(0, 1).topk(
            self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, self.num_classes, rounding_mode='floor')
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        if self.test_cfg.get('obj_normalization', None):
            mask_pred_thr = mask_pred_sigmoid > \
                self.test_cfg.obj_normalization_thr
            mask_scores = (mask_pred_sigmoid * mask_pred_thr).sum(1) / \
                (mask_pred_thr.sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel=kernel)

        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr
        mask_pred = mask_pred[:, superpoints]
        # score_thr
        score_mask = scores > score_threshold
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return mask_pred, labels, scores
   
    def pred_sem(self, pred_masks, superpoints):
        """Predict semantic masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_points, n_semantic_classes).
            superpoints (Tensor): of shape (n_raw_points,).        

        Returns:
            Tensor: semantic preds of shape
                (n_raw_points, 1).
        """
        mask_pred = pred_masks.sigmoid()
        mask_pred = mask_pred[:, superpoints]
        seg_map = mask_pred.argmax(0)
        return seg_map

    def pred_pan(self, pred_masks, pred_scores, pred_labels,
                 superpoints):
        """Predict panoptic masks for a single scene.
        
        Args:
            pred_masks (Tensor): of shape (n_queries, n_points).
            pred_scores (Tensor): of shape (n_queris, 1).
            pred_labels (Tensor): of shape (n_queries, n_instance_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
        
        Returns:
            Tuple:
                Tensor: semantic mask of shape (n_raw_points,),
                Tensor: instance mask of shape (n_raw_points,).
        """
        stuff_cls = pred_masks.new_tensor(self.test_cfg.stuff_cls).long()
        sem_map = self.pred_sem(
            pred_masks[-self.test_cfg.num_sem_cls + stuff_cls, :], superpoints)
        sem_map_src_mapping = stuff_cls[sem_map]

        n_cls = self.test_cfg.num_sem_cls
        thr = self.test_cfg.pan_score_thr
        mask_pred, labels, scores = self.pred_inst(
            pred_masks[:-n_cls, :], pred_scores[:-n_cls, :],
            pred_labels[:-n_cls, :], superpoints, thr)
        
        thing_idxs = torch.zeros_like(labels)
        for thing_cls in self.test_cfg.thing_cls:
            thing_idxs = thing_idxs.logical_or(labels == thing_cls)
        
        mask_pred = mask_pred[thing_idxs]
        scores = scores[thing_idxs]
        labels = labels[thing_idxs]

        if mask_pred.shape[0] == 0:
            return sem_map_src_mapping, sem_map

        scores, idxs = scores.sort()
        labels = labels[idxs]
        mask_pred = mask_pred[idxs]

        inst_idxs = torch.arange(
            0, mask_pred.shape[0], device=mask_pred.device).view(-1, 1)
        insts = inst_idxs * mask_pred
        things_inst_mask, idxs = insts.max(axis=0)
        things_sem_mask = labels[idxs]

        inst_idxs, num_pts = things_inst_mask.unique(return_counts=True)
        for inst, pts in zip(inst_idxs, num_pts):
            if pts <= self.test_cfg.npoint_thr and inst != 0:
                things_inst_mask[things_inst_mask == inst] = 0

        things_inst_mask = torch.unique(
            things_inst_mask, return_inverse=True)[1]
        things_inst_mask[things_inst_mask != 0] += len(stuff_cls) - 1
        things_sem_mask[things_inst_mask == 0] = 0
      
        sem_map_src_mapping[things_inst_mask != 0] = 0
        sem_map[things_inst_mask != 0] = 0
        sem_map += things_inst_mask
        sem_map_src_mapping += things_sem_mask
        return sem_map_src_mapping, sem_map

    @staticmethod
    def get_gt_semantic_masks(mask_src, sp_pts_mask, num_classes):    
        """Create ground truth semantic masks.
        
        Args:
            mask_src (Tensor): of shape (n_raw_points, 1).
            sp_pts_mask (Tensor): of shape (n_raw_points, 1).
            num_classes (Int): number of classes.
        
        Returns:
            sp_masks (Tensor): semantic mask of shape (n_points, num_classes).
        """

        mask = torch.nn.functional.one_hot(
            mask_src, num_classes=num_classes + 1)

        mask = mask.T
        sp_masks = scatter_mean(mask.float(), sp_pts_mask, dim=-1)
        sp_masks = sp_masks > 0.5
        sp_masks[-1, sp_masks.sum(axis=0) == 0] = True
        assert sp_masks.sum(axis=0).max().item() == 1

        return sp_masks

    @staticmethod
    def get_gt_inst_masks(mask_src, sp_pts_mask):
        """Create ground truth instance masks.
        
        Args:
            mask_src (Tensor): of shape (n_raw_points, 1).
            sp_pts_mask (Tensor): of shape (n_raw_points, 1).
        
        Returns:
            sp_masks (Tensor): semantic mask of shape (n_points, num_inst_obj).
        """
        mask = mask_src.clone()
        if torch.sum(mask == -1) != 0:
            mask[mask == -1] = torch.max(mask) + 1
            mask = torch.nn.functional.one_hot(mask)[:, :-1]
        else:
            mask = torch.nn.functional.one_hot(mask)

        mask = mask.T
        sp_masks = scatter_mean(mask, sp_pts_mask, dim=-1)
        sp_masks = sp_masks > 0.5

        return sp_masks


@MODELS.register_module()
class InstanceOnlyOneFormer3D(Base3DDetector):
    r"""InstanceOnlyOneFormer3D for training on different datasets jointly.

    Args:
        in_channels (int): Number of input channels.
        num_channels (int): Number of output channels.
        voxel_size (float): Voxel size.
        num_classes_1dataset (int): Number of classes in the first dataset.
        num_classes_2dataset (int): Number of classes in the second dataset.
        prefix_1dataset (string): Prefix for the first dataset.
        prefix_2dataset (string): Prefix for the second dataset.
        min_spatial_shape (int): Minimal shape for spconv tensor.
        backbone (ConfigDict): Config dict of the backbone.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 num_channels,
                 voxel_size,
                 num_classes_1dataset,
                 num_classes_2dataset,
                 prefix_1dataset,
                 prefix_2dataset,
                 min_spatial_shape,
                 backbone=None,
                 decoder=None,
                 criterion=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super(InstanceOnlyOneFormer3D, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.num_classes_1dataset = num_classes_1dataset 
        self.num_classes_2dataset = num_classes_2dataset
        
        self.prefix_1dataset = prefix_1dataset 
        self.prefix_2dataset = prefix_2dataset
        
        self.unet = MODELS.build(backbone)
        self.decoder = MODELS.build(decoder)
        self.criterion = MODELS.build(criterion)
        self.voxel_size = voxel_size
        self.min_spatial_shape = min_spatial_shape
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers(in_channels, num_channels)
    
    def _init_layers(self, in_channels, num_channels):
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                num_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='subm1'))
        self.output_layer = spconv.SparseSequential(
            torch.nn.BatchNorm1d(num_channels, eps=1e-4, momentum=0.1),
            torch.nn.ReLU(inplace=True))

    def extract_feat(self, x):
        """Extract features from sparse tensor.

        Args:
            x (SparseTensor): Input sparse tensor of shape
                (n_points, in_channels).

        Returns:
            List[Tensor]: of len batch_size,
                each of shape (n_points_i, n_channels).
        """
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        out = []
        for i in x.indices[:, 0].unique():
            out.append(x.features[x.indices[:, 0] == i])
        return out

    def collate(self, points, elastic_points=None):
        """Collate batch of points to sparse tensor.

        Args:
            points (List[Tensor]): Batch of points.
            quantization_mode (SparseTensorQuantizationMode): Minkowski
                quantization mode. We use random sample for training
                and unweighted average for inference.

        Returns:
            TensorField: Containing features and coordinates of a
                sparse tensor.
        """
        if elastic_points is None:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((p[:, :3] - p[:, :3].min(0)[0]) / self.voxel_size,
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                 for p in points])
        else:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((el_p - el_p.min(0)[0]),
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                 for el_p, p in zip(elastic_points, points)])
        
        spatial_shape = torch.clip(
            coordinates.max(0)[0][1:] + 1, self.min_spatial_shape)
        field = ME.TensorField(features=features, coordinates=coordinates)
        tensor = field.sparse()
        coordinates = tensor.coordinates
        features = tensor.features
        inverse_mapping = field.inverse_mapping(tensor.coordinate_map_key)

        return coordinates, features, inverse_mapping, spatial_shape

    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instances_3d` and `gt_sem_seg_3d`.
        Returns:
            dict: A dictionary of loss components.
        """
        
        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'],
            batch_inputs_dict.get('elastic_coords', None))
        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))

        x = self.extract_feat(x)

        scene_names = []
        for i in range(len(batch_data_samples)):
           scene_names.append(batch_data_samples[i].lidar_path)
        x = self.decoder(x, scene_names)

        sp_gt_instances = []
        for i in range(len(batch_data_samples)):
            voxel_superpoints = inverse_mapping[
                coordinates[:, 0][inverse_mapping] == i]
            voxel_superpoints = torch.unique(
                voxel_superpoints, return_inverse=True)[1]
            inst_mask = batch_data_samples[i].gt_pts_seg.pts_instance_mask
            assert voxel_superpoints.shape == inst_mask.shape

            batch_data_samples[i].gt_instances_3d.sp_masks = \
                S3DISOneFormer3D.get_gt_inst_masks(inst_mask, voxel_superpoints)
            sp_gt_instances.append(batch_data_samples[i].gt_instances_3d)

        loss = self.criterion(x, sp_gt_instances)
        return loss
    
    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.
        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instance_3d` and `gt_sem_seg_3d`.
        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample contains 'pred_pts_seg'.
            And the `pred_pts_seg` contains following keys.
                - instance_scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - instance_labels (Tensor): Labels of instances, has a shape
                    (num_instances, )
                - pts_instance_mask (Tensor): Instance mask, has a shape
                    (num_points, num_instances) of type bool.
        """
        
        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'])
        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))

        x = self.extract_feat(x)

        scene_names = []
        for i in range(len(batch_data_samples)):
            scene_names.append(batch_data_samples[i].lidar_path)
        x = self.decoder(x, scene_names)

        results_list = self.predict_by_feat(x, inverse_mapping, scene_names)

        for i, data_sample in enumerate(batch_data_samples):
            data_sample.pred_pts_seg = results_list[i]
        return batch_data_samples

    def predict_by_feat(self, out, superpoints, scene_names):
        """Predict instance masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
            scene_names (List[string]): of len 1, which contain scene name.

        Returns:
            List[PointData]: of len 1 with `pts_instance_mask`, 
                `instance_labels`, `instance_scores`.
        """
        pred_labels = out['cls_preds']
        pred_masks = out['masks']
        pred_scores = out['scores']
        scene_name = scene_names[0]

        scores = F.softmax(pred_labels[0], dim=-1)[:, :-1]
        scores *= pred_scores[0]

        if self.prefix_1dataset in scene_name:
            labels = torch.arange(
                self.num_classes_1dataset,
                device=scores.device).unsqueeze(0).repeat(
                    self.decoder.num_queries_1dataset,  
                    1).flatten(0, 1)
        elif self.prefix_2dataset in scene_name:
            labels = torch.arange(
                self.num_classes_2dataset,
                device=scores.device).unsqueeze(0).repeat(
                    self.decoder.num_queries_2dataset,
                    1).flatten(0, 1)          
        else:
            raise RuntimeError(f'Invalid scene name "{scene_name}".')
        
        scores, topk_idx = scores.flatten(0, 1).topk(
            self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]

        if self.prefix_1dataset in scene_name:
            topk_idx = torch.div(topk_idx, self.num_classes_1dataset, 
                                 rounding_mode='floor')
        elif self.prefix_2dataset in scene_name:
            topk_idx = torch.div(topk_idx, self.num_classes_2dataset,
                                 rounding_mode='floor')        
        else:
            raise RuntimeError(f'Invalid scene name "{scene_name}".')
        
        mask_pred = pred_masks[0]
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        if self.test_cfg.get('obj_normalization', None):
            mask_pred_thr = mask_pred_sigmoid > \
                self.test_cfg.obj_normalization_thr
            mask_scores = (mask_pred_sigmoid * mask_pred_thr).sum(1) / \
                (mask_pred_thr.sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel=kernel)

        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr
        mask_pred = mask_pred[:, superpoints]
        # score_thr
        score_mask = scores > self.test_cfg.score_thr
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return [
            PointData(
                pts_instance_mask=mask_pred,
                instance_labels=labels,
                instance_scores=scores)
        ]
