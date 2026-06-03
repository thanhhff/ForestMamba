import numpy as np
import scipy
import torch
from torch_scatter import scatter_mean
from mmcv.transforms import BaseTransform
from mmdet3d.datasets.transforms import PointSample

from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class ElasticTransfrom(BaseTransform):
    """Apply elastic augmentation to a 3D scene. Required Keys:

    Args:
        gran (List[float]): Size of the noise grid (in same scale[m/cm]
            as the voxel grid).
        mag (List[float]): Noise multiplier.
        voxel_size (float): Voxel size.
        p (float): probability of applying this transform.
    """

    def __init__(self, gran, mag, voxel_size, p=1.0):
        self.gran = gran
        self.mag = mag
        self.voxel_size = voxel_size
        self.p = p

    def transform(self, input_dict):
        """Private function-wrapper for elastic transform.

        Args:
            input_dict (dict): Result dict from loading pipeline.
        
        Returns:
            dict: Results after elastic, 'points' is updated
            in the result dict.
        """
        coords = input_dict['points'].tensor[:, :3].numpy() / self.voxel_size
        if np.random.rand() < self.p:
            coords = self.elastic(coords, self.gran[0], self.mag[0])
            coords = self.elastic(coords, self.gran[1], self.mag[1])
        input_dict['elastic_coords'] = coords
        return input_dict

    def elastic(self, x, gran, mag):
        """Private function for elastic transform to a points.

        Args:
            x (ndarray): Point cloud.
            gran (List[float]): Size of the noise grid (in same scale[m/cm]
                as the voxel grid).
            mag: (List[float]): Noise multiplier.
        
        Returns:
            dict: Results after elastic, 'points' is updated
                in the result dict.
        """
        blur0 = np.ones((3, 1, 1)).astype('float32') / 3
        blur1 = np.ones((1, 3, 1)).astype('float32') / 3
        blur2 = np.ones((1, 1, 3)).astype('float32') / 3

        noise_dim = np.abs(x).max(0).astype(np.int32) // gran + 3
        noise = [
            np.random.randn(noise_dim[0], noise_dim[1],
                            noise_dim[2]).astype('float32') for _ in range(3)
        ]

        for blur in [blur0, blur1, blur2, blur0, blur1, blur2]:
            noise = [
                scipy.ndimage.filters.convolve(
                    n, blur, mode='constant', cval=0) for n in noise
            ]

        ax = [
            np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in noise_dim
        ]
        interp = [
            scipy.interpolate.RegularGridInterpolator(
                ax, n, bounds_error=0, fill_value=0) for n in noise
        ]

        return x + np.hstack([i(x)[:, None] for i in interp]) * mag

@TRANSFORMS.register_module()
class AddSuperPointAnnotations(BaseTransform):
    """Prepare ground truth markup for training.
    
    Required Keys:
    - pts_semantic_mask (np.float32)
    
    Added Keys:
    - gt_sp_masks (np.int64)
    
    Args:
        num_classes (int): Number of classes.
    """
    
    def __init__(self,
                 num_classes,
                 stuff_classes,
                 merge_non_stuff_cls=True):
        self.num_classes = num_classes
        self.stuff_classes = stuff_classes
        self.merge_non_stuff_cls = merge_non_stuff_cls
 
    def transform(self, input_dict):
        """Private function for preparation ground truth 
        markup for training.
        
        Args:
            input_dict (dict): Result dict from loading pipeline.
        
        Returns:
            dict: results, 'gt_sp_masks' is added.
        """
        # create class mapping
        # because pts_instance_mask contains instances from non-instaces classes
        pts_instance_mask = torch.tensor(input_dict['pts_instance_mask'])
        pts_semantic_mask = torch.tensor(input_dict['pts_semantic_mask'])
        
        pts_instance_mask[pts_semantic_mask == self.num_classes] = -1
        for stuff_cls in self.stuff_classes:
            pts_instance_mask[pts_semantic_mask == stuff_cls] = -1
        
        idxs = torch.unique(pts_instance_mask)
        assert idxs[0] == -1

        mapping = torch.zeros(torch.max(idxs) + 2, dtype=torch.long)
        new_idxs = torch.arange(len(idxs), device=idxs.device)
        mapping[idxs] = new_idxs - 1
        pts_instance_mask = mapping[pts_instance_mask]
        input_dict['pts_instance_mask'] = pts_instance_mask.numpy()


        # create gt instance markup     
        insts_mask = pts_instance_mask.clone()
        
        if torch.sum(insts_mask == -1) != 0:
            insts_mask[insts_mask == -1] = torch.max(insts_mask) + 1
            insts_mask = torch.nn.functional.one_hot(insts_mask)[:, :-1]
        else:
            insts_mask = torch.nn.functional.one_hot(insts_mask)

        if insts_mask.shape[1] != 0:
            insts_mask = insts_mask.T
            sp_pts_mask = torch.tensor(input_dict['sp_pts_mask'])
            sp_masks_inst = scatter_mean(
                insts_mask.float(), sp_pts_mask, dim=-1)
            sp_masks_inst = sp_masks_inst > 0.5
        else:
            sp_masks_inst = insts_mask.new_zeros(
                (0, input_dict['sp_pts_mask'].max() + 1), dtype=torch.bool)

        num_stuff_cls = len(self.stuff_classes)
        insts = new_idxs[1:] - 1
        if self.merge_non_stuff_cls:
            gt_labels = insts.new_zeros(len(insts) + num_stuff_cls + 1)
        else:
            gt_labels = insts.new_zeros(len(insts) + self.num_classes + 1)

        for inst in insts:
            index = pts_semantic_mask[pts_instance_mask == inst][0]
            gt_labels[inst] = index - num_stuff_cls
        
        input_dict['gt_labels_3d'] = gt_labels.numpy()

        # create gt semantic markup
        sem_mask = torch.tensor(input_dict['pts_semantic_mask'])
        sem_mask = torch.nn.functional.one_hot(sem_mask, 
                                    num_classes=self.num_classes + 1)
       
        sem_mask = sem_mask.T
        sp_pts_mask = torch.tensor(input_dict['sp_pts_mask'])
        sp_masks_seg = scatter_mean(sem_mask.float(), sp_pts_mask, dim=-1)
        sp_masks_seg = sp_masks_seg > 0.5

        sp_masks_seg[-1, sp_masks_seg.sum(axis=0) == 0] = True

        assert sp_masks_seg.sum(axis=0).max().item()
        
        if self.merge_non_stuff_cls:
            sp_masks_seg = torch.vstack((
                sp_masks_seg[:num_stuff_cls, :], 
                sp_masks_seg[num_stuff_cls:, :].sum(axis=0).unsqueeze(0)))
        
        sp_masks_all = torch.vstack((sp_masks_inst, sp_masks_seg))

        input_dict['gt_sp_masks'] = sp_masks_all.numpy()

        # create eval markup
        if 'eval_ann_info' in input_dict.keys(): 
            pts_instance_mask[pts_instance_mask != -1] += num_stuff_cls
            for idx, stuff_cls in enumerate(self.stuff_classes):
                pts_instance_mask[pts_semantic_mask == stuff_cls] = idx

            input_dict['eval_ann_info']['pts_instance_mask'] = \
                pts_instance_mask.numpy()

        return input_dict


@TRANSFORMS.register_module()
class SwapChairAndFloor(BaseTransform):
    """Swap two categories for ScanNet200 dataset. It is convenient for
    panoptic evaluation. After this swap first two categories are
    `stuff` and other 198 are `thing`.
    """
    def transform(self, input_dict):
        """Private function-wrapper for swap transform.

        Args:
            input_dict (dict): Result dict from loading pipeline.
        
        Returns:
            dict: Results after swap, 'pts_semantic_mask' is updated
                in the result dict.
        """
        mask = input_dict['pts_semantic_mask'].copy()
        mask[input_dict['pts_semantic_mask'] == 2] = 3
        mask[input_dict['pts_semantic_mask'] == 3] = 2
        input_dict['pts_semantic_mask'] = mask
        if 'eval_ann_info' in input_dict:
            input_dict['eval_ann_info']['pts_semantic_mask'] = mask
        return input_dict


@TRANSFORMS.register_module()
class PointInstClassMapping_(BaseTransform):
    """Delete instances from non-instaces classes.

    Required Keys:
    - pts_instance_mask (np.float32)
    - pts_semantic_mask (np.float32)

    Modified Keys:
    - pts_instance_mask (np.float32)
    - pts_semantic_mask (np.float32)

    Added Keys:
    - gt_labels_3d (int)

    Args:
        num_classes (int): Number of classes.
    """

    def __init__(self, num_classes, structured3d=False):
        self.num_classes = num_classes
        self.structured3d = structured3d

    def transform(self, input_dict):
        """Private function for deleting 
            instances from non-instaces classes.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: results, 'pts_instance_mask', 'pts_semantic_mask',
            are updated in the result dict. 'gt_labels_3d' is added.
        """

        # because pts_instance_mask contains instances from non-instaces 
        # classes
        pts_instance_mask = np.array(input_dict['pts_instance_mask'])
        pts_semantic_mask = input_dict['pts_semantic_mask']

        if self.structured3d:
            # wall as one instance
            pts_instance_mask[pts_semantic_mask == 0] = \
                pts_instance_mask.max() + 1
            # floor as one instance
            pts_instance_mask[pts_semantic_mask == 1] = \
                pts_instance_mask.max() + 1
        
        pts_instance_mask[pts_semantic_mask == self.num_classes] = -1
        pts_semantic_mask[pts_semantic_mask == self.num_classes] = -1

        idxs = np.unique(pts_instance_mask)
        mapping = np.zeros(np.max(idxs) + 2, dtype=int)
        new_idxs = np.arange(len(idxs))
        if idxs[0] == -1:
            mapping[idxs] = new_idxs - 1
            new_idxs = new_idxs[:-1]
        else:
            mapping[idxs] = new_idxs
        pts_instance_mask = mapping[pts_instance_mask]

        input_dict['pts_instance_mask'] = pts_instance_mask
        input_dict['pts_semantic_mask'] = pts_semantic_mask

        gt_labels = np.zeros(len(new_idxs), dtype=int)
        for inst in new_idxs:
            gt_labels[inst] = pts_semantic_mask[pts_instance_mask == inst][0]

        input_dict['gt_labels_3d'] = gt_labels

        return input_dict

@TRANSFORMS.register_module()
class PointSample_(PointSample):

    def _points_random_sampling(self, points, num_samples):
        """Points random sampling. Sample points to a certain number.
        
        Args:
            points (:obj:`BasePoints`): 3D Points.
            num_samples (int): Number of samples to be sampled.

        Returns:
            tuple[:obj:`BasePoints`, np.ndarray] | :obj:`BasePoints`:
                - points (:obj:`BasePoints`): 3D Points.
                - choices (np.ndarray, optional): The generated random samples.
        """

        choices = np.random.choice(len(points), min(num_samples, len(points)))
        
        return points[choices], choices

    def transform(self, input_dict):
        """Transform function to sample points to in indoor scenes.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after sampling, 'points', 'pts_instance_mask',
            'pts_semantic_mask', sp_pts_mask' keys are updated in the 
            result dict.
        """
        points = input_dict['points']

        # if point number smaller than num_point, skip
        if len(points) < self.num_points:
            return input_dict

        points, choices = self._points_random_sampling(
            points, self.num_points)
        input_dict['points'] = points
        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        pts_semantic_mask = input_dict.get('pts_semantic_mask', None)
        vote_label = input_dict.get('vote_label', None)
        instance_mask = input_dict.get('instance_mask', None)
        sp_pts_mask = input_dict.get('sp_pts_mask', None)

        if pts_instance_mask is not None:
            pts_instance_mask = pts_instance_mask[choices]
            
            idxs = np.unique(pts_instance_mask)
            mapping = np.zeros(np.max(idxs) + 2, dtype=int)
            new_idxs = np.arange(len(idxs))
            if idxs[0] == -1:
                mapping[idxs] = new_idxs - 1
            else:
                mapping[idxs] = new_idxs
            pts_instance_mask = mapping[pts_instance_mask]

            input_dict['pts_instance_mask'] = pts_instance_mask

        if pts_semantic_mask is not None:
            pts_semantic_mask = pts_semantic_mask[choices]
            input_dict['pts_semantic_mask'] = pts_semantic_mask

        if vote_label is not None:
            vote_label = vote_label[choices]
            input_dict['vote_label'] = vote_label

        if instance_mask is not None:
            instance_mask = instance_mask[choices]
            input_dict['instance_mask'] = instance_mask

        if sp_pts_mask is not None:
            sp_pts_mask = sp_pts_mask[choices]
            sp_pts_mask = np.unique(
                sp_pts_mask, return_inverse=True)[1]
            input_dict['sp_pts_mask'] = sp_pts_mask

        if 'eval_ann_info' in input_dict:
            input_dict['eval_ann_info']['pts_instance_mask'] = pts_instance_mask
            input_dict['eval_ann_info']['pts_semantic_mask'] = pts_semantic_mask
            input_dict['eval_ann_info']['instance_mask'] = instance_mask
            
        return input_dict
    
@TRANSFORMS.register_module()
class SkipEmptyScene(BaseTransform):
    """Skip empty scene during training.

    Required Keys:
    - pts_instance_mask (np.float32)
    - pts_semantic_mask (np.float32)
    - points (:obj:`BasePoints`)
    - gt_labels_3d (int)

    Modified Keys:
    - pts_instance_mask (np.float32)
    - pts_semantic_mask (np.float32)
    - points (:obj:`BasePoints`)
    - gt_labels_3d (int)

    """

    def transform(self, input_dict):
        """Private function for skipping empty scene during training.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: results, 'pts_instance_mask', 'pts_semantic_mask',
            'points', 'gt_labels_3d' are updated in the result dict.
        """

        if len(input_dict['gt_labels_3d']) != 0:
            self.inst = input_dict['pts_instance_mask']
            self.sem = input_dict['pts_semantic_mask']
            self.gt_labels = input_dict['gt_labels_3d']
            self.points = input_dict['points']
        else:
            input_dict['pts_instance_mask'] = self.inst
            input_dict['pts_semantic_mask'] = self.sem 
            input_dict['gt_labels_3d'] = self.gt_labels
            input_dict['points'] = self.points

        return input_dict

@TRANSFORMS.register_module()
class SkipEmptyScene_(BaseTransform):
    """Skip empty scene during training.

    Required Keys:
    - pts_instance_mask (np.float32)
    - pts_semantic_mask (np.float32)
    - points (:obj:`BasePoints`)
    - gt_labels_3d (int)

    Modified Keys:
    - pts_instance_mask (np.float32)
    - pts_semantic_mask (np.float32)
    - points (:obj:`BasePoints`)
    - gt_labels_3d (int)

    """

    def transform(self, input_dict):
        """Private function for skipping empty scene during training.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: results, 'pts_instance_mask', 'pts_semantic_mask',
            'points', 'gt_labels_3d' are updated in the result dict.
        """

        if len(input_dict["points"]) == 0:
            return None
        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        if len(np.unique(pts_instance_mask)) < 2:
            return None

        return input_dict

@TRANSFORMS.register_module()
class CylinderCrop(BaseTransform):
    def __init__(self, radius=8):
        self.radius = radius

    def transform(self, input_dict):

        assert "points" in input_dict.keys()
        
        # Get the tensor of points
        points_tensor = input_dict["points"].tensor.numpy()
        
        # Select a random center point
        center = points_tensor[np.random.randint(points_tensor.shape[0])]
        
        # Calculate indices of points within the radius
        choices = np.where(
            (np.sum(np.square(points_tensor[:, :2] - center[:2]), 1) < self.radius**2)
        )[0]
        
        # Update points tensor
        if "points" in input_dict.keys():
            input_dict["points"] = input_dict["points"][choices]
        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        pts_semantic_mask = input_dict.get('pts_semantic_mask', None)
        sp_pts_mask = input_dict.get('sp_pts_mask', None)

        # Initialize the instance mask
        instance_mask = pts_semantic_mask != 0  # Background points have -1 in pts_instance_mask
        pts_instance_mask = pts_instance_mask.copy()
        pts_instance_mask[~instance_mask] = -1
        
        if pts_instance_mask is not None:
            original_pts_instance_mask = pts_instance_mask
            pts_instance_mask = pts_instance_mask[choices]

            idxs = np.unique(pts_instance_mask)
            mapping = np.zeros(np.max(idxs) + 2, dtype=int)
            new_idxs = np.arange(len(idxs))
            if idxs[0] == -1:
                mapping[idxs] = new_idxs - 1
            else:
                mapping[idxs] = new_idxs
            pts_instance_mask = mapping[pts_instance_mask]
            input_dict['pts_instance_mask'] = pts_instance_mask

            # Initialize vote_label
            vote_label = np.empty((len(choices), 3))
            vote_label[:] = np.nan

            fg_idxs = idxs[idxs != -1]  # foreground instance IDs in original space

            if len(fg_idxs) > 0:
                # --- Vectorized ratio_inspoint ---
                non_bg_orig = original_pts_instance_mask != -1
                u_orig, c_orig = np.unique(original_pts_instance_mask[non_bg_orig], return_counts=True)
                orig_count_dict = dict(zip(u_orig.tolist(), c_orig.tolist()))

                non_bg_crop = pts_instance_mask != -1
                if non_bg_crop.any():
                    u_crop, c_crop = np.unique(pts_instance_mask[non_bg_crop], return_counts=True)
                    new_count_dict = dict(zip(u_crop.tolist(), c_crop.tolist()))
                else:
                    new_count_dict = {}

                ratio_inspoint = {}
                for idx in fg_idxs:
                    mapped = int(mapping[idx])
                    orig_c = orig_count_dict.get(int(idx), 0)
                    new_c = new_count_dict.get(mapped, 0)
                    ratio_inspoint[mapped] = new_c / orig_c if orig_c > 0 else 0

                # --- Vectorized vote_label using reduceat ---
                # Sort original foreground points by instance ID once
                orig_fg_idx = np.where(non_bg_orig)[0]
                orig_fg_ids = original_pts_instance_mask[orig_fg_idx]
                sort_order = np.argsort(orig_fg_ids, kind='stable')
                sorted_ids = orig_fg_ids[sort_order]
                sorted_pts = points_tensor[orig_fg_idx[sort_order], :3]

                u_sorted, first_occ = np.unique(sorted_ids, return_index=True)
                inst_max = np.maximum.reduceat(sorted_pts, first_occ)  # (n_inst, 3)
                inst_min = np.minimum.reduceat(sorted_pts, first_occ)  # (n_inst, 3)
                inst_centers = 0.5 * (inst_max + inst_min)             # (n_inst, 3)

                # LUT: original instance ID -> center row index
                center_lut = np.full(int(u_sorted.max()) + 1, -1, dtype=np.intp)
                center_lut[u_sorted] = np.arange(len(u_sorted), dtype=np.intp)

                # Assign vote labels for all cropped foreground points at once
                crop_orig_ids = original_pts_instance_mask[choices]
                crop_fg_mask = crop_orig_ids != -1
                if crop_fg_mask.any():
                    cidxs = center_lut[crop_orig_ids[crop_fg_mask]]
                    vote_label[crop_fg_mask] = inst_centers[cidxs] - points_tensor[choices[crop_fg_mask], :3]
            else:
                ratio_inspoint = {}
            
            
            input_dict['ratio_inspoint'] = ratio_inspoint
            input_dict['vote_label'] = torch.tensor(vote_label, dtype=torch.float32)
            input_dict['instance_mask'] = torch.tensor(instance_mask, dtype=torch.bool)

        if pts_semantic_mask is not None:
            pts_semantic_mask = pts_semantic_mask[choices]
            input_dict['pts_semantic_mask'] = pts_semantic_mask

        if instance_mask is not None:
            instance_mask = instance_mask[choices]
            input_dict['instance_mask'] = instance_mask 

        if sp_pts_mask is not None:
            sp_pts_mask = sp_pts_mask[choices]
            sp_pts_mask = np.unique(
                sp_pts_mask, return_inverse=True)[1]
            input_dict['sp_pts_mask'] = sp_pts_mask

        if 'eval_ann_info' in input_dict:
            input_dict['eval_ann_info']['pts_instance_mask'] = pts_instance_mask
            input_dict['eval_ann_info']['pts_semantic_mask'] = pts_semantic_mask
            input_dict['eval_ann_info']['instance_mask'] = instance_mask

        return input_dict


@TRANSFORMS.register_module()
class CylinderCrop_RemoveOutpoints(BaseTransform):
    def __init__(self, radius=8):
        self.radius = radius

    def transform(self, input_dict):
        assert "points" in input_dict.keys()

        # Get the tensor of points
        points_tensor = input_dict["points"].tensor.numpy()

        # Select a random center point
        center = points_tensor[np.random.randint(points_tensor.shape[0])]

        # Calculate indices of points within the radius
        choices = np.where(
            (np.sum(np.square(points_tensor[:, :2] - center[:2]), 1) < self.radius**2)
        )[0]

        # Update the points tensor
        if "points" in input_dict.keys():
            input_dict["points"] = input_dict["points"][choices]
        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        pts_semantic_mask = input_dict.get('pts_semantic_mask', None)
        sp_pts_mask = input_dict.get('sp_pts_mask', None)

        # Initialize the instance mask based on semantic labels
        instance_mask = pts_semantic_mask != 0  # Background points have -1 in pts_instance_mask
        pts_instance_mask = pts_instance_mask.copy()
        pts_instance_mask[~instance_mask] = -1  # Set background points to -1 in instance mask
        

        if pts_instance_mask is not None:
            original_pts_instance_mask = pts_instance_mask
            pts_instance_mask = pts_instance_mask[choices]  # Only work with selected points after cropping

            # Initialize vote_label and ratio_inspoint
            vote_label = np.empty((len(choices), 3))
            vote_label[:] = np.nan  # Set initial values to NaN for easier debugging
            ratio_inspoint = {}

            # Find all unique instance indices after cropping
            valid_choices = []
            unique_instances = np.unique(pts_instance_mask)

            for instance in unique_instances:
                if instance != -1:  # Only process non-background instances
                    original_count = np.sum(original_pts_instance_mask == instance)
                    new_count = np.sum(pts_instance_mask == instance)

                    if original_count > 0:
                        ratio = new_count / original_count
                    else:
                        ratio = 0

                    if ratio == 1:  # Only keep fully contained instances
                        ratio_inspoint[instance] = ratio
                        valid_choices.extend(np.where(pts_instance_mask == instance)[0])

                        # Calculate the vote_label for the instance
                        ind = np.where(original_pts_instance_mask == instance)[0]
                        if len(ind) > 0:
                            pos = points_tensor[ind, :3]
                            max_pos = pos.max(0)
                            min_pos = pos.min(0)
                            center = 0.5 * (min_pos + max_pos)

                            # Find the points in the cylinder that belong to this instance
                            cylinder_ind = np.where(pts_instance_mask == instance)[0]
                            vote_label[cylinder_ind, :] = center - points_tensor[choices[cylinder_ind], :3]

            # Add background points to valid choices
            background_choices = np.where(pts_instance_mask == -1)[0]
            valid_choices = np.array(valid_choices + background_choices.tolist())

            # Filter instance_mask and create new continuous instance labels
            filtered_instance_mask = pts_instance_mask[valid_choices]
            valid_instance_ids = np.unique(filtered_instance_mask)

            # Create a mapping from original instance IDs to new continuous instance labels, keeping -1 as background
            instance_mapping = {-1: -1}
            non_background_instances = [inst for inst in valid_instance_ids if inst != -1]
            instance_mapping.update({old_id: new_id for new_id, old_id in enumerate(non_background_instances)})

            # Apply the new mapping to filtered_instance_mask
            new_instance_mask = np.vectorize(instance_mapping.get)(filtered_instance_mask)

            # Update ratio_inspoint with new continuous instance labels
            new_ratio_inspoint = {instance_mapping[inst]: ratio_inspoint[inst] for inst in valid_instance_ids if inst in ratio_inspoint}

            # Update input_dict with valid choices
            input_dict['points'] = input_dict['points'][valid_choices]
            input_dict['pts_instance_mask'] = new_instance_mask  
            input_dict['vote_label'] = torch.tensor(vote_label[valid_choices], dtype=torch.float32)
            input_dict['instance_mask'] = instance_mask[choices][valid_choices]
            input_dict['ratio_inspoint'] = new_ratio_inspoint 

        # Update semantic mask if present
        if pts_semantic_mask is not None:
            input_dict['pts_semantic_mask'] = pts_semantic_mask[choices][valid_choices]

        # Update superpoint mask if present
        if sp_pts_mask is not None:
            sp_pts_mask = sp_pts_mask[choices][valid_choices]
            sp_pts_mask = np.unique(sp_pts_mask, return_inverse=True)[1]
            input_dict['sp_pts_mask'] = sp_pts_mask

        # Update evaluation annotation info if present
        if 'eval_ann_info' in input_dict:
            input_dict['eval_ann_info']['pts_instance_mask'] = input_dict['pts_instance_mask']
            input_dict['eval_ann_info']['pts_semantic_mask'] = input_dict['pts_semantic_mask']
            input_dict['eval_ann_info']['instance_mask'] = input_dict['instance_mask']

        return input_dict



@TRANSFORMS.register_module()
class GridSample(BaseTransform):
    def __init__(self, grid_size=0.2, mode="train", hash_type="fnv"):
        self.grid_size = grid_size
        self.mode = mode
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec

    def transform(self, input_dict):
        assert "points" in input_dict.keys()
        points = input_dict["points"]
        
        scaled_points = points.tensor / self.grid_size
        grid_points = torch.floor(scaled_points).int()
        min_points = torch.min(grid_points, dim=0).values
        grid_points -= min_points
        scaled_points -= min_points
        min_points = min_points * self.grid_size

        key = self.hash(grid_points)
        idx_sort = torch.argsort(key)
        key_sort = key[idx_sort]
        unique_results = torch.unique(key_sort, return_inverse=True, return_counts=True)
        _, inverse, count = unique_results

        if self.mode == "train":  # train mode
            idx_select = (
                torch.cumsum(torch.cat((torch.tensor([0]), count[:-1])), dim=0)
                + torch.randint(0, count.max(), count.size()) % count
            )
            choices = idx_sort[idx_select]
        else:
            raise NotImplementedError("Only train mode is implemented in this example")

        # Subsampled data
        input_dict["points"] = points[choices]

        #print(input_dict["points"].shape)
        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        pts_semantic_mask = input_dict.get('pts_semantic_mask', None)
        vote_label = input_dict.get('vote_label', None)
        instance_mask = input_dict.get('instance_mask', None)
        sp_pts_mask = input_dict.get('sp_pts_mask', None)

        if pts_instance_mask is not None:
            pts_instance_mask = pts_instance_mask[choices]
            
            idxs = np.unique(pts_instance_mask)
            mapping = np.zeros(np.max(idxs) + 2, dtype=int)
            new_idxs = np.arange(len(idxs))
            if idxs[0] == -1:
                mapping[idxs] = new_idxs - 1
            else:
                mapping[idxs] = new_idxs
            pts_instance_mask = mapping[pts_instance_mask]

            input_dict['pts_instance_mask'] = pts_instance_mask

        if pts_semantic_mask is not None:
            pts_semantic_mask = pts_semantic_mask[choices]
            input_dict['pts_semantic_mask'] = pts_semantic_mask

        if vote_label is not None:
            vote_label = vote_label[choices]
            input_dict['vote_label'] = vote_label

        if instance_mask is not None:
            instance_mask = instance_mask[choices]
            input_dict['instance_mask'] = instance_mask

        if sp_pts_mask is not None:
            sp_pts_mask = sp_pts_mask[choices]
            sp_pts_mask = np.unique(
                sp_pts_mask, return_inverse=True)[1]
            input_dict['sp_pts_mask'] = sp_pts_mask

        if 'eval_ann_info' in input_dict:
            input_dict['eval_ann_info']['pts_instance_mask'] = pts_instance_mask
            input_dict['eval_ann_info']['pts_semantic_mask'] = pts_semantic_mask
            input_dict['eval_ann_info']['instance_mask'] = instance_mask

        return input_dict

    def fnv_hash_vec(self, vec):
        # Use smaller values to avoid overflow issues
        FNV_prime = torch.tensor(16777619, dtype=torch.int64)
        offset_basis = torch.tensor(2166136261, dtype=torch.int64)
        hash = torch.full((vec.shape[0],), offset_basis, dtype=torch.int64)
        for i in range(vec.shape[1]):
            hash = hash ^ vec[:, i].to(torch.int64)
            hash = hash * FNV_prime
        return hash

    def ravel_hash_vec(self, vec):
        # Implement the ravel hash function for vectors
        vec_max = torch.max(vec, dim=0).values + 1
        hash = torch.ravel_multi_index(vec.t(), vec_max)
        return hash


@TRANSFORMS.register_module()
class GeometricFeatureAugmentation(BaseTransform):
    """Wood-Aware Geometric Augmentation (training only).

    Uses local geometry to detect stem/wood point candidates — those with
    high linearity (λ₁ >> λ₂ ≈ λ₃, i.e. elongated neighbourhood) AND high
    verticality (principal axis ≈ [0,0,1]) — and oversamples them with small
    XY jitter to address the severe wood/leaf class imbalance in forest LiDAR.

    This is a **true augmentation**: it does NOT add channels to the point
    tensor.  ``in_channels`` stays at 3 (XYZ) and the transform is safe to
    omit at val/test time without any architectural mismatch.

    Required Keys:
        - points (:obj:`BasePoints`): XYZ in the first 3 dims.
        - pts_semantic_mask (np.ndarray): Per-point semantic label (N,).
        - pts_instance_mask (np.ndarray): Per-point instance label (N,).

    Modified Keys:
        - points, pts_semantic_mask, pts_instance_mask — extended with
          oversampled wood candidates.

    Args:
        k (int): Neighbours for local covariance estimation. Default: 16.
        linearity_thr (float): Minimum linearity score to be a wood
            candidate. Default: 0.7.
        verticality_thr (float): Minimum verticality score to be a wood
            candidate. Default: 0.6.
        oversample_ratio (float): Multiplier on the wood point count.
            ``2.0`` doubles the number of wood points. Default: 2.0.
        jitter_std (float): Std (metres) of Gaussian XY noise added to
            each duplicated point. Z is left unchanged to preserve height
            structure. Default: 0.05.
        chunk_size (int): Points processed per chunk to bound RAM usage.
            Default: 50000.
    """

    def __init__(
        self,
        k: int = 16,
        linearity_thr: float = 0.7,
        verticality_thr: float = 0.6,
        oversample_ratio: float = 2.0,
        jitter_std: float = 0.05,
        chunk_size: int = 50000,
    ):
        self.k = k
        self.linearity_thr = linearity_thr
        self.verticality_thr = verticality_thr
        self.oversample_ratio = oversample_ratio
        self.jitter_std = jitter_std
        self.chunk_size = chunk_size

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _compute_linearity_verticality(self, xyz: np.ndarray) -> tuple:
        """Chunked computation of linearity and verticality.

        The KDTree is built once over all points. Intermediate arrays
        are allocated one chunk at a time to keep peak RAM bounded.

        Args:
            xyz (np.ndarray): Float array of shape (N, 3).

        Returns:
            tuple: Two float32 arrays of shape (N,) —
                ``(linearity, verticality)``.
        """
        from scipy.spatial import cKDTree

        N = xyz.shape[0]
        k_query = min(self.k + 1, N)  # +1 because query includes self

        tree = cKDTree(xyz)

        linearity   = np.empty(N, dtype=np.float32)
        verticality = np.empty(N, dtype=np.float32)

        for start in range(0, N, self.chunk_size):
            end = min(start + self.chunk_size, N)

            _, indices = tree.query(xyz[start:end], k=k_query, workers=-1)

            neighbors = xyz[indices]                          # (C, k, 3)
            mu        = neighbors.mean(axis=1, keepdims=True) # (C, 1, 3)
            diff      = neighbors - mu                        # (C, k, 3)
            cov       = np.einsum('nki,nkj->nij', diff, diff) / k_query

            eigvals, eigvecs = np.linalg.eigh(cov)  # ascending order

            l1 = eigvals[:, 2]
            l2 = eigvals[:, 1]
            e1 = eigvecs[:, :, 2]  # principal eigenvector

            safe_l1 = np.where(l1 < 1e-10, 1e-10, l1)

            linearity[start:end]   = np.clip((l1 - l2) / safe_l1, 0.0, 1.0)
            verticality[start:end] = np.clip(np.abs(e1[:, 2]),     0.0, 1.0)

        return linearity, verticality

    # ------------------------------------------------------------------
    # Transform entry point
    # ------------------------------------------------------------------

    def transform(self, input_dict: dict) -> dict:
        """Oversample wood-candidate points with small XY jitter.

        Args:
            input_dict (dict): Pipeline result dict.

        Returns:
            dict: Same dict with points and masks extended by oversampled
                wood candidates.
        """
        points   = input_dict['points']
        xyz      = points.tensor[:, :3].numpy()  # (N, 3)

        linearity, verticality = self._compute_linearity_verticality(xyz)

        # Wood candidates: highly linear AND highly vertical
        wood_mask = (linearity >= self.linearity_thr) & \
                    (verticality >= self.verticality_thr)
        wood_idx  = np.where(wood_mask)[0]

        if wood_idx.size == 0:
            return input_dict  # no candidates — skip augmentation

        # Number of extra points to add
        n_extra = int((self.oversample_ratio - 1.0) * wood_idx.size)
        if n_extra <= 0:
            return input_dict

        # Sample with replacement from wood candidates
        sampled_idx = np.random.choice(wood_idx, size=n_extra, replace=True)

        # Duplicate point tensor and add XY jitter (Z unchanged)
        extra_pts = points.tensor[sampled_idx].clone()          # (n_extra, D)
        jitter    = torch.zeros_like(extra_pts[:, :3])
        jitter[:, :2] = torch.randn(n_extra, 2) * self.jitter_std
        extra_pts[:, :3] += jitter

        new_tensor = torch.cat([points.tensor, extra_pts], dim=0)
        new_points = type(points)(
            new_tensor,
            points_dim=points.tensor.shape[1],
            attribute_dims=points.attribute_dims)
        input_dict['points'] = new_points

        # Propagate per-point masks for the duplicated points
        for key in ('pts_semantic_mask', 'pts_instance_mask'):
            if key in input_dict and input_dict[key] is not None:
                mask = input_dict[key]
                extra = mask[sampled_idx]
                input_dict[key] = np.concatenate([mask, extra], axis=0)

        # instance_mask is a boolean foreground flag (torch.bool or np.bool_)
        # created by CylinderCrop and subsampled by PointSample_ — must stay
        # in sync with the point count.
        if 'instance_mask' in input_dict and input_dict['instance_mask'] is not None:
            inst_mask = input_dict['instance_mask']
            if isinstance(inst_mask, torch.Tensor):
                extra_inst = inst_mask[sampled_idx]
                input_dict['instance_mask'] = torch.cat([inst_mask, extra_inst], dim=0)
            else:
                extra_inst = inst_mask[sampled_idx]
                input_dict['instance_mask'] = np.concatenate([inst_mask, extra_inst], axis=0)

        return input_dict