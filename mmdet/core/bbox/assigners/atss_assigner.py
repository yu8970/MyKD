import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@BBOX_ASSIGNERS.register_module()
class ATSSAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `0` or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        topk (float): number of bbox selected in each level
    """

    def __init__(self,
                 topk,
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 ignore_iof_thr=-1):
        self.topk = topk
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.ignore_iof_thr = ignore_iof_thr

    # https://github.com/sfzhang15/ATSS/blob/master/atss_core/modeling/rpn/atss/loss.py

    def assign(self,
               bboxes,
               num_level_bboxes,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None):
        """Assign gt to bboxes.

        The assignment is done in following steps

        1. compute iou between all bbox (bbox of all pyramid levels) and gt
        2. compute center distance between all bbox and gt
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center, so we total select k*l bbox as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as postive
        6. limit the positive sample's center in gt


        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            num_level_bboxes (List): num of bboxes in each level
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        INF = 100000000
        bboxes = bboxes[:, :4]
        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        # compute iou between all bbox and gt
        overlaps = self.iou_calculator(bboxes, gt_bboxes)

        # assign 0 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             0,
                                             dtype=torch.long)

        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # compute center distance between all bbox and gt
        gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        gt_points = torch.stack((gt_cx, gt_cy), dim=1)

        bboxes_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        bboxes_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        bboxes_points = torch.stack((bboxes_cx, bboxes_cy), dim=1)

        distances = (bboxes_points[:, None, :] -
                     gt_points[None, :, :]).pow(2).sum(-1).sqrt()

        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            ignore_overlaps = self.iou_calculator(
                bboxes, gt_bboxes_ignore, mode='iof')
            ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            ignore_idxs = ignore_max_overlaps > self.ignore_iof_thr
            distances[ignore_idxs, :] = INF
            assigned_gt_inds[ignore_idxs] = -1

        # Selecting candidates based on the center distance
        candidate_idxs = []
        start_idx = 0
        for level, bboxes_per_level in enumerate(num_level_bboxes):
            # on each pyramid level, for each gt,
            # select k bbox whose center are closest to the gt center
            end_idx = start_idx + bboxes_per_level
            distances_per_level = distances[start_idx:end_idx, :]
            selectable_k = min(self.topk, bboxes_per_level)
            _, topk_idxs_per_level = distances_per_level.topk(
                selectable_k, dim=0, largest=False)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = torch.cat(candidate_idxs, dim=0)

        # get corresponding iou for the these candidates, and compute the
        # mean and std, set mean + std as the iou threshold
        candidate_overlaps = overlaps[candidate_idxs, torch.arange(num_gt)]
        overlaps_mean_per_gt = candidate_overlaps.mean(0)
        overlaps_std_per_gt = candidate_overlaps.std(0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]

        # limit the positive sample's center in gt
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        ep_bboxes_cx = bboxes_cx.view(1, -1).expand(
            num_gt, num_bboxes).contiguous().view(-1)
        ep_bboxes_cy = bboxes_cy.view(1, -1).expand(
            num_gt, num_bboxes).contiguous().view(-1)
        candidate_idxs = candidate_idxs.view(-1)

        # calculate the left, top, right, bottom distance between positive
        # bbox center and gt side
        l_ = ep_bboxes_cx[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 0]
        t_ = ep_bboxes_cy[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - ep_bboxes_cx[candidate_idxs].view(-1, num_gt)
        b_ = gt_bboxes[:, 3] - ep_bboxes_cy[candidate_idxs].view(-1, num_gt)
        is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01
        is_pos = is_pos & is_in_gts

        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        overlaps_inf = torch.full_like(overlaps,
                                       -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]

        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()
        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)

        overlaps_inf = torch.full_like(overlaps,
                                       -INF).t().contiguous().view(-1)
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()
        assigned_gt_inds[
            max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1
        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def get_vlr_region(self,
                       bboxes,
                       num_level_bboxes,
                       gt_bboxes,
                       gt_bboxes_ignore=None,
                       gt_labels=None):

        INF = 100000000
        bboxes = bboxes[:, :4]  # 只取其前四个列,即坐标值
        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        # compute iou between all bbox and gt

        overlaps = self.iou_calculator(bboxes, gt_bboxes)           # IOU
        diou = self.iou_calculator(bboxes, gt_bboxes, mode='diou')  # DIOU
        # assign 0 by default
        # 表示每个建议的边界框被分配给哪个真实的边界框
        assigned_gt_inds = overlaps.new_full((num_bboxes, ), 0, dtype=torch.long)
        vlr_region_iou = (assigned_gt_inds + 0).float()
        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ), -1,mdtype=torch.long)
            return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # compute center distance between all bbox and gt
        gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        gt_points = torch.stack((gt_cx, gt_cy), dim=1)

        bboxes_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        bboxes_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        bboxes_points = torch.stack((bboxes_cx, bboxes_cy), dim=1)

        distances = (bboxes_points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()

        # 确保了那些与需要忽略的gt_bbox具有较高IoU的bbox不会被分配给真实的边界框
        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            ignore_overlaps = self.iou_calculator(bboxes, gt_bboxes_ignore, mode='iof')
            ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            ignore_idxs = ignore_max_overlaps > self.ignore_iof_thr
            distances[ignore_idxs, :] = INF
            assigned_gt_inds[ignore_idxs] = -1

        # Selecting candidates based on the center distance
        candidate_idxs = []
        candidate_idxs_t = []

        start_idx = 0
        for level, bboxes_per_level in enumerate(num_level_bboxes):
            # level：代表了当前的金字塔层级索引。
            # bboxes_per_level：代表了当前金字塔层级上的边界框数量。
            # on each pyramid level, for each gt,
            # select k bbox whose center are closest to the gt center
            end_idx = start_idx + bboxes_per_level
            distances_per_level = distances[start_idx:end_idx, :]
            selectable_t = min(self.topk, bboxes_per_level)
            selectable_k = min(bboxes_per_level, bboxes_per_level)
            _, topk_idxs_per_level = distances_per_level.topk(   # 选择与gt中心距离最近的那些bbox的索引
                selectable_k, dim=0, largest=False)
            _, topt_idxs_per_level = distances_per_level.topk(
                selectable_t, dim=0, largest=False)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            candidate_idxs_t.append(topt_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = torch.cat(candidate_idxs, dim=0)
        candidate_idxs_t = torch.cat(candidate_idxs_t, dim=0)

        # get corresponding iou for the these candidates, and compute the
        # mean and std, set mean + std as the iou threshold
        # overlaps: 这是一个矩阵，其形状为(num_bboxes, num_gt)，表示每个bbox与每个gt之间的IoU值。
        # 假设candidate_idxs_t是tensor([1, 4, 7])，并且num_gt是3，那么这个索引操作将会从overlaps矩阵中选择下列的IoU值：(1,0)，(4,1) 和 (7,2)。
        # 总的来说，candidate_overlaps_t是一个1D Tensor，它包含了每个gt与与其中心最近的topk个建议框之间的IoU值。
        candidate_overlaps_t = overlaps[candidate_idxs_t, torch.arange(num_gt)]
        # t_overlaps = overlaps[candidate_idxs_t, torch.arange(num_gt)]
        # candidate_idxs是取了所有bbox的idx，candidate_idxs_t是取了中心距离最近的topk个bbox的索引
        t_diou = diou[candidate_idxs, torch.arange(num_gt)]  # 包含了每个gt与选定的建议框之间的DIoU值。
        overlaps_mean_per_gt = candidate_overlaps_t.mean(0)  # 计算了每个gt与所有建议框之间的IoU值的平均值
        overlaps_std_per_gt = candidate_overlaps_t.std(0)    # 计算了每个gt与所有建议框之间的IoU值的标准差
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        #is_pos = (candidate_overlaps < overlaps_thr_per_gt[None, :]) & (candidate_overlaps >= 0.5*overlaps_thr_per_gt[None, :])
        is_pos = (t_diou < overlaps_thr_per_gt[None, :]) & (t_diou >= 0.25 * overlaps_thr_per_gt[None, :])

        # 目的是更新candidate_idxs矩阵，使其对于每个gt都有正确的引用索引。
        # limit the positive sample's center in gt
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes

        candidate_idxs = candidate_idxs.view(-1) # 将其变为一个一维张量。

        # 创建一个与overlaps形状相同，但所有元素都为-INF的张量，然后将其转置并改变为一维张量。
        overlaps_inf = torch.full_like(overlaps, -INF).t().contiguous().view(-1)
        # index包含了candidate_idxs中那些在is_pos中被标记为True的元素的索引。
        index = candidate_idxs.view(-1)[is_pos.view(-1)]

        # 将overlaps_inf在index位置的值替换为overlaps中对应的值。
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        # 将overlaps_inf的形状更改为(num_gt, some_value)
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()
        # 这在overlaps_inf的第二维（即列）上计算最大值。它返回两个张量：一个是每行的最大值，另一个是每行最大值的索引。
        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)

        overlaps_inf = torch.full_like(overlaps,-INF).t().contiguous().view(-1)
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()
        # assigned_gt_inds中的每个位置将包含匹配到的gt真实框的索引或0（表示未匹配）。
        assigned_gt_inds[max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1
        # 对于那些与真实框有有效交集的预测框，更新其在vlr_region_iou数组中的IoU值。
        vlr_region_iou[max_overlaps != -INF] = max_overlaps[max_overlaps != -INF] + 0
        # vlr_region_iou 存储每个预测的边界框与其匹配的gt边界框之间的IoU（交并比）值。但需要注意，它并不是直接存放原始的IoU值
        return vlr_region_iou
