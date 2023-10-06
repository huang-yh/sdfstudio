import torch, numpy as np
import torch.nn.functional as F
import torch.nn as nn


# This function must use fp32!!!
@torch.cuda.amp.autocast(enabled=False)
def point_sampling(reference_points, img_metas):
    reference_points = reference_points.float()

    lidar2img = []
    for img_meta in img_metas:
        lidar2img.append(img_meta["lidar2img"])
    lidar2img = np.asarray(lidar2img)
    lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)

    reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)  # bs, n, 4

    # reference_points = reference_points.permute(1, 0, 2, 3)
    B, num_query = reference_points.size()[:2]
    num_cam = lidar2img.size(1)

    reference_points = reference_points.view(B, 1, num_query, 4, 1)
    lidar2img = lidar2img.view(B, num_cam, 1, 4, 4)
    reference_points_cam = torch.matmul(lidar2img.to(torch.float32), reference_points.to(torch.float32)).squeeze(
        -1
    )  # bs, N, n, 4

    eps = 1e-5

    reference_points_cam[..., 0:2] = reference_points_cam[..., 0:2] * img_metas[0]["scale_rate"]

    if (
        "img_augmentation" in img_metas[0]
        and "post_rots" in img_metas[0]["img_augmentation"]
        and "post_trans" in img_metas[0]["img_augmentation"]
    ):
        post_rots = []
        post_trans = []
        for img_meta in img_metas:
            post_rots.append(img_meta["img_augmentation"]["post_rots"].numpy())
            post_trans.append(img_meta["img_augmentation"]["post_trans"].numpy())
        post_rots = np.asarray(post_rots)
        post_trans = np.asarray(post_trans)
        post_rots = reference_points.new_tensor(post_rots)
        post_trans = reference_points.new_tensor(post_trans)

        reference_points_cam[..., :2] = reference_points_cam[..., :2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps
        )

        # D, B, N, Q, 3, 1
        reference_points_cam = reference_points_cam[..., :3].unsqueeze(-1)
        post_rots = post_rots.view(1, B, num_cam, 1, 3, 3)
        reference_points_cam = torch.matmul(
            post_rots.to(torch.float32), reference_points_cam.to(torch.float32)
        ).squeeze(-1)
        # D, B, N, Q, 3
        post_trans = post_trans.view(1, B, num_cam, 1, 3)
        reference_points_cam = reference_points_cam + post_trans
        tpv_mask = reference_points_cam[..., 2:3] > eps
        reference_points_cam = reference_points_cam[..., :2]
    else:
        tpv_mask = reference_points_cam[..., 2:3] > eps
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps
        )

    reference_points_cam[..., 0] /= img_metas[0]["img_shape"][0][1]
    reference_points_cam[..., 1] /= img_metas[0]["img_shape"][0][0]

    tpv_mask = (
        tpv_mask
        & (reference_points_cam[..., 1:2] > 0.0)
        & (reference_points_cam[..., 1:2] < 1.0)
        & (reference_points_cam[..., 0:1] < 1.0)
        & (reference_points_cam[..., 0:1] > 0.0)
    )

    tpv_mask = torch.nan_to_num(tpv_mask).squeeze(-1)  # bs, N, n
    return reference_points_cam, tpv_mask


def sample_from_2d_img_feats(img_feats, img_metas, sampling_points_2d):
    bs, num_cams, dims = img_feats.shape[0:3]
    num_points = sampling_points_2d.shape[0]
    ref_3d = sampling_points_2d.unsqueeze(0).repeat(bs, 1, 1)  # bs, n, 3
    reference_points_cam, bev_mask = point_sampling(ref_3d, img_metas)  # bs, N, n, 2
    bev_mask = bev_mask.transpose(0, 1)  # N, bs, n
    reference_points_cam = reference_points_cam.transpose(0, 1)  # N, bs, n, 2

    slots = torch.zeros(bs, num_points, dims, dtype=img_feats.dtype, device=img_feats.device)
    indexes = []
    for _, mask_per_img in enumerate(bev_mask):
        index_query_per_img = mask_per_img[0].nonzero().squeeze(-1)
        indexes.append(index_query_per_img)
    max_len = max([len(each) for each in indexes])

    reference_points_rebatch = reference_points_cam.new_zeros([bs * num_cams, max_len, 2])  # N, L, 2
    for i, reference_points_per_img in enumerate(reference_points_cam):
        for j in range(bs):
            index_query_per_img = indexes[i]
            reference_points_rebatch[j * num_cams + i, : len(index_query_per_img)] = reference_points_per_img[
                j, index_query_per_img
            ]

    feats_3d = (
        F.grid_sample(
            img_feats.flatten(0, 1),  # bs * N, C, H, W
            reference_points_rebatch.unsqueeze(1),  # bs*N, 1, L, 2
            mode="bilinear",
            align_corners=True,
        )
        .squeeze(2)
        .transpose(1, 2)
    )  # bs * N, L, C,

    for i, index_query_per_img in enumerate(indexes):
        for j in range(bs):
            slots[j, index_query_per_img] += feats_3d[j * num_cams + i, : len(index_query_per_img)]

    count = bev_mask  # N, bs, n
    count = count.permute(1, 2, 0).sum(-1)  # bs, n
    count = torch.clamp(count, min=1.0)
    slots = slots / count[..., None]
    return slots
