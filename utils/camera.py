import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from einops import rearrange
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
from PIL import Image
from typing import Optional


def compute_relative_pose(pose_a, pose_b, use_torch=False):
    """Compute relative pose matrix of camera B with respect to camera A"""
    assert pose_a.shape == (4, 4), f"Camera A extrinsic matrix should be (4,4), got {pose_a.shape}"
    assert pose_b.shape == (4, 4), f"Camera B extrinsic matrix should be (4,4), got {pose_b.shape}"
    
    if use_torch:
        if not isinstance(pose_a, torch.Tensor):
            pose_a = torch.from_numpy(pose_a).float()
        if not isinstance(pose_b, torch.Tensor):
            pose_b = torch.from_numpy(pose_b).float()
        
        pose_a_inv = torch.inverse(pose_a)
        relative_pose = torch.matmul(pose_b, pose_a_inv)
    else:
        if not isinstance(pose_a, np.ndarray):
            pose_a = np.array(pose_a, dtype=np.float32)
        if not isinstance(pose_b, np.ndarray):
            pose_b = np.array(pose_b, dtype=np.float32)
        
        pose_a_inv = np.linalg.inv(pose_a)
        relative_pose = np.matmul(pose_b, pose_a_inv)
    
    return relative_pose

def project_to_so3_svd(Rm: np.ndarray) -> np.ndarray:
    """
    Project a rotation matrix onto the nearest SO(3) (orthogonal, det=+1) via SVD.
    Input: (3,3) or (N,3,3)
    Output: same shape
    """
    Rm = np.asarray(Rm)
    if Rm.ndim == 2:
        U, _, Vt = np.linalg.svd(Rm)
        R_ortho = U @ Vt
        if np.linalg.det(R_ortho) < 0:
            U[:, -1] *= -1
            R_ortho = U @ Vt
        return R_ortho.astype(np.float32)

    if Rm.ndim == 3:
        # Batch case: per-frame SVD (clear, reliable, and fast enough).
        out = np.empty_like(Rm, dtype=np.float32)
        for i in range(Rm.shape[0]):
            U, _, Vt = np.linalg.svd(Rm[i])
            Ri = U @ Vt
            if np.linalg.det(Ri) < 0:
                U[:, -1] *= -1
                Ri = U @ Vt
            out[i] = Ri.astype(np.float32)
        return out

    raise ValueError("Rm must be (3,3) or (N,3,3)")

def quat_trans_to_c2w(pose7: np.ndarray,
                      quat_order: str = "wxyz") -> np.ndarray:
    """
    Convert (quaternion + translation) back to c2w pose matrices.

    Args:
        pose7: np.ndarray, shape (N,7)
               [qw,qx,qy,qz, tx,ty,tz] or xyzw
        quat_order: "wxyz" or "xyzw"

    Returns:
        c2w: np.ndarray, shape (N,4,4)
    """
    pose7 = pose7.astype(np.float32)
    N = pose7.shape[0]

    quat = pose7[:, :4]
    trans = pose7[:, 4:7]

    # SciPy expects (x,y,z,w)
    if quat_order.lower() == "wxyz":
        quat_xyzw = np.concatenate([quat[:, 1:4], quat[:, 0:1]], axis=1)
    elif quat_order.lower() == "xyzw":
        quat_xyzw = quat
    else:
        raise ValueError("quat_order must be 'wxyz' or 'xyzw'")

    # quaternion -> rotation matrix
    rot_mats = R.from_quat(quat_xyzw).as_matrix().astype(np.float32)  # (N,3,3)

    # assemble 4x4 c2w
    c2w = np.zeros((N, 4, 4), dtype=np.float32)
    c2w[:, :3, :3] = rot_mats
    c2w[:, :3, 3] = trans
    c2w[:, 3, 3] = 1.0

    return c2w

def normalize_camera_sequence_quat_trans(extrinsics: np.ndarray,
                                         make_quat_continuous: bool = True,
                                         quat_order: str = "wxyz",
                                         svd_stabilize: bool = True) -> np.ndarray:
    """
    Make poses relative to the first frame and flatten to quaternion+translation: (N,7).

    extrinsics: (N,4,4) Camera-to-World (Pose)
    Output: (N,7) -> [qw,qx,qy,qz, tx,ty,tz] or xyzw
    """
    extrinsics = extrinsics.astype(np.float32)
    N = extrinsics.shape[0]

    m0 = extrinsics[0]
    try:
        inv_m0 = np.linalg.inv(m0)
    except np.linalg.LinAlgError:
        inv_m0 = np.linalg.pinv(m0)

    rel = inv_m0[None, :, :] @ extrinsics  # (N,4,4)

    rot_mats = rel[:, :3, :3]              # (N,3,3)
    trans = rel[:, :3, 3].astype(np.float32)

    if svd_stabilize:
        rot_mats = project_to_so3_svd(rot_mats)

    # SciPy: as_quat() returns (x,y,z,w) by default
    quat_xyzw = R.from_matrix(rot_mats).as_quat().astype(np.float32)

    if make_quat_continuous:
        for i in range(1, N):
            if np.dot(quat_xyzw[i], quat_xyzw[i - 1]) < 0:
                quat_xyzw[i] *= -1

    if quat_order.lower() == "xyzw":
        quat = quat_xyzw
    elif quat_order.lower() == "wxyz":
        quat = np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, 0:3]], axis=1)
    else:
        raise ValueError("quat_order must be 'wxyz' or 'xyzw'")

    return np.concatenate([quat, trans], axis=1)  # (N,7)

def c2w_to_plucker_embedding(
    c2w: np.ndarray,   # (T,4,4)  relative camera-to-world extrinsic
    H: int, W: int,
    fx_n: float = 1.0, fy_n: float = 1.0, cx_n: float = 0.5, cy_n: float = 0.5,
    normalize_dir: bool = True,
    pixel_center: bool = True,
) -> np.ndarray:
    """
    Build the per-frame, per-pixel Pluecker embedding in the frame0 coordinate system: [d(3), m(3)], m = o x d
    
    Args:
        relative_c2w: np.ndarray, shape (T, 4, 4), relative camera-to-world extrinsic
        H, W: image height and width
        fx_n, fy_n: normalized focal lengths (relative to image width/height); default 1.0 (~90 degree FOV)
        cx_n, cy_n: normalized principal point (relative to image width/height); default 0.5 (image center)
        normalize_dir: whether to normalize the direction vectors
        pixel_center: whether to sample at pixel centers
    
    Returns: (T, H, W, 6) float32
    """
    c2w = c2w.astype(np.float32)
    T = c2w.shape[0]

    # 1) pixel grid (sampling at pixel centers is the common convention)
    if pixel_center:
        xs = np.arange(W, dtype=np.float32) + 0.5
        ys = np.arange(H, dtype=np.float32) + 0.5
    else:
        xs = np.arange(W, dtype=np.float32)
        ys = np.arange(H, dtype=np.float32)

    u, v = np.meshgrid(xs, ys)  # (H,W)

    # 2) normalized intrinsics -> pixel-unit intrinsics
    fx = fx_n * W
    fy = fy_n * H
    cx = cx_n * W
    cy = cy_n * H

    # 3) per-pixel ray directions in camera coordinates (pinhole model, z=1)
    x = (u - cx) / fx
    y = (v - cy) / fy
    d_cam = np.stack([x, y, np.ones_like(x)], axis=-1)  # (H,W,3)

    if normalize_dir:
        d_cam = d_cam / (np.linalg.norm(d_cam, axis=-1, keepdims=True) + 1e-8)

    # 4) transform into frame0: d = R @ d_cam, o = t
    R_rel = c2w[:, :3, :3]  # (T,3,3)
    t_rel = c2w[:, :3, 3]   # (T,3)

    # batched rotation: (T,3,3) @ (H,W,3) -> (T,H,W,3)
    d0 = np.einsum("tij,hwj->thwi", R_rel, d_cam).astype(np.float32)

    if normalize_dir:
        d0 = d0 / (np.linalg.norm(d0, axis=-1, keepdims=True) + 1e-8)

    # broadcast origins: (T,1,1,3)
    o0 = t_rel[:, None, None, :].astype(np.float32)

    # 5) Plücker: m = o x d
    m0 = np.cross(o0, d0).astype(np.float32)

    # 6) concatenate [d, m]
    plucker = np.concatenate([d0, m0], axis=-1).astype(np.float32)  # (T,H,W,6)
    return plucker

def od_to_standard_plucker(od_plucker: np.ndarray) -> np.ndarray:
    """
    Convert [o, d] plucker format to standard [d, o×d] plucker format.

    Args:
        od_plucker: np.ndarray, shape (..., 6) where last dim is [o(3), d(3)]

    Returns:
        standard_plucker: np.ndarray, shape (..., 6) where last dim is [d(3), m(3)],
                          m = o × d
    """
    od_plucker = np.asarray(od_plucker, dtype=np.float32)
    o = od_plucker[..., :3]
    d = od_plucker[..., 3:]
    m = np.cross(o, d)  # o × d
    return np.concatenate([d, m], axis=-1).astype(np.float32)


def standard_plucker_to_donly(standard_plucker: np.ndarray) -> np.ndarray:
    """
    Extract direction-only rays [d] from standard plucker format [d, m].

    Args:
        standard_plucker: np.ndarray, shape (..., 6) [d(3), m(3)]

    Returns:
        donly: np.ndarray, shape (..., 3) ray directions
    """
    standard_plucker = np.asarray(standard_plucker, dtype=np.float32)
    return standard_plucker[..., :3].copy()


def c2w_to_od_plucker(
    c2w: np.ndarray,
    H: int, W: int,
    fx_n: float = 1.0, fy_n: float = 1.0, cx_n: float = 0.5, cy_n: float = 0.5,
    normalize_dir: bool = True,
    pixel_center: bool = True,
) -> np.ndarray:
    """
    Generate per-frame, per-pixel [o, d] plucker embedding directly from c2w.

    Args:
        c2w: np.ndarray, shape (T, 4, 4) camera-to-world extrinsic
        H, W: image height and width
        fx_n, fy_n: normalized focal lengths (relative to W/H), default 1.0
        cx_n, cy_n: normalized principal point (relative to W/H), default 0.5
        normalize_dir: whether to normalize direction vectors
        pixel_center: whether to use pixel center

    Returns:
        od_plucker: np.ndarray, shape (T, H, W, 6) [o(3), d(3)]
    """
    c2w = c2w.astype(np.float32)
    T = c2w.shape[0]

    # Pixel grid
    if pixel_center:
        xs = np.arange(W, dtype=np.float32) + 0.5
        ys = np.arange(H, dtype=np.float32) + 0.5
    else:
        xs = np.arange(W, dtype=np.float32)
        ys = np.arange(H, dtype=np.float32)

    u, v = np.meshgrid(xs, ys)  # (H, W)

    # Normalized intrinsics -> pixel intrinsics
    fx = fx_n * W
    fy = fy_n * H
    cx = cx_n * W
    cy = cy_n * H

    # Camera-space ray direction (pinhole, z=1)
    x = (u - cx) / fx
    y = (v - cy) / fy
    d_cam = np.stack([x, y, np.ones_like(x)], axis=-1)  # (H, W, 3)

    if normalize_dir:
        d_cam = d_cam / (np.linalg.norm(d_cam, axis=-1, keepdims=True) + 1e-8)

    # Transform to world space: d = R @ d_cam, o = t
    R_rel = c2w[:, :3, :3]  # (T, 3, 3)
    t_rel = c2w[:, :3, 3]   # (T, 3)

    d0 = np.einsum("tij,hwj->thwi", R_rel, d_cam).astype(np.float32)  # (T, H, W, 3)

    if normalize_dir:
        d0 = d0 / (np.linalg.norm(d0, axis=-1, keepdims=True) + 1e-8)

    # Broadcast origin: (T, 1, 1, 3) -> (T, H, W, 3)
    o0 = t_rel[:, None, None, :].astype(np.float32)
    o0 = np.broadcast_to(o0, d0.shape).copy()

    return np.concatenate([o0, d0], axis=-1).astype(np.float32)  # (T, H, W, 6)


def c2w_to_donly_plucker(
    c2w: np.ndarray,
    H: int, W: int,
    fx_n: float = 1.0, fy_n: float = 1.0, cx_n: float = 0.5, cy_n: float = 0.5,
    normalize_dir: bool = True,
    pixel_center: bool = True,
) -> np.ndarray:
    """
    Generate per-frame, per-pixel direction-only rays [d] directly from c2w.

    Args:
        c2w: np.ndarray, shape (T, 4, 4) camera-to-world extrinsic
        H, W: image height and width
        fx_n, fy_n: normalized focal lengths (relative to W/H), default 1.0
        cx_n, cy_n: normalized principal point (relative to W/H), default 0.5
        normalize_dir: whether to normalize direction vectors
        pixel_center: whether to use pixel center

    Returns:
        donly: np.ndarray, shape (T, H, W, 3) ray directions
    """
    c2w = c2w.astype(np.float32)

    # Pixel grid
    if pixel_center:
        xs = np.arange(W, dtype=np.float32) + 0.5
        ys = np.arange(H, dtype=np.float32) + 0.5
    else:
        xs = np.arange(W, dtype=np.float32)
        ys = np.arange(H, dtype=np.float32)

    u, v = np.meshgrid(xs, ys)  # (H, W)

    # Normalized intrinsics -> pixel intrinsics
    fx = fx_n * W
    fy = fy_n * H
    cx = cx_n * W
    cy = cy_n * H

    # Camera-space ray direction (pinhole, z=1)
    x = (u - cx) / fx
    y = (v - cy) / fy
    d_cam = np.stack([x, y, np.ones_like(x)], axis=-1)  # (H, W, 3)

    if normalize_dir:
        d_cam = d_cam / (np.linalg.norm(d_cam, axis=-1, keepdims=True) + 1e-8)

    # Transform to world space: d = R @ d_cam
    R_rel = c2w[:, :3, :3]  # (T, 3, 3)
    d0 = np.einsum("tij,hwj->thwi", R_rel, d_cam).astype(np.float32)  # (T, H, W, 3)

    if normalize_dir:
        d0 = d0 / (np.linalg.norm(d0, axis=-1, keepdims=True) + 1e-8)

    return d0  # (T, H, W, 3)


def plucker_embedding_to_c2w(
    plucker_embedding: np.ndarray,  # (T, H, W, 6) Plücker embedding [d(3), m(3)]
    H: int, W: int,
    fx_n: float = 1.0, fy_n: float = 1.0, cx_n: float = 0.5, cy_n: float = 0.5, 
    normalize_dir: bool = True,
    pixel_center: bool = True,
) -> np.ndarray:
    """
    Recover camera-to-world matrices from a Pluecker embedding.
    
    Args:
        plucker_embedding: np.ndarray, shape (T, H, W, 6)
                          Plücker embedding [d(3), m(3)], where m = o × d
        H, W: image height and width
        fx_n, fy_n: normalized focal lengths (relative to image width/height); default 1.0 (~90 degree FOV)
        cx_n, cy_n: normalized principal point (relative to image width/height); default 0.5 (image center)
        normalize_dir: whether to normalize the direction vectors
        pixel_center: whether to sample at pixel centers
    
    Returns:
        c2w: np.ndarray, shape (T, 4, 4)
                     Relative camera-to-world extrinsic matrices
    """
    plucker_embedding = plucker_embedding.astype(np.float32)
    T = plucker_embedding.shape[0]
    
    # 1) extract d0 and m0
    d0 = plucker_embedding[:, :, :, :3]  # (T, H, W, 3)
    m0 = plucker_embedding[:, :, :, 3:]  # (T, H, W, 3)
    
    # 2) compute the pixel grid and camera-frame ray directions
    if pixel_center:
        xs = np.arange(W, dtype=np.float32) + 0.5
        ys = np.arange(H, dtype=np.float32) + 0.5
    else:
        xs = np.arange(W, dtype=np.float32)
        ys = np.arange(H, dtype=np.float32)
    
    u, v = np.meshgrid(xs, ys)  # (H,W)
    
    # normalized intrinsics -> pixel-unit intrinsics
    fx = fx_n * W
    fy = fy_n * H
    cx = cx_n * W
    cy = cy_n * H
    
    # per-pixel ray directions in camera coordinates (pinhole model, z=1)
    x = (u - cx) / fx
    y = (v - cy) / fy
    d_cam = np.stack([x, y, np.ones_like(x)], axis=-1)  # (H,W,3)
    
    if normalize_dir:
        d_cam = d_cam / (np.linalg.norm(d_cam, axis=-1, keepdims=True) + 1e-8)
    
    # 3) recover the camera position o from m0 = o x d0
    # cross-product identity: d0 x m0 = d0 x (o x d0) = o(d0.d0) - d0(d0.o)
    # if d0 is a unit vector: d0 x m0 = o - d0(d0.o)
    # with many pixels, solve by least squares
    c2w = np.zeros((T, 4, 4), dtype=np.float32)
    
    for t in range(T):
        d0_t = d0[t]  # (H, W, 3)
        m0_t = m0[t]  # (H, W, 3)
        
        # flatten d_cam and d0 for the computations below
        d_cam_flat = d_cam.reshape(-1, 3)  # (H*W, 3)
        d0_flat = d0_t.reshape(-1, 3)  # (H*W, 3)
        m0_flat = m0_t.reshape(-1, 3)  # (H*W, 3)
        
        # 3) recover the camera position o from m0 = o x d0
        # note: in c2w_to_plucker_embedding, m0 = o0 x d0
        # since o0 x d0 = -d0 x o0 = -[d0]_x o0, we have m0 = -[d0]_x o0
        # so to recover o0, solve -m0 = [d0]_x o0, i.e. m0 = [d0]_x (-o0)
        # or directly solve -m0 = [d0]_x o, in which case o_est is already correct
        # write -m0 = [d0]_x o as a linear system: -[d0]_x o = m0, i.e. [d0]_x o = -m0
        # where [d0]_x is the skew-symmetric matrix of d0
        # least-squares solve: min ||[d0]_x o - (-m0)||^2
        
        # build the linear system: for each pixel, -m = [d]_x o
        # [d]_× = [[0, -d_z, d_y], [d_z, 0, -d_x], [-d_y, d_x, 0]]
        A_list = []
        b_list = []
        
        for i in range(d0_flat.shape[0]):
            d = d0_flat[i]  # (3,)
            m = m0_flat[i]  # (3,)
            
            # build the skew-symmetric matrix [d]_x
            skew_symmetric = np.array([
                [0, -d[2], d[1]],
                [d[2], 0, -d[0]],
                [-d[1], d[0], 0]
            ], dtype=np.float32)
            
            A_list.append(skew_symmetric)
            b_list.append(-m)  # use -m because m0 = o x d0 = -d0 x o = -[d0]_x o
        
        # stack into one large linear system
        A = np.stack(A_list, axis=0)  # (H*W, 3, 3)
        b = np.stack(b_list, axis=0)  # (H*W, 3)
        
        # flatten to (3*H*W, 3) and (3*H*W,)
        A_flat = A.reshape(-1, 3)  # (3*H*W, 3)
        b_flat = b.reshape(-1)  # (3*H*W,)
        
        # least squares: o = (A^T A)^(-1) A^T b
        try:
            o_est = np.linalg.lstsq(A_flat, b_flat, rcond=None)[0]  # (3,)
        except np.linalg.LinAlgError:
            # fall back to a simplified method if the solve fails
            # note: m0 = o x d0, so d0 x m0 = d0 x (o x d0) = o(d0.d0) - d0(d0.o)
            # if d0 is a unit vector, d0 x m0 is approximately o
            # more precisely: since m0 = o x d0 = -d0 x o, we get -m0 = d0 x o, i.e. o = -d0 x m0 / |d0|^2
            # or: d0 x m0 = d0 x (o x d0) = o(d0.d0) - d0(d0.o)
            d0_cross_m0 = np.cross(d0_flat, m0_flat)  # (H*W, 3)
            o_est = -np.mean(d0_cross_m0, axis=0)  # (3,), note the negative sign
        
        # 4) recover the rotation matrix R from d0 = R @ d_cam
        # via the Kabsch algorithm / SVD
        
        # normalize (if requested)
        if normalize_dir:
            d_cam_flat = d_cam_flat / (np.linalg.norm(d_cam_flat, axis=-1, keepdims=True) + 1e-8)
            d0_flat = d0_flat / (np.linalg.norm(d0_flat, axis=-1, keepdims=True) + 1e-8)
        
        # solve for the optimal rotation with the Kabsch algorithm
        # compute the covariance matrix
        # if d0 = R @ d_cam (row-wise), then H = d_cam^T @ d0
        H_kabsch = d_cam_flat.T @ d0_flat  # (3, 3)
        
        # SVD: H = U @ S @ V^T, where Vt = V^T
        U, S, Vt = np.linalg.svd(H_kabsch)
        
        # standard Kabsch: R = V @ U^T
        # since Vt = V^T, V = Vt^T, hence R = Vt^T @ U^T
        # note: Vt.T = V, so R = Vt.T @ U.T = V @ U^T
        R_est = Vt.T @ U.T  # (3, 3), equivalent to V @ U^T
        
        # ensure a right-handed frame (det(R) = 1)
        if np.linalg.det(R_est) < 0:
            Vt[-1, :] *= -1
            R_est = Vt.T @ U.T
        
        # project onto SO(3) via SVD
        R_est = project_to_so3_svd(R_est)
        
        # 5) assemble the relative_c2w matrix
        c2w[t, :3, :3] = R_est
        c2w[t, :3, 3] = o_est
        c2w[t, 3, 3] = 1.0
    
    return c2w

def plucker_embedding_to_video(
    plucker_embedding: np.ndarray,  # (T, H, W, 6) Plücker embedding [d(3), m(3)]
    normalize: bool = True,
    separate_channels: bool = False,
) -> np.ndarray:
    """
    Visualize a Pluecker embedding as video data.
    
    Args:
        plucker_embedding: np.ndarray, shape (T, H, W, 6)
                          Plücker embedding [d(3), m(3)], where m = o × d
        normalize: bool, whether to normalize vectors to [-1, 1] before mapping to [0, 255]
        separate_channels: bool, if True return a dict {'d': d_video, 'm': m_video}; if False return the concatenated video
    
    Returns:
        video: np.ndarray, shape (T, H, W, 3), or (T, H, 2*W, 3) if separate_channels=False,
               or a dict {'d': (T, H, W, 3), 'm': (T, H, W, 3)} if separate_channels=True.
               dtype uint8 with values in [0, 255]; can be passed directly to export_to_video(video / 255.0, ...)
    """
    plucker_embedding = plucker_embedding.astype(np.float32)
    T, H, W, _ = plucker_embedding.shape
    
    # extract the d and m vectors
    d = plucker_embedding[:, :, :, :3]  # (T, H, W, 3)
    m = plucker_embedding[:, :, :, 3:]  # (T, H, W, 3)
    
    def vector_to_rgb(v, normalize_vec=True):
        """
        Convert vectors to an RGB image.
        v: (T, H, W, 3)
        """
        if normalize_vec:
            # normalize to [-1, 1]
            v_norm = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)
            # map to [0, 1]
            v_rgb = (v_norm + 1.0) / 2.0
        else:
            # directly normalize to [0, 1] (assumes vectors are already in a reasonable range)
            v_min = v.min(axis=-1, keepdims=True)
            v_max = v.max(axis=-1, keepdims=True)
            v_range = v_max - v_min + 1e-8
            v_rgb = (v - v_min) / v_range
        
        # map to [0, 255] and convert to uint8
        v_rgb = (v_rgb * 255.0).clip(0, 255).astype(np.uint8)
        return v_rgb
    
    d_rgb = vector_to_rgb(d, normalize)
    m_rgb = vector_to_rgb(m, normalize)
    
    if separate_channels:
        # return a dict containing the d and m videos
        return {'d': d_rgb, 'm': m_rgb}
    else:
        # concatenate d and m side by side
        combined = np.concatenate([d_rgb, m_rgb], axis=2)  # (T, H, 2*W, 3)
        return combined

def c2w_to_relative_c2w(c2w: np.ndarray) -> np.ndarray:
    """
    Convert camera-to-world extrinsic to relative camera-to-world extrinsic
    
    Args:
        c2w: np.ndarray, shape (N, 4, 4)
             Camera-to-world extrinsic matrices
    
    Returns:
        relative_c2w: np.ndarray, shape (N, 4, 4)
                     Relative camera-to-world extrinsic matrices (relative to first frame)
                     First frame will be identity matrix
    """
    c2w = c2w.astype(np.float32)
    N = c2w.shape[0]
    
    # Get first frame's inverse
    c2w_0 = c2w[0]
    try:
        inv_c2w_0 = np.linalg.inv(c2w_0)
    except np.linalg.LinAlgError:
        inv_c2w_0 = np.linalg.pinv(c2w_0)
    
    # Compute relative poses: relative_c2w[i] = inv(c2w[0]) @ c2w[i]
    relative_c2w = inv_c2w_0[None, :, :] @ c2w  # (N, 4, 4)
    
    return relative_c2w

def c2w_to_delta_rt(extrinsic: np.ndarray) -> np.ndarray:
    """
    Get adjacent delta rotation and translation from camera-to-world extrinsic
    """
    extrinsic_inv = np.linalg.inv(extrinsic)
    delta_rt = extrinsic_inv[:-1] @ extrinsic[1:]
    delta_rt = np.concatenate([np.eye(4)[None,...], delta_rt], axis=0)
    return delta_rt

def rt34_to_homogeneous_44(rt34: np.ndarray) -> np.ndarray:
    """
    Convert rotation-translation matrices from (N, 3, 4) to homogeneous (N, 4, 4) format
    
    Args:
        rt34: np.ndarray, shape (N, 3, 4)
              Rotation-translation matrices where first 3x3 is rotation and last column is translation
    
    Returns:
        homogeneous_44: np.ndarray, shape (N, 4, 4)
                       Homogeneous transformation matrices with bottom row [0, 0, 0, 1]
    """
    rt34 = np.asarray(rt34, dtype=np.float32)
    N = rt34.shape[0]
    bottom_row = np.tile(np.array([[0, 0, 0, 1]], dtype=np.float32), (N, 1, 1))  # (N, 1, 4)
    homogeneous_44 = np.concatenate([rt34, bottom_row], axis=1)  # (N, 4, 4)
    return homogeneous_44

def delta_rt_to_relative_c2w(delta_rt: np.ndarray) -> np.ndarray:
    """
    Get relative camera-to-world extrinsic from adjacent delta rotation and translation
    
    Args:
        delta_rt: np.ndarray, shape (N, 4, 4)
                  First frame is identity matrix, subsequent frames are adjacent delta transforms
    
    Returns:
        relative_c2w: np.ndarray, shape (N, 4, 4)
                     Relative camera-to-world extrinsic matrices
    """
    N = delta_rt.shape[0]
    relative_c2w = np.zeros_like(delta_rt, dtype=np.float32)
    
    # First frame is identity (relative to itself)
    relative_c2w[0] = np.eye(4, dtype=np.float32)
    
    # Accumulate transforms: relative_c2w[i] = relative_c2w[i-1] @ delta_rt[i]
    for i in range(1, N):
        relative_c2w[i] = relative_c2w[i-1] @ delta_rt[i]
    
    return relative_c2w

def get_camera_trajectory_image(flattened_extrinsics, arrow_density=1):
    """
    Visualize the trajectory and orientations of a flattened camera sequence.
    
    IMPORTANT ASSUMPTION: the input matrices are Camera-to-World (pose) matrices,
    i.e.: P_world = R * P_cam + t
    - t is directly the camera position in world coordinates.
    - the columns of R are the camera axes expressed in world coordinates.
    
    Args:
        flattened_extrinsics: np.ndarray, shape (N, 16)
        arrow_density: int, controls how densely orientation arrows are drawn.
    """
    import matplotlib
    matplotlib.use('Agg') 
    
    # 1. prepare data
    n_frames = flattened_extrinsics.shape[0]
    # reshape back to (N, 4, 4)
    poses = flattened_extrinsics.reshape(n_frames, 4, 4)
    
    centers = []
    directs = []
    
    for i in range(n_frames):
        # --- key assumption ---
        # assume Camera-to-World pose matrices
        
        # 1. extract centers (world space)
        # the translation part is directly the camera position in world coordinates
        center = poses[i, :3, 3]
        centers.append(center)
        
        # 2. extract viewing directions (world space)
        # assume +Z is the forward direction [0, 0, 1] in camera coordinates.
        # its world-space direction is the third column of R.
        # R_cw = poses[i, :3, :3]
        view_dir = poses[i, :3, 2] 
        # if the arrows point backwards (behind the camera), uncomment the next line:
        # view_dir = -view_dir 
        directs.append(view_dir)

    centers = np.array(centers)
    directs = np.array(directs)

    # 2. compute the scene scale (used to auto-set the arrow length)
    scene_scale = np.ptp(centers, axis=0).max()
    # for very short trajectories, fall back to a default arrow length
    arrow_len = scene_scale * 0.1 if scene_scale > 1e-5 else 0.1

    # 3. plot
    fig = plt.figure(figsize=(10, 8), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    # --- A. draw the path ---
    # connect the centers with a gray dashed line to show the path
    ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], 'k--', alpha=0.3, label='Path', linewidth=1)

    # --- B. draw the center points ---
    # use a colormap to encode time; keep the points small and unobtrusive
    sc = ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], 
                    c=np.arange(n_frames), cmap='plasma', s=15, alpha=0.8)

    # --- C. draw the orientation arrows ---
    step = arrow_density
    # only keep valid indices
    valid_idx = np.arange(0, n_frames, step)
    if len(valid_idx) > 0:
        ax.quiver(
            centers[valid_idx, 0], centers[valid_idx, 1], centers[valid_idx, 2],  # start points
            directs[valid_idx, 0], directs[valid_idx, 1], directs[valid_idx, 2],  # directions
            length=arrow_len,  # dynamic length
            normalize=True,
            color='teal',
            alpha=0.7,
            arrow_length_ratio=0.4,  # slightly larger arrow heads for readability
            linewidth=1
        )

    # clearly mark the start (red triangle) and end (green cross)
    ax.scatter(centers[0, 0], centers[0, 1], centers[0, 2], c='red', s=100, marker='^', label='Start', edgecolors='k', zorder=10)
    ax.scatter(centers[-1, 0], centers[-1, 1], centers[-1, 2], c='green', s=100, marker='x', label='End', edgecolors='k', zorder=10)

    # 4. auto-scale the axes (keep a consistent 3D aspect)
    max_range = np.ptp(centers, axis=0).max() / 2.0
    if max_range < 1e-6: max_range = 0.1  # avoid division by zero
        
    mid = (centers.max(axis=0) + centers.min(axis=0)) * 0.5
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    ax.set_xlabel('World X')
    ax.set_ylabel('World Y')
    ax.set_zlabel('World Z')
    ax.set_title(f'Camera Trajectory (Pose Assumption)\nArrows define looking direction')
    
    plt.colorbar(sc, label='Frame Index', shrink=0.5)
    ax.legend()

    # 5. convert to a PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    img = Image.open(buf)
    
    plt.close(fig)
    return img

def get_new_camera_from_keyboard(
    c2w: np.ndarray,
    delta_t=0.1,
    delta_r=0.1 * np.pi,
    chunk_size=3
):
    """Get a new camera pose sequence and matching key sequences from keyboard/mouse input.

    Returns:
        poses: (chunk_size, 4, 4) camera c2w sequence
        mouse_actions: list[str], length chunk_size, per-frame mouse key (e.g. 'i','k','j','l','u')
        keyboard_actions: list[str], length chunk_size, per-frame keyboard key (e.g. 'w','s','a','d','space','ctrl','q')
    """
    MOUSE_IDX = {
        "i":  [delta_r, 0],
        "k":  [-delta_r, 0],
        "j":  [0, -delta_r],
        "l":  [0, delta_r],
        "u":  [0, 0],
    }
    KEYBOARD_IDX = { 
        "w": [0, 0, delta_t], 
        "s": [0, 0, -delta_t], 
        "a": [-delta_t, 0, 0], 
        "d": [delta_t, 0, 0],
        "space": [0, -delta_t, 0],
        "ctrl": [0, delta_t, 0],
        "q":  [0, 0, 0],
    }
    flag = 0
    while flag != 1:
        try:
            idx_mouse = input('Please input the mouse action (e.g. `U`):\n').strip().lower()
            idx_keyboard = input('Please input the keyboard action (e.g. `W`):\n').strip().lower()
            if idx_mouse in MOUSE_IDX.keys() and idx_keyboard in KEYBOARD_IDX.keys():
                flag = 1
        except:
            pass
    mouse_cond = np.array(MOUSE_IDX[idx_mouse])
    keyboard_cond = np.array(KEYBOARD_IDX[idx_keyboard])

    # yaw about the world Y axis, pitch about the camera right axis; only these two DOF of the pose are updated
    # for chunk_size>1 use a constant-velocity model: repeatedly apply the same increment to emit poses for the next chunk_size frames

    def _one_step(curr_c2w: np.ndarray) -> np.ndarray:
        R_c2w = curr_c2w[:3, :3]
        t_c2w = curr_c2w[:3, 3]

        pitch_delta, yaw_delta = mouse_cond.astype(np.float32)

        yaw_rot = R.from_rotvec(np.array([0.0, yaw_delta, 0.0], dtype=np.float32)).as_matrix()
        R_after_yaw = yaw_rot @ R_c2w

        right_axis = R_after_yaw[:, 0]
        right_axis = right_axis / (np.linalg.norm(right_axis) + 1e-8)
        pitch_rot = R.from_rotvec(right_axis * pitch_delta).as_matrix()

        R_new = pitch_rot @ R_after_yaw
        R_new = project_to_so3_svd(R_new)

        trans_delta_world = R_new @ keyboard_cond.astype(np.float32)
        t_new = t_c2w + trans_delta_world

        out = np.eye(4, dtype=np.float32)
        out[:3, :3] = R_new
        out[:3, 3] = t_new
        return out

    chunk_len = int(max(1, chunk_size))
    c2w_curr = c2w.astype(np.float32)
    poses = []
    for _ in range(chunk_len):
        c2w_curr = _one_step(c2w_curr)
        poses.append(c2w_curr)

    # key sequences: one entry per pose, repeating the same mouse/keyboard pair each frame
    mouse_actions = [idx_mouse] * chunk_len
    keyboard_actions = [idx_keyboard] * chunk_len

    return np.stack(poses, axis=0), mouse_actions, keyboard_actions

def convert_to_opencv(poses: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Convert pose matrix/matrices from Input coordinates 
    (defined by transform) to OpenCV coordinates

    Supports input shape:
      - (4, 4)
      - (N, 4, 4)
    """
    poses = np.asarray(poses)

    if poses.shape == (4, 4):
        poses_in = poses[None]
        squeeze = True
    elif poses.ndim == 3 and poses.shape[1:] == (4, 4):
        poses_in = poses
        squeeze = False
    else:
        raise ValueError(f"poses must have shape (4,4) or (N,4,4), got {poses.shape}")

    transform = transform.astype(poses.dtype)

    T = transform
    T_inv = np.linalg.inv(T)

    out = T[None, :, :] @ poses_in @ T_inv[None, :, :]

    return out[0] if squeeze else out

def c2w_to_euler(
    c2w: np.ndarray,
    degrees: bool = True,
) -> np.ndarray:
    """
    Convert camera-to-world matrices to Euler angles.

    c2w: (N, 4, 4)
    degrees: whether to return in degrees or radians
    return: (N, 6)
    """
    c2w = np.asarray(c2w, dtype=np.float64)
    N = c2w.shape[0]
    assert c2w.shape == (N, 4, 4), f"Expected (N, 4, 4), got {c2w.shape}"

    trans = c2w[:, :3, 3].copy()  # (N, 3)
    rot_mats = c2w[:, :3, :3]  # (N, 3, 3)
    euler_angles = R.from_matrix(rot_mats).as_euler("YXZ", degrees=degrees)  # (N, 3) -> [yaw, pitch, roll]

    yaw   = euler_angles[:, 0]  # (N,)
    pitch = euler_angles[:, 1]  # (N,)
    roll = euler_angles[:, 2]   # (N,)

    if degrees:
        yaw   = (yaw + 180.0) % 360.0 - 180.0
        pitch = (pitch + 180.0) % 360.0 - 180.0
        roll = (roll + 180.0) % 360.0 - 180.0
    else:
        yaw   = (yaw + np.pi) % (2 * np.pi) - np.pi
        pitch = (pitch + np.pi) % (2 * np.pi) - np.pi
        roll = (roll + np.pi) % (2 * np.pi) - np.pi

    poses = np.stack([trans[:, 0], trans[:, 1], trans[:, 2],
                      pitch, yaw, roll], axis=-1)  # (N, 6)

    return poses.astype(np.float32)


def delta_rt_to_relative_euler(
    delta_rt: np.ndarray,
    degrees: bool = True,
) -> np.ndarray:
    """
    Convert adjacent delta rotation-translation matrices to relative Euler angles.

    delta_rt: (N, 4, 4) delta RT matrices (first frame should be identity)
    degrees: whether to return Euler angles in degrees
    return: (N, 6)  [x, y, z, pitch, yaw, roll]  relative to the first frame
    """
    relative_c2w = delta_rt_to_relative_c2w(delta_rt)   # (N, 4, 4)
    relative_euler = c2w_to_euler(relative_c2w, degrees=degrees)  # (N, 6)
    return relative_euler


def generate_constant_action_sequence(
    c2w: np.ndarray,
    idx_mouse: str,
    idx_keyboard: str,
    num_frames: int = 21,
    delta_t: float = 0.1,
    delta_r: float = 0.1 * np.pi,
) -> np.ndarray:
    """
    Uses the same motion model as get_new_camera_from_keyboard,
    but without interactive input: given fixed mouse/keyboard keys, generate a camera pose sequence of length num_frames.
    """
    return generate_action_sequence_from_keys(
        c2w,
        key_sequence=[idx_keyboard] * num_frames,
        mouse_sequence=[idx_mouse] * num_frames,
        num_frames=num_frames,
        delta_t=delta_t,
        delta_r=delta_r,
    )


_KEYBOARD_KEYS = {"w", "s", "a", "d", "space", "ctrl", "q"}
_MOUSE_KEYS = {"i", "k", "j", "l", "u"}


def parse_key_sequence(raw_seq: str, num_frames: int, block_size: int = 4):
    """Parse a user key string into (mouse_sequence, keyboard_sequence).

    Each character corresponds to one **block** (block_size latent frames share the same key):
        - WASD / space / ctrl -> keyboard action, mouse defaults to 'u'
        - IJKL               -> mouse action, keyboard defaults to 'q'
        - Q / U              -> stationary (keyboard='q', mouse='u')

    Sequences shorter than num_frames are padded with 'q' (stationary); longer ones are truncated.
    """
    mouse_seq: list[str] = []
    kb_seq: list[str] = []
    for ch in raw_seq.lower():
        if ch in _KEYBOARD_KEYS:
            kb_seq.extend([ch] * block_size)
            mouse_seq.extend(["u"] * block_size)
        elif ch in _MOUSE_KEYS:
            mouse_seq.extend([ch] * block_size)
            kb_seq.extend(["q"] * block_size)
        else:
            kb_seq.extend(["q"] * block_size)
            mouse_seq.extend(["u"] * block_size)

    while len(kb_seq) < num_frames:
        kb_seq.append("q")
        mouse_seq.append("u")

    return mouse_seq[:num_frames], kb_seq[:num_frames]


def generate_action_sequence_from_keys(
    c2w: np.ndarray,
    key_sequence: list,
    mouse_sequence: list,
    num_frames: int = 21,
    delta_t: float = 0.1,
    delta_r: float = 0.1 * np.pi,
) -> np.ndarray:
    """Generate a camera pose sequence from per-frame key sequences.

    Args:
        c2w: initial c2w (4, 4)
        key_sequence: keyboard key list of length >= num_frames
        mouse_sequence: mouse key list of length >= num_frames
        num_frames: number of output frames
        delta_t / delta_r: motion step sizes
    Returns:
        poses: (num_frames, 4, 4) np.float32
    """
    MOUSE_IDX = {
        "i":  [delta_r, 0],
        "k":  [-delta_r, 0],
        "j":  [0, -delta_r],
        "l":  [0, delta_r],
        "u":  [0, 0],
    }
    KEYBOARD_IDX = {
        "w": [0, 0, delta_t],
        "s": [0, 0, -delta_t],
        "a": [-delta_t, 0, 0],
        "d": [delta_t, 0, 0],
        "space": [0, -delta_t, 0],
        "ctrl": [0, delta_t, 0],
        "q": [0, 0, 0],
    }

    poses = []
    c2w_curr = c2w.astype(np.float32)
    for i in range(int(num_frames)):
        m_key = mouse_sequence[i] if i < len(mouse_sequence) else "u"
        k_key = key_sequence[i] if i < len(key_sequence) else "q"
        mouse_cond = np.array(MOUSE_IDX.get(m_key, [0, 0]), dtype=np.float32)
        keyboard_cond = np.array(KEYBOARD_IDX.get(k_key, [0, 0, 0]), dtype=np.float32)

        R_c2w = c2w_curr[:3, :3]
        t_c2w = c2w_curr[:3, 3]

        pitch_delta, yaw_delta = mouse_cond
        yaw_rot = R.from_rotvec(np.array([0.0, yaw_delta, 0.0], dtype=np.float32)).as_matrix()
        R_after_yaw = yaw_rot @ R_c2w
        right_axis = R_after_yaw[:, 0]
        right_axis = right_axis / (np.linalg.norm(right_axis) + 1e-8)
        pitch_rot = R.from_rotvec(right_axis * pitch_delta).as_matrix()
        R_new = pitch_rot @ R_after_yaw
        R_new = project_to_so3_svd(R_new)

        trans_delta_world = R_new @ keyboard_cond
        t_new = t_c2w + trans_delta_world

        out = np.eye(4, dtype=np.float32)
        out[:3, :3] = R_new
        out[:3, 3] = t_new
        c2w_curr = out
        poses.append(out)

    return np.stack(poses, axis=0).astype(np.float32)


# ---- MULTI-KEY (simultaneous) composite-pose builder ---------------------------
# Unit deltas (scaled by delta_t / delta_r below). Same axes as KEYBOARD_IDX/MOUSE_IDX.
_KEYBOARD_UNIT = {"w": [0, 0, 1], "s": [0, 0, -1], "a": [-1, 0, 0], "d": [1, 0, 0],
                  "space": [0, -1, 0], "ctrl": [0, 1, 0]}
_MOUSE_UNIT = {"i": [1, 0], "k": [-1, 0], "j": [0, -1], "l": [0, 1]}  # [pitch, yaw]


def generate_multikey_action_sequence(
    c2w: np.ndarray,
    combos: list,
    num_frames: int = 48,
    block_size: int = 4,
    delta_t: float = 0.1,
    delta_r: float = 0.1 * np.pi,
) -> np.ndarray:
    """Composite-pose trajectory for SIMULTANEOUS multi-key actions.

    ``combos``: list of strings, ONE per block (block_size latent frames share it).
    Each string = the keys pressed together that block, e.g. 'wa' (W+A = forward+left
    diagonal), 'wl' (W+L = forward + turn right), 'wal' (3-key). Per block we SUM the
    keyboard translation deltas + SUM the mouse rotation deltas, then apply the SAME
    yaw(world-Y) -> pitch(camera right axis) -> translate(in rotated frame) step as
    ``generate_action_sequence_from_keys`` (single-key == a 1-element combo). This makes
    W+A a genuine diagonal pose, testing whether the learned pose manifold generalizes to
    composite poses unseen as single keys.
    """
    poses = []
    c2w_curr = c2w.astype(np.float32)
    for i in range(int(num_frames)):
        b = i // block_size
        combo = (combos[b] if b < len(combos) else "").lower()
        kb = np.zeros(3, dtype=np.float32)
        ms = np.zeros(2, dtype=np.float32)
        for ch in combo:
            if ch in _KEYBOARD_UNIT:
                kb += np.array(_KEYBOARD_UNIT[ch], dtype=np.float32) * delta_t
            elif ch in _MOUSE_UNIT:
                ms += np.array(_MOUSE_UNIT[ch], dtype=np.float32) * delta_r
        R_c2w = c2w_curr[:3, :3]
        t_c2w = c2w_curr[:3, 3]
        pitch_delta, yaw_delta = ms
        yaw_rot = R.from_rotvec(np.array([0.0, yaw_delta, 0.0], dtype=np.float32)).as_matrix()
        R_after_yaw = yaw_rot @ R_c2w
        right_axis = R_after_yaw[:, 0]
        right_axis = right_axis / (np.linalg.norm(right_axis) + 1e-8)
        pitch_rot = R.from_rotvec(right_axis * pitch_delta).as_matrix()
        R_new = project_to_so3_svd(pitch_rot @ R_after_yaw)
        t_new = t_c2w + R_new @ kb
        out = np.eye(4, dtype=np.float32)
        out[:3, :3] = R_new
        out[:3, 3] = t_new
        c2w_curr = out
        poses.append(out)
    return np.stack(poses, axis=0).astype(np.float32)

# if __name__ == "__main__":
#     history = np.eye(4, dtype=np.float32)
#     history = history[None]
#     for i in range(3):
#         print(history)
#         this_poses, this_mouse_actions, this_keyboard_actions = get_new_camera_from_keyboard(history[-1], chunk_size=5)
#         breakpoint()
#         history = np.concatenate([history, this_poses], axis=0)
    
#     from visualize  import visualize_trajectory
#     history = np.asarray(history)
#     visualize_trajectory(history, "trajectory_vis.html")