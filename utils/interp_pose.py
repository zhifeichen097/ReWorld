import numpy as np
from scipy.linalg import logm, expm

def downsample_poses_se3(
    poses: np.ndarray,
    df: int = 4,
    sampling_mode: str = 'interpolation',
) -> np.ndarray:
    """
    Downsample an SE(3) pose sequence [N, 4, 4] in a geometrically correct way.

    - interval: keep the first pose, then take every df-th pose.
    - interpolation: keep the first pose, then for each df-sized window,
      compute the SE(3) interpolation at the window midpoint (Lie-group aware).

    Args:
        poses: np.ndarray, shape [N, 4, 4]
        df: downsampling factor (>= 1)
        sampling_mode: 'interval' or 'interpolation'

    Returns:
        np.ndarray, shape [M, 4, 4] where M = 1 + ceil((N-1)/df)
    """
    poses = np.asarray(poses, dtype=np.float64)
    assert poses.ndim == 3 and poses.shape[1:] == (4, 4), \
        f"Expected (N, 4, 4), got {poses.shape}"
    if df < 1:
        raise ValueError(f"df must be >= 1, got {df}")

    N = poses.shape[0]
    if N <= 1:
        return poses.astype(np.float32)

    poses = np.stack([_sanitize_pose(T) for T in poses], axis=0)

    if sampling_mode == 'interval':
        # Keep first pose, then every df-th pose
        indices = np.concatenate([[0], np.arange(1, N, df)])
        return poses[indices].astype(np.float32)

    elif sampling_mode == 'interpolation':
        # Keep first pose, then for each window take the midpoint via SE(3) interp
        result = [poses[0]]
        for start in range(1, N, df):
            end = min(start + df, N)
            window_size = end - start

            if window_size == 1:
                # Only one pose in window, use it directly
                result.append(_sanitize_pose(poses[start]))
            else:
                # Interpolate at the midpoint of the window
                # midpoint in [start, end-1] range -> alpha in [0, 1]
                mid_idx = (window_size - 1) / 2.0  # fractional index within window
                i_local = int(np.floor(mid_idx))
                alpha = mid_idx - i_local

                T0 = poses[start + i_local]
                T1 = poses[start + i_local + 1]
                result.append(_se3_interp_two(T0, T1, alpha))

        return np.stack(result, axis=0).astype(np.float32)

    else:
        raise ValueError(f"Invalid sampling mode: {sampling_mode}")
def _project_to_so3(R: np.ndarray) -> np.ndarray:
    """Project a near-rotation matrix to SO(3) with SVD."""
    U, _, Vt = np.linalg.svd(R)
    R_proj = U @ Vt
    if np.linalg.det(R_proj) < 0:
        U[:, -1] *= -1
        R_proj = U @ Vt
    return R_proj


def _sanitize_pose(T: np.ndarray) -> np.ndarray:
    """Make pose numerically closer to a valid SE(3) element."""
    T = np.asarray(T, dtype=np.float64).copy()
    assert T.shape == (4, 4)

    T[:3, :3] = _project_to_so3(T[:3, :3])
    T[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return T


def _se3_interp_two(T0: np.ndarray, T1: np.ndarray, alpha: float) -> np.ndarray:
    """
    Interpolate between two SE(3) poses on the Lie group.

    Args:
        T0: [4,4]
        T1: [4,4]
        alpha: scalar in [0,1]

    Returns:
        T(alpha): [4,4]
    """
    T0 = _sanitize_pose(T0)
    T1 = _sanitize_pose(T1)

    T_rel = np.linalg.inv(T0) @ T1
    xi = logm(T_rel)
    T_alpha = T0 @ expm(alpha * xi)

    # logm/expm may introduce tiny imaginary parts
    T_alpha = np.real_if_close(T_alpha, tol=1000).astype(np.float64)

    # sanitize again for numerical stability
    T_alpha = _sanitize_pose(T_alpha)
    return T_alpha


# def interpolate_poses_se3(poses: np.ndarray, target_len: int) -> np.ndarray:
#     """
#     Resample a pose sequence [N,4,4] to [target_len,4,4] using piecewise SE(3) interpolation.

#     Args:
#         poses: np.ndarray, shape [N,4,4]
#         target_len: int

#     Returns:
#         np.ndarray, shape [target_len,4,4]
#     """
#     poses = np.asarray(poses, dtype=np.float64)
#     assert poses.ndim == 3 and poses.shape[1:] == (4, 4)
#     assert target_len >= 1

#     N = poses.shape[0]

#     if N == 1:
#         return np.repeat(_sanitize_pose(poses[0])[None, ...], target_len, axis=0)

#     poses = np.stack([_sanitize_pose(T) for T in poses], axis=0)
#     new_t = np.linspace(0.0, N - 1, target_len)
#     out = np.zeros((target_len, 4, 4), dtype=np.float64)

#     for k, t in enumerate(new_t):
#         if t <= 0:
#             out[k] = poses[0]
#             continue
#         if t >= N - 1:
#             out[k] = poses[-1]
#             continue

#         i = int(np.floor(t))
#         alpha = t - i  # in [0,1)

#         T0 = poses[i]
#         T1 = poses[i + 1]
#         out[k] = _se3_interp_two(T0, T1, alpha)

#     return out
from scipy.linalg import logm, expm

# (helper functions _project_to_so3, _sanitize_pose, _se3_interp_two are defined above)

def sample_and_interpolate_poses_se3(poses: np.ndarray, original_video_len: int, target_indices: np.ndarray) -> np.ndarray:
    """
    Performance-optimized SE(3) interpolator.
    Instead of generating the full e.g. 900-frame sequence, it computes the interpolated
    poses exactly at the sampled target_indices (e.g. 81 of them), saving a large amount of DataLoader CPU work.
    """
    poses = np.asarray(poses, dtype=np.float64)
    assert poses.ndim == 3 and poses.shape[1:] == (4, 4)
    
    N = poses.shape[0]
    if N == 1:
        return np.repeat(_sanitize_pose(poses[0])[None, ...], len(target_indices), axis=0)

    poses = np.stack([_sanitize_pose(T) for T in poses], axis=0)
    
    # map discrete video frame numbers proportionally onto the continuous action-axis time t in [0, N-1]
    mapping_ratio = (N - 1) / (original_video_len - 1) if original_video_len > 1 else 1.0
    mapped_t = target_indices * mapping_ratio

    out = np.zeros((len(target_indices), 4, 4), dtype=np.float64)

    for k, t in enumerate(mapped_t):
        if t <= 0:
            out[k] = poses[0]
            continue
        if t >= N - 1:
            out[k] = poses[-1]
            continue

        i = int(np.floor(t))
        alpha = t - i  # in [0,1)

        T0 = poses[i]
        T1 = poses[i + 1]
        out[k] = _se3_interp_two(T0, T1, alpha)

    return out
