import numpy as np
from scipy.spatial.transform import Rotation as R
from utils.camera import (
    c2w_to_relative_c2w,
    c2w_to_delta_rt,
    delta_rt_to_relative_c2w,
    convert_to_opencv,
    c2w_to_plucker_embedding,
    plucker_embedding_to_c2w,
    od_to_standard_plucker,
    c2w_to_od_plucker,
    c2w_to_donly_plucker,
)


def _parse_plucker_representation(representation: str):
    """
    Parse a plucker representation string.

    Format: "plucker_<type>_<H>_<W>"
        type: "ori" | "donly" | "od"
        H, W: integers for image height and width

    Returns:
        (plucker_type, H, W)
    """
    parts = representation.split("_")
    assert len(parts) == 4, (
        f"Plucker representation must be 'plucker_<type>_<H>_<W>', got '{representation}'"
    )
    plucker_type = parts[1]
    assert plucker_type in ("ori", "donly", "od"), (
        f"Plucker type must be 'ori', 'donly', or 'od', got '{plucker_type}'"
    )
    H = int(parts[2])
    W = int(parts[3])
    return plucker_type, H, W


def _parse_intrinsic(format_dict: dict):
    """
    Parse intrinsic from format_dict.

    Expected format: [[fx, fy], [cx, cy]] (normalized).
    Default: [[1, 1], [0.5, 0.5]].

    Returns:
        (fx_n, fy_n, cx_n, cy_n)
    """
    intrinsic = format_dict.get("intrinsic", None)
    if intrinsic is None:
        return 1.0, 1.0, 0.5, 0.5
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    assert intrinsic.shape == (2, 2), (
        f"intrinsic must be [[fx, fy], [cx, cy]], got shape {intrinsic.shape}"
    )
    fx_n, fy_n = intrinsic[0, 0], intrinsic[0, 1]
    cx_n, cy_n = intrinsic[1, 0], intrinsic[1, 1]
    return float(fx_n), float(fy_n), float(cx_n), float(cy_n)


def _plucker_to_c2w(poses: np.ndarray, representation: str,
                    format_dict: dict) -> np.ndarray:
    """
    Convert plucker embedding to relative c2w matrices.

    Args:
        poses: np.ndarray
            - "ori":   (N, H, W, 6) [d(3), o×d(3)]
            - "donly": (N, H, W, 3) [d(3)]
            - "od":    (N, H, W, 6) [o(3), d(3)]
        representation: plucker representation string
        format_dict: dict containing optional "intrinsic"

    Returns:
        c2w: np.ndarray, shape (N, 4, 4)
    """
    plucker_type, H, W = _parse_plucker_representation(representation)
    fx_n, fy_n, cx_n, cy_n = _parse_intrinsic(format_dict)
    poses = np.asarray(poses, dtype=np.float32)

    if plucker_type == "ori":
        # Standard plucker: [d, o×d] -> directly use plucker_embedding_to_c2w
        assert poses.shape[1:] == (H, W, 6), \
            f"ori plucker expects (N,{H},{W},6), got {poses.shape}"
        plucker_std = poses
    elif plucker_type == "od":
        # [o, d] -> convert to standard [d, o×d]
        assert poses.shape[1:] == (H, W, 6), \
            f"od plucker expects (N,{H},{W},6), got {poses.shape}"
        plucker_std = od_to_standard_plucker(poses)
    elif plucker_type == "donly":
        # [d] only -> convert to standard [d, 0]; translation not recoverable
        assert poses.shape[1:] == (H, W, 3), \
            f"donly plucker expects (N,{H},{W},3), got {poses.shape}"
        raise NotImplementedError("donly to standard plucker is not available")
    else:
        raise ValueError(f"Unknown plucker type: {plucker_type}")

    c2w = plucker_embedding_to_c2w(
        plucker_std, H=H, W=W,
        fx_n=fx_n, fy_n=fy_n, cx_n=cx_n, cy_n=cy_n
    )

    # donly cannot recover translation
    if plucker_type == "donly":
        c2w[:, :3, 3] = 0.0

    return c2w


def _c2w_to_plucker(c2w: np.ndarray, representation: str,
                    format_dict: dict) -> np.ndarray:
    """
    Convert relative c2w matrices to plucker embedding.

    Args:
        c2w: np.ndarray, shape (N, 4, 4)
        representation: plucker representation string
        format_dict: dict containing optional "intrinsic"

    Returns:
        poses: np.ndarray
            - "ori":   (N, H, W, 6) [d(3), o×d(3)]
            - "donly": (N, H, W, 3) [d(3)]
            - "od":    (N, H, W, 6) [o(3), d(3)]
    """
    plucker_type, H, W = _parse_plucker_representation(representation)
    fx_n, fy_n, cx_n, cy_n = _parse_intrinsic(format_dict)

    if plucker_type == "ori":
        # Standard plucker embedding [d, o×d]
        return c2w_to_plucker_embedding(
            c2w, H=H, W=W,
            fx_n=fx_n, fy_n=fy_n, cx_n=cx_n, cy_n=cy_n
        )  # (N, H, W, 6)
    elif plucker_type == "od":
        # Directly generate [o, d] from c2w
        return c2w_to_od_plucker(
            c2w, H=H, W=W,
            fx_n=fx_n, fy_n=fy_n, cx_n=cx_n, cy_n=cy_n
        )  # (N, H, W, 6)
    elif plucker_type == "donly":
        # Directly generate [d] from c2w (direction only, no origin)
        return c2w_to_donly_plucker(
            c2w, H=H, W=W,
            fx_n=fx_n, fy_n=fy_n, cx_n=cx_n, cy_n=cy_n
        )  # (N, H, W, 3)
    else:
        raise ValueError(f"Unknown plucker type: {plucker_type}")


def _representation_to_matrix(poses: np.ndarray, representation: str) -> np.ndarray:
    """
    Convert various pose representations to 4x4 matrices.

    Args:
        poses: np.ndarray
            - "matrix": (N, 4, 4)
            - "quat_wxyz": (N, 7) [x, y, z, qw, qx, qy, qz]
            - "quat_xyzw": (N, 7) [x, y, z, qx, qy, qz, qw]
            - "euler_<convention>_<unit>": rotation columns are ALWAYS in XYZ order,
              reordered internally according to convention.
              <convention> follows scipy Rotation.from_euler (e.g. YXZ, xyz, XY).
              <unit> is "degree" or "radian".
              Examples:
                "euler_YXZ_degree":  (N, 6) [x, y, z, rx, ry, rz] in degrees
                "euler_XY_radian":   (N, 5) [x, y, z, rx, ry]      in radians
              For 3-axis convention the input has 6 columns [x,y,z, rx,ry,rz].
              For 2-axis convention (e.g. XY) the input has 5 columns [x,y,z, rx,ry],
              the missing axis (z) is implicitly 0.
        representation: str describing the format.

    Returns:
        matrices: np.ndarray, shape (N, 4, 4)
    """
    poses = np.asarray(poses, dtype=np.float32)

    # ---------- matrix ----------
    if representation == "matrix":
        assert poses.ndim == 3 and (poses.shape[1:] == (4, 4) or poses.shape[1:] == (3, 4)), \
            f"matrix representation expects (N,4,4) or (N,3,4), got {poses.shape}"
        
        if poses.shape[1:] == (3, 4):
            homo = np.zeros((poses.shape[0], 1, 4))
            homo[:, :, 3] = 1
            poses = np.concatenate([poses, homo], axis=1)
        return poses.astype(np.float32)

    # ---------- quaternion ----------
    if representation.startswith("quat"):
        N = poses.shape[0]
        trans = poses[:, :3]  # (N, 3)
        quat_raw = poses[:, 3:]  # (N, 4)

        if representation == "quat_wxyz":
            # input: [qw, qx, qy, qz] -> scipy needs [qx, qy, qz, qw]
            quat_xyzw = np.concatenate([quat_raw[:, 1:4], quat_raw[:, 0:1]], axis=1)
        elif representation == "quat_xyzw":
            quat_xyzw = quat_raw
        else:
            raise ValueError(f"Unknown quaternion order: {representation}")

        rot_mats = R.from_quat(quat_xyzw).as_matrix()  # (N, 3, 3)

        c2w = np.zeros((N, 4, 4), dtype=np.float32)
        c2w[:, :3, :3] = rot_mats
        c2w[:, :3, 3] = trans
        c2w[:, 3, 3] = 1.0
        return c2w

    # ---------- euler ----------
    if representation.startswith("euler"):
        # Format: "euler_<convention>_<unit>"
        # e.g. "euler_YXZ_degree", "euler_xyz_radian", "euler_XY_degree"

        parts = representation.split("_")
        assert len(parts) == 3, (
            f"Euler representation must be 'euler_<convention>_<unit>', got '{representation}'"
        )
        convention = parts[1]  # e.g. "YXZ", "xyz", "XY"
        unit = parts[2]        # "degree" or "radian"
        assert unit in ("degree", "radian"), \
            f"Euler unit must be 'degree' or 'radian', got '{unit}'"
        use_degrees = (unit == "degree")

        num_rot_axes = len(convention)
        assert num_rot_axes in (2, 3), \
            f"Euler convention must have 2 or 3 axes, got '{convention}'"

        N = poses.shape[0]

        # Input rotation columns are ALWAYS in XYZ order.
        # For 3-axis convention: (N, 6) -> [x, y, z, rx, ry, rz]
        # For 2-axis convention: (N, 5) -> [x, y, z, r_axis1, r_axis2]
        #   where the two axes are determined by the convention letters.
        #   The missing axis is implicitly 0.
        if num_rot_axes == 3:
            assert poses.shape[1] == 6, \
                f"3-axis euler expects 6 columns [x,y,z,rx,ry,rz], got {poses.shape[1]}"
            trans = poses[:, :3]  # (N, 3)
            # poses[:, 3:] = [rx, ry, rz], i.e. rotations around X, Y, Z
            rot_xyz = poses[:, 3:]  # (N, 3) indexed as [X=0, Y=1, Z=2]
        else:  # num_rot_axes == 2
            assert poses.shape[1] == 5, \
                f"2-axis euler expects 5 columns [x,y,z,r1,r2], got {poses.shape[1]}"
            trans = poses[:, :3]  # (N, 3)
            # The two columns correspond to the two axes named in convention.
            # Expand to full 3-axis [rx, ry, rz] with missing axis = 0.
            rot_xyz = np.zeros((N, 3), dtype=np.float32)
            _axis_map = {'x': 0, 'X': 0, 'y': 1, 'Y': 1, 'z': 2, 'Z': 2}
            # Input columns are always in XYZ (pitch, yaw, roll) order.
            # Determine which two axes are present and assign in XYZ order.
            present_axes = sorted([_axis_map[c] for c in convention])  # e.g. "YX" -> [0, 1]
            for col_idx, axis_idx in enumerate(present_axes):
                rot_xyz[:, axis_idx] = poses[:, 3 + col_idx]
        
        # Reorder [rx, ry, rz] to match the convention order expected by scipy.
        # e.g. convention="YXZ" -> scipy expects [ry, rx, rz]
        _axis_map = {'x': 0, 'X': 0, 'y': 1, 'Y': 1, 'z': 2, 'Z': 2}
        reorder_indices = [_axis_map[c] for c in convention]
        euler_angles = rot_xyz[:, reorder_indices]  # (N, num_rot_axes)

        rot_mats = R.from_euler(convention, euler_angles, degrees=use_degrees).as_matrix()  # (N, 3, 3)

        mat = np.zeros((N, 4, 4), dtype=np.float32)
        mat[:, :3, :3] = rot_mats
        mat[:, :3, 3] = trans
        mat[:, 3, 3] = 1.0
        return mat

    # ---------- plucker ----------
    if representation.startswith("plucker"):
        raise ValueError(
            "plucker representation cannot be converted to matrix directly. "
            "Use match_coord_to_standard() which handles the full pipeline."
        )

    raise ValueError(f"Unknown representation: {representation}")


def _matrix_to_representation(matrices: np.ndarray, representation: str) -> np.ndarray:
    """
    Convert 4x4 matrices back to the specified pose representation.
    This is the inverse of _representation_to_matrix.

    Args:
        matrices: np.ndarray, shape (N, 4, 4)
        representation: str describing the target format.
            Same syntax as _representation_to_matrix.

    Returns:
        poses: np.ndarray
            - "matrix": (N, 4, 4)
            - "quat_wxyz": (N, 7) [x, y, z, qw, qx, qy, qz]
            - "quat_xyzw": (N, 7) [x, y, z, qx, qy, qz, qw]
            - "euler_<convention>_<unit>": rotation columns output in XYZ order.
              For 3-axis: (N, 6) [x, y, z, rx, ry, rz]
              For 2-axis: (N, 5) [x, y, z, r1, r2] (only the two axes in convention)
    """
    matrices = np.asarray(matrices, dtype=np.float32)
    assert matrices.ndim == 3 and matrices.shape[1:] == (4, 4), \
        f"Expected (N, 4, 4), got {matrices.shape}"
    N = matrices.shape[0]
    trans = matrices[:, :3, 3]      # (N, 3)
    rot_mats = matrices[:, :3, :3]  # (N, 3, 3)

    # ---------- matrix ----------
    if representation == "matrix":
        return matrices.astype(np.float32)

    # ---------- quaternion ----------
    if representation.startswith("quat"):
        quat_xyzw = R.from_matrix(rot_mats).as_quat()  # (N, 4) scipy default [qx,qy,qz,qw]

        if representation == "quat_wxyz":
            # [qx,qy,qz,qw] -> [qw,qx,qy,qz]
            quat_out = np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, 0:3]], axis=1)
        elif representation == "quat_xyzw":
            quat_out = quat_xyzw
        else:
            raise ValueError(f"Unknown quaternion order: {representation}")

        poses = np.concatenate([trans, quat_out], axis=1)  # (N, 7)
        return poses.astype(np.float32)

    # ---------- euler ----------
    if representation.startswith("euler"):
        parts = representation.split("_")
        assert len(parts) == 3, (
            f"Euler representation must be 'euler_<convention>_<unit>', got '{representation}'"
        )
        convention = parts[1]  # e.g. "YXZ", "xyz", "XY"
        unit = parts[2]        # "degree" or "radian"
        assert unit in ("degree", "radian"), \
            f"Euler unit must be 'degree' or 'radian', got '{unit}'"
        use_degrees = (unit == "degree")

        num_rot_axes = len(convention)
        assert num_rot_axes in (2, 3), \
            f"Euler convention must have 2 or 3 axes, got '{convention}'"

        _axis_map = {'x': 0, 'X': 0, 'y': 1, 'Y': 1, 'z': 2, 'Z': 2}

        # scipy.spatial.transform.Rotation.as_euler does not support 2-axis
        # conventions like "YX". For 2-axis cases, append the missing axis
        # as the 3rd axis, compute 3-axis Euler angles, then verify that the
        # appended axis angle is close to 0.
        if num_rot_axes == 2:
            all_axes = ['X', 'Y', 'Z'] if convention[0].isupper() else ['x', 'y', 'z']
            missing_axes = [a for a in all_axes if a.lower() not in {c.lower() for c in convention}]
            assert len(missing_axes) == 1, \
                f"2-axis Euler convention must contain 2 different axes, got '{convention}'"

            padded_convention = convention + missing_axes[0]

            # scipy returns euler angles in the convention order
            # e.g. padded_convention="YXZ" -> [ry, rx, rz]
            euler_conv_full = R.from_matrix(rot_mats).as_euler(
                padded_convention, degrees=use_degrees
            )  # (N, 3)

            # Check appended axis is close to zero
            appended_angle = euler_conv_full[:, 2]
            tol = 1e-4 if not use_degrees else 1e-2
            if not np.all(np.abs(appended_angle) < tol):
                max_err = np.max(np.abs(appended_angle))
                raise ValueError(
                    f"Rotation cannot be represented by 2-axis Euler convention "
                    f"'{convention}': appended axis '{missing_axes[0]}' is not near 0 "
                    f"(max abs = {max_err}, tol = {tol})"
                )

            euler_conv = euler_conv_full[:, :2]  # keep only requested 2 axes
        else:
            # scipy returns euler angles in the convention order
            # e.g. convention="YXZ" -> [ry, rx, rz]
            euler_conv = R.from_matrix(rot_mats).as_euler(
                convention, degrees=use_degrees
            )  # (N, 3)

        # Reorder from convention order back to XYZ order
        _axis_map = {'x': 0, 'X': 0, 'y': 1, 'Y': 1, 'z': 2, 'Z': 2}
        rot_xyz = np.zeros((N, 3), dtype=np.float32)
        for col_idx, axis_char in enumerate(convention):
            rot_xyz[:, _axis_map[axis_char]] = euler_conv[:, col_idx]

        if num_rot_axes == 3:
            # Output: (N, 6) [x, y, z, rx, ry, rz]
            poses = np.concatenate([trans, rot_xyz], axis=1)
        else:  # num_rot_axes == 2
            # Output: (N, 5) [x, y, z, r1, r2] in XYZ order (only present axes)
            present_axes = sorted([_axis_map[c] for c in convention])  # e.g. "YX" -> [0, 1]
            rot_2 = rot_xyz[:, present_axes]  # (N, 2)
            poses = np.concatenate([trans, rot_2], axis=1)

        return poses.astype(np.float32)

    # ---------- plucker ----------
    if representation.startswith("plucker"):
        raise ValueError(
            "plucker representation cannot be converted from matrix directly. "
            "Use standard_to_match_coord() which handles the full pipeline."
        )

    raise ValueError(f"Unknown representation: {representation}")


def match_coord_to_standard(
    poses: np.ndarray,
    format_dict: dict,
    skip_relative: bool = False,
) -> np.ndarray:
    """
    Convert arbitrary pose data to the standard format:
        - representation -> 4x4 matrix
        - reference      -> c2w  (camera-to-world)
        - mode           -> relative (relative to the first frame)   [optionally skipped]
        - transform      -> apply coordinate-system conversion to OpenCV

    Args:
        poses: np.ndarray, raw pose data whose layout is described by format_dict.
        format_dict: dict with keys
            - "representation": str, one of "matrix", "quat_wxyz", "quat_xyzw",
                                or "euler_<convention>_<unit>"
                                (e.g. "euler_YXZ_degree", "euler_XY_radian").
            - "mode": str, one of "absolute", "relative", "delta".
            - "reference": str, "world" (c2w) or "camera" (w2c).
            - "transform": list/np.ndarray (4,4), coordinate-system transform matrix.
                           If null / not present, no transform is applied.
        skip_relative: when True, skip Step 3 (mode -> relative) and return the poses in
            the **original mode** (usually absolute c2w) expressed in OpenCV coordinates.
            Use case: the caller wants to control "which moment is the anchor" (typically:
            upsample + sample the clip first, then make it relative on the clip, instead
            of making the whole video relative and re-anchoring afterwards).
            Note: with mode == "delta", skip_relative=True keeps the delta_rt input in its
            un-accumulated form, which is semantically different from absolute; the caller
            must handle that.

    Returns:
        np.ndarray, shape (N, 4, 4): c2w matrices in OpenCV coordinates.
        Default is relative-to-first-frame; with skip_relative=True it is absolute (depending on mode).
    """
    representation = format_dict["representation"]
    mode = format_dict["mode"]
    reference = format_dict["reference"]
    transform = format_dict.get("transform", None)

    # --- Plucker special path ---
    # Plucker embedding already encodes c2w information directly (reference is
    # always "world" / c2w implicitly). We only need to handle mode & transform.
    if representation.startswith("plucker"):
        # Step 1: plucker -> c2w (already c2w, skip reference)
        c2w = _plucker_to_c2w(poses, representation, format_dict)  # (N, 4, 4)

        # Step 2: reference -> c2w
        if reference == "camera":
            c2w = np.linalg.inv(c2w).astype(np.float32)
        elif reference == "world":
            pass
        else:
            raise ValueError(f"Unknown reference: {reference}")

        # Step 3: mode -> relative
        if not skip_relative:
            if mode == "absolute":
                c2w = c2w_to_relative_c2w(c2w)
            elif mode == "relative":
                pass
            elif mode == "delta":
                c2w = delta_rt_to_relative_c2w(c2w)
            else:
                raise ValueError(f"Unknown mode: {mode}")

        # Step 4: coordinate-system transform
        if transform is not None:
            transform = np.asarray(transform, dtype=np.float32)
            c2w = convert_to_opencv(c2w, transform)

        return c2w.astype(np.float32)

    # --- Standard path (non-plucker) ---
    # --- Step 1: representation -> 4x4 matrices ---
    c2w = _representation_to_matrix(poses, representation)  # (N, 4, 4)

    # --- Step 2: reference -> c2w ---
    if reference == "camera":
        # w2c -> c2w by inversion
        c2w = np.linalg.inv(c2w).astype(np.float32)
    elif reference == "world":
        pass  # already c2w
    else:
        raise ValueError(f"Unknown reference: {reference}")

    # --- Step 3: mode -> relative (w.r.t. first frame) ---
    if not skip_relative:
        if mode == "absolute":
            c2w = c2w_to_relative_c2w(c2w)
        elif mode == "relative":
            pass  # already relative to first frame
        elif mode == "delta":
            c2w = delta_rt_to_relative_c2w(c2w)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    # --- Step 4: apply coordinate-system transform to OpenCV ---
    if transform is not None:
        transform = np.asarray(transform, dtype=np.float32)
        c2w = convert_to_opencv(c2w, transform)

    return c2w.astype(np.float32)


def standard_to_match_coord(c2w: np.ndarray, format_dict: dict) -> np.ndarray:
    """
    Convert from the standard format (relative c2w in OpenCV coordinates)
    back to the format specified by format_dict.
    This is the inverse of match_coord_to_standard.

    NOTE: mode "absolute" degrades to "relative" in reverse, because the
    original absolute first-frame information is lost during forward conversion.

    Inverse pipeline (each step undoes the corresponding forward step):
        Step 1: undo coordinate-system transform  (inverse of forward Step 4)
        Step 2: relative -> target mode            (inverse of forward Step 3)
        Step 3: c2w -> target reference            (inverse of forward Step 2)
        Step 4: matrix -> target representation    (inverse of forward Step 1)

    Args:
        c2w: np.ndarray, shape (N, 4, 4)
             Relative c2w matrices in OpenCV coordinates (standard format).
        format_dict: dict with the same keys as match_coord_to_standard:
            - "representation": str
            - "mode": str, one of "absolute", "relative", "delta".
                      "absolute" is treated as "relative" (first-frame info lost).
            - "reference": str, "world" (c2w) or "camera" (w2c).
            - "transform": list/np.ndarray (4,4) or null.

    Returns:
        poses: np.ndarray, in the format described by format_dict.
    """
    representation = format_dict["representation"]
    mode = format_dict["mode"]
    reference = format_dict["reference"]
    transform = format_dict.get("transform", None)

    c2w = np.asarray(c2w, dtype=np.float32)

    # --- Step 1 (inverse of forward Step 4): undo OpenCV coordinate-system transform ---
    # Forward: out = T @ poses @ T_inv
    # Inverse: poses = T_inv @ out @ T
    if transform is not None:
        transform = np.asarray(transform, dtype=np.float32)
        transform_inv = np.linalg.inv(transform)
        c2w = convert_to_opencv(c2w, transform_inv)

    # --- Step 2 (inverse of forward Step 3): relative -> target mode ---
    # Forward: absolute -> relative (c2w_to_relative_c2w)
    #          relative -> pass
    #          delta    -> relative (delta_rt_to_relative_c2w)
    # Inverse: "absolute" degrades to "relative" (no first-frame info)
    #          "relative" -> pass
    #          "delta"    -> c2w_to_delta_rt
    if mode == "absolute":
        # Cannot recover absolute; keep as relative
        pass
    elif mode == "relative":
        pass
    elif mode == "delta":
        c2w = c2w_to_delta_rt(c2w)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # --- Step 3 (inverse of forward Step 2): c2w -> target reference ---
    if reference == "camera":
        # c2w -> w2c by inversion
        c2w = np.linalg.inv(c2w)
    elif reference == "world":
        pass  # already c2w
    else:
        raise ValueError(f"Unknown reference: {reference}")

    # --- Plucker special path ---
    if representation.startswith("plucker"):
        # c2w here is already in the target coordinate system with target mode/reference
        # Convert c2w matrices to plucker embedding
        c2w_for_plucker = c2w.astype(np.float32)
        # [plucker-v2, opt-in] per-chunk re-anchor: rebase every `anchor_chunk`
        # latent frames onto that group's first frame, so |o| stays bounded to
        # within-chunk motion and the input distribution is stationary across
        # the clip. A constant left-multiplied pose offset cancels exactly, so
        # eval trajectories whose frame0 is not identity become consistent with
        # training for free. Legacy plucker_* format dicts have neither key —
        # their output is byte-identical to before.
        anchor_chunk = format_dict.get("anchor_chunk", None)
        if anchor_chunk:
            ac = int(anchor_chunk)
            rebased = c2w_for_plucker.copy()
            for s in range(0, rebased.shape[0], ac):
                ref_inv = np.linalg.inv(c2w_for_plucker[s])
                rebased[s:s + ac] = np.einsum(
                    "ij,njk->nik", ref_inv, c2w_for_plucker[s:s + ac])
            c2w_for_plucker = rebased.astype(np.float32)
        # [plucker-v2, opt-in] fixed metric divisor for translation, so the
        # moment channel m = o x d lands at O(1) like the direction channel.
        # Fixed (not per-clip max) to preserve true speed information.
        t_scale = format_dict.get("t_scale", None)
        if t_scale:
            c2w_for_plucker = c2w_for_plucker.copy()
            c2w_for_plucker[:, :3, 3] /= float(t_scale)
        poses = _c2w_to_plucker(c2w_for_plucker, representation, format_dict)
        return poses

    matrices = c2w.astype(np.float32)

    # --- Step 4 (inverse of forward Step 1): matrix -> target representation ---
    poses = _matrix_to_representation(matrices, representation)

    return poses
