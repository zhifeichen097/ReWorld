default_coord = {
    "standard": {
        "representation": "matrix", 
        "mode": "relative", 
        "reference": "world", 
        "transform": [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
    },
    "worldmem": {
        "representation": "euler_YX_degree", 
        "mode": "absolute", 
        "reference": "world", 
        "transform": [
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
    },
    "sekai": {
        "representation": "matrix", 
        "mode": "absolute", 
        "reference": "world", 
        "transform": [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
    },
    "realestate": {
        "representation": "matrix", 
        "mode": "absolute", 
        "reference": "camera", 
        "transform": [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
    },
    "spatialvid": {
        "representation": "quat_xyzw", 
        "mode": "relative", 
        "reference": "camera", 
        "transform": [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
    },
    "delta_euler": {
        "representation": "euler_YXZ_radian", 
        "mode": "delta", 
        "reference": "world", 
        "transform": [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
    },
    "relative_euler": {
        "representation": "euler_YX_degree",
        "mode": "relative",
        "reference": "world",
        "transform": [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
    },
    # Per-token Plücker action maps [d(3), m=o×d(3)] at the DiT TOKEN grid
    # (latent/patch2: 480p latent 24x40 -> 12x20; 720p latent 44x80 -> 22x40).
    # mode 'relative' = pass the relative-c2w trajectory (clip[0]=I) straight to
    # the plucker builder (utils/camera.py c2w_to_plucker_embedding; fx_n=1.0
    # default intrinsics — SAME convention as the interactive consumer at
    # pipeline/causal_diffusion_inference.py:691, keep them in sync).
    # Consumed by the existing ndim==5 branch (action_model.py:388); requires
    # H*W == frame_seqlen. Output per sample: (T_latent, H_tok, W_tok, 6).
    "plucker_480p": {
        "representation": "plucker_ori_12_20",
        "mode": "relative",
        "reference": "world",
        "transform": [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
    },
    "plucker_720p": {
        "representation": "plucker_ori_22_40",
        "mode": "relative",
        "reference": "world",
        "transform": [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
    },
    # [plucker-v2] Fixed-recipe variant of plucker_480p (opt-in via
    # target_action_mode: 'plucker_480p_v2'; legacy 'plucker_480p' untouched).
    # Fixes vs v1 (convention audit, cross-checked against public camera-control setups):
    #   anchor_chunk 4  — re-anchor every num_frame_per_block latent frames:
    #                     bounded |o|, stationary input, and a constant pose
    #                     offset (the eval frame0!=I bug) cancels exactly.
    #   t_scale 2.0     — fixed metric divisor (NOT per-clip max: keeps true
    #                     speed); within-chunk |t|<=~2m at UE walking speed
    #                     puts the moment channel at O(1) like directions.
    #   intrinsic       — real UE K (fov 55 x 38.28 deg -> fx_n=0.9605,
    #                     fy_n=1.4407) instead of the fictitious square 53 deg;
    #                     canonical for eval/keyboard synthesis too.
    "plucker_480p_v2": {
        "representation": "plucker_ori_12_20",
        "mode": "relative",
        "reference": "world",
        "anchor_chunk": 4,
        "t_scale": 2.0,
        "intrinsic": [[0.9605, 1.4407], [0.5, 0.5]],
        "transform": [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
    },
    "vipe": {
        "representation": "matrix",
        "mode": "relative",
        "reference": "world",
        "transform": [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
    },
    # UE (Unreal) fly-through export. camera.json stores ABSOLUTE 4x4 c2w in a
    # right-handed Y-up world (coordinate_system="right_handed_y_up"), but the CAMERA
    # local axes follow UE's native sense: forward = local +X, vertical = local +Y.
    # The pipeline's "standard" frame is OpenCV: forward = +Z, right = +X, down = +Y,
    # which is what generate_action_sequence_from_keys uses (w=+Z, yaw about Y, etc.).
    # convert_to_opencv conjugates c2w by `transform` (T @ c2w @ T^-1) so delta_euler
    # translation t -> P t and rotation R -> P R P^T. The transform below is the proper
    # rotation P that maps the UE camera basis (X-forward, Y-vertical, Z-side) to OpenCV
    # (Z-forward, Y-down, X-right): a 90 deg turn about Y (X->Z, Z->-X). EMPIRICALLY
    # DERIVED and verified empirically: forward UE clips -> +tz
    # delta_euler, yaw clips -> ry, matching the synthetic NIAH eval convention. A plain
    # sign-flip (diag(1,-1,-1,1)) is WRONG here — UE forward is the X axis, not Z.
    "ue": {
        "representation": "matrix",
        "mode": "absolute",
        "reference": "world",
        "transform": [
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ]
    }
}
