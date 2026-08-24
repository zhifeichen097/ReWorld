"""Utility functions for visualization and video/image export."""

import os
import re
import subprocess
from pathlib import Path
from typing import Optional, Union

import cv2
import io
import matplotlib.pyplot as plt
import numpy as np
# NOTE: plotly is imported lazily inside visualize_trajectory() (the only user) so the
# inference/eval video path does not hard-depend on plotly being pip-installed.
from diffusers.utils import export_to_video
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # some 3D plotting functions depend on this import
from PIL import Image

# Video-gallery constants
# Supported video/image/npy extensions are centralized here for easy extension.
VIDEO_GALLERY_VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".avi", ".mkv"}
VIDEO_GALLERY_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_GALLERY_NPY_EXTENSION = ".npy"

def visualize_trajectory(poses, output_file="trajectory_vis.html"):
    """
    Visualizes camera trajectory and orientation using Plotly.
    Converts from OpenCV format (Y down, Z forward) to Z-up format for visualization.
    
    Args:
        poses: (N, 4, 4) numpy array representing camera-to-world transformation matrices.
        output_file: Name of the output HTML file.
    """
    import plotly.graph_objects as go  # lazy: only this fn needs plotly

    # Coordinate transformation: (x, y, z) -> (x, z, -y)
    # This maps:
    #   +X (Right) -> +X (Right)
    #   +Y (Down)  -> -Z (Down) => So -Y (Up) -> +Z (Up)
    #   +Z (Forward)-> +Y (Forward)
    def transform_point(p):
        return np.array([p[0], p[2], -p[1]])

    def transform_points(pts):
        # pts: (N, 3)
        return np.stack([pts[:, 0], pts[:, 2], -pts[:, 1]], axis=1)

    # 1. Transform trajectory path
    raw_translations = poses[:, :3, 3] # (N, 3)
    translations = transform_points(raw_translations)
    
    # Create the trace for the trajectory path
    traj_trace = go.Scatter3d(
        x=translations[:, 0],
        y=translations[:, 1],
        z=translations[:, 2],
        mode='lines',
        line=dict(color='blue', width=2),
        name='Trajectory Path'
    )
    
    # 2. Create Frustums and 3D Markers
    scale = 0.05 

    def create_octahedron_mesh(centers, radius, color, name):
        """
        Creates a Mesh3d trace representing octahedrons (diamond shape) at given centers.
        This ensures the markers have a fixed 3D size relative to the scene.
        """
        # Vertices of a unit octahedron
        # 6 vertices
        v_unit = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ]) * radius
        
        # Faces of unit octahedron (indices)
        f_unit = np.array([
            [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], # Top
            [0, 4, 2], [2, 4, 1], [1, 4, 3], [3, 4, 0], # Back faces if strictly 1-sided, but Mesh3d handles it.
            # Actually standard octahedron indices:
            [0, 2, 4], [0, 4, 3], [0, 3, 5], [0, 5, 2], 
            [1, 2, 5], [1, 5, 3], [1, 3, 4], [1, 4, 2]
        ])
        # Wait, that permutation was guessed. Let's do a reliable one.
        # Top vertex: 4 (0,0,1). Bottom: 5 (0,0,-1).
        # Ring: 0, 2, 1, 3 (counter clockwise looking from top?)
        # Let's stick to simple triangles.
        
        # Correct indices for 8 faces:
        # Top pyramid (peak 4)
        # (0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4)
        # Bottom pyramid (peak 5)
        # (0, 5, 2), (2, 5, 1), (1, 5, 3), (3, 5, 0)
        
        # We need to replicate these for each center
        all_x, all_y, all_z = [], [], []
        all_i, all_j, all_k = [], [], []
        
        N_verts = 6
        
        for k, center in enumerate(centers):
            # Vertices
            v_curr = v_unit + center
            all_x.extend(v_curr[:, 0])
            all_y.extend(v_curr[:, 1])
            all_z.extend(v_curr[:, 2])
            
            # Indices
            offset = k * N_verts
            # Faces
            faces = [
                # Top
                [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
                # Bottom
                [0, 5, 2], [2, 5, 1], [1, 5, 3], [3, 5, 0]
            ]
            for face in faces:
                all_i.append(face[0] + offset)
                all_j.append(face[1] + offset)
                all_k.append(face[2] + offset)
                
        return go.Mesh3d(
            x=all_x, y=all_y, z=all_z,
            i=all_i, j=all_j, k=all_k,
            color=color,
            name=name,
            flatshading=True
        )

    # Trajectory points (Small Blue) - Kept as 3D Mesh
    # Radius relative to scale. Frustum width is ~scale/2. 
    traj_markers = create_octahedron_mesh(translations, scale * 0.1, 'blue', 'Images')

    # Start point (Scatter3d with text/legend)
    start_trace = go.Scatter3d(
        x=[translations[0, 0]],
        y=[translations[0, 1]],
        z=[translations[0, 2]],
        mode='markers+text',
        marker=dict(size=8, color='green', symbol='diamond'),
        text=['Start'],
        textposition="top center",
        name='Start'
    )

    # End point (Scatter3d with text/legend)
    end_trace = go.Scatter3d(
        x=[translations[-1, 0]],
        y=[translations[-1, 1]],
        z=[translations[-1, 2]],
        mode='markers+text',
        marker=dict(size=8, color='red', symbol='diamond'),
        text=['End'],
        textposition="top center",
        name='End'
    )
    
    data = [traj_trace, traj_markers, start_trace, end_trace] 
    
    # Pre-define frustum corners in CAMERA frame (OpenCV: X right, Y down, Z forward)
    # Tips: In OpenCV cam frame, Look direction is +Z. 
    # Frustum should extend into +Z.
    d = scale
    w = scale * 0.5
    h = scale * 0.5
    
    corners_cam = np.array([
        [0, 0, 0],    # Center
        [-w, -h, d],  # Top-Left (in image space? No, -Y is Up in normal space, but here Y is down)
                      # Let's verify OpenCV frustum:
                      # Y is down. -h is "up" in image. +h is "down".
                      # X is right.
        [w, -h, d],
        [w, h, d],
        [-w, h, d]
    ])
    
    # We will accumulate all lines to avoid adding thousands of traces which slows down Plotly
    line_x = []
    line_y = []
    line_z = []
    
    # Helper to append a line segment separated by None
    def add_line(p1, p2):
        line_x.extend([p1[0], p2[0], None])
        line_y.extend([p1[1], p2[1], None])
        line_z.extend([p1[2], p2[2], None])

    for i in range(len(poses)):
        # Camera-to-World
        R = poses[i, :3, :3]
        t = poses[i, :3, 3]
        
        # Transform frustum corners to World Frame (Original Coordinates)
        corners_world_orig = (R @ corners_cam.T).T + t
        
        # Now transform World Frame (OpenCV style) to Viz Frame (Z-up)
        corners_viz = transform_points(corners_world_orig)
        
        center = corners_viz[0]
        # Corners 1-4
        c1, c2, c3, c4 = corners_viz[1], corners_viz[2], corners_viz[3], corners_viz[4]
        
        # Edges from center
        add_line(center, c1)
        add_line(center, c2)
        add_line(center, c3)
        add_line(center, c4)
        
        # Base rectangle
        add_line(c1, c2)
        add_line(c2, c3)
        add_line(c3, c4)
        add_line(c4, c1)

    frustum_trace = go.Scatter3d(
        x=line_x,
        y=line_y,
        z=line_z,
        mode='lines',
        line=dict(color='red', width=1),
        name='Cameras',
        showlegend=True
    )
    data.append(frustum_trace)

    # 3. Calculate consistent axis ranges
    all_x = translations[:, 0]
    all_y = translations[:, 1]
    all_z = translations[:, 2]
    
    # It's better to include frustum points in bounds calculation so they don't get clipped
    # but strictly speaking, the trajectory center is most important. 
    # Let's use trajectory min/max
    min_x, max_x = np.min(all_x), np.max(all_x)
    min_y, max_y = np.min(all_y), np.max(all_y)
    min_z, max_z = np.min(all_z), np.max(all_z)
    
    mid_x = (min_x + max_x) / 2
    mid_y = (min_y + max_y) / 2
    mid_z = (min_z + max_z) / 2
    
    range_x = max_x - min_x
    range_y = max_y - min_y
    range_z = max_z - min_z
    
    max_range = max(range_x, range_y, range_z)
    if max_range == 0:
        max_range = 1.0 # Fallback for single point
        
    half_range = max_range / 2 * 1.1 # Add 10% padding
    
    scene_matches = dict(
        xaxis=dict(range=[mid_x - half_range, mid_x + half_range], title='X (Right)'),
        yaxis=dict(range=[mid_y - half_range, mid_y + half_range], title='Y (Forward)'),
        zaxis=dict(range=[mid_z - half_range, mid_z + half_range], title='Z (Up)'),
        aspectmode='cube' # Forces the box to be a cube
    )

    layout = go.Layout(
        title='Trajectory Visualization',
        scene=scene_matches,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    fig = go.Figure(data=data, layout=layout)
    
    # 4. Set initial camera view to see everything
    # Default is usually okay with aspectmode='cube', but we can force a nice angle
    fig.update_layout(scene_camera=dict(
        eye=dict(x=1.5, y=1.5, z=1.5) # Diagonal view
    ))
    
    fig.write_html(output_file)
    print(f"Visualization saved to {output_file}")

def parse_config(config, mode="universal"):
    """
    Generate key data and mouse data from the config.
    - config: the list_actions[i] config
    - Returns: key_data and mouse_data
    """
    assert mode in ['universal', 'gta_drive', 'templerun']
    key_data = {}
    mouse_data = {}
    if mode != 'templerun':
        key, mouse = config
    else:
        key = config
    # iterate over each configured segment
    for i in range(len(key)):
        
        if mode == 'templerun':
            still, w, s, left, right, a, d = key[i]
        elif mode == 'universal':
            w, s, a, d = key[i]
        else:
            w, s, a, d = key[i][0], key[i][1], mouse[i][1] < 0, mouse[i][1] > 0
        if mode == 'universal':
            mouse_y, mouse_x = mouse[i]
            mouse_y = -1 * mouse_y
        try:
            tt = int(htb.index(1) + 1)
        except:
            tt = 0
        # key states
        key_data[i] = {
            "W": bool(w),
            "A": bool(a),
            "S": bool(s),
            "D": bool(d),
        }
        if mode == 'templerun':
            key_data[i].update({"left": bool(left), "right": bool(right)})
        # mouse position
        if mode == 'universal':
            if i == 0:
                mouse_data[i] = (320, 352//2)  # default initial position
            else:
                global_scale_factor = 0.1
                mouse_scale_x = 15 * global_scale_factor
                mouse_scale_y = 15 * 4 * global_scale_factor
                mouse_data[i] = (
                    mouse_data[i-1][0] + mouse_x * mouse_scale_x,  # accumulated x coordinate
                    mouse_data[i-1][1] + mouse_y * mouse_scale_y,  # accumulated y coordinate
                )
    return key_data, mouse_data


# Convert an action embedding (mouse+keyboard) into the key_data/mouse_data needed by process_video
def action_embedding_to_key_mouse(action_embeds, mode="universal"):
    """
    action_embeds:
      - supports shape (T, 12) or (B, T, 12)
      - first 5 dims are the mouse-direction one-hot (i,k,j,l,u); the last 7 dims are the keyboard one-hot (w,s,a,d,space,ctrl,q)
    Returns:
      key_data: {frame_idx: {"W":bool,"A":bool,"S":bool,"D":bool}}
      mouse_data: {frame_idx: (x, y)}, only used when mode='universal'
    """
    assert mode == "universal", "only universal mode is supported for now; extend as needed"

    # accept torch.Tensor / np.ndarray / list
    if hasattr(action_embeds, "detach"):  # torch.Tensor
        import torch
        # bfloat16/float16 may not convert to numpy directly; normalize to float32
        arr = action_embeds.detach().to(dtype=torch.float32).cpu().numpy()
    else:
        arr = np.asarray(action_embeds)

    # accept both (T, 12) and (B, T, 12); only a single trajectory is handled here — take batch 0 if batched
    if arr.ndim == 1:
        arr = arr[None, :]
    elif arr.ndim == 3:
        arr = arr[0]

    key_data = {}
    mouse_data = {}

    # scaling of the accumulated mouse displacement, kept consistent with parse_config
    global_scale_factor = 0.1
    mouse_scale_x = 15 * global_scale_factor
    mouse_scale_y = 15 * 4 * global_scale_factor
    mouse_pos = (640, 352 // 2)  # initial position

    # mouse direction order: i(up), k(down), j(left), l(right), u(idle)
    mouse_dirs = {
        0: (0, -1),  # i: up (y decreases)
        1: (0, 1),   # k: down
        2: (1, 0),  # l: right
        3: (-1, 0),   # j: left
        4: (0, 0),   # u: stay
    }

    for idx, row in enumerate(arr):
        row = np.asarray(row).reshape(-1)  # guard against extra dims causing ambiguous boolean checks
        if row.shape[0] < 12:
            raise ValueError(f"each action_embeds row needs at least 12 dims, got {row.shape[0]}")
        mouse_vec = row[7:]
        keyboard_vec = row[:7]

        mouse_choice = int(np.argmax(mouse_vec))
        dx, dy = mouse_dirs.get(mouse_choice, (0, 0))
        mouse_pos = (
            mouse_pos[0] + dx * mouse_scale_x,
            mouse_pos[1] + dy * mouse_scale_y,
        )

        k = np.asarray(keyboard_vec).reshape(-1)
        key_data[idx] = {
            "W": bool(k[0] > 0.5),
            "A": bool(k[1] > 0.5),
            "S": bool(k[2] > 0.5),
            "D": bool(k[3] > 0.5),
            "space": bool(k[4] > 0.5), # jump up
            "ctrl": bool(k[5] > 0.5), # dive down
        }
        mouse_data[idx] = mouse_pos

    return key_data, mouse_data


# Keyboard key format: makes it easy to add new keys. (input_key, display_name) or (input_key, display_name, wide)
# wide=True means the key is drawn with extra width (e.g. space/ctrl)
DEFAULT_KEYBOARD_FORMAT = [
    ("w", "W"),
    ("a", "A"),
    ("s", "S"),
    ("d", "D"),
    ("space", "space", True),
    ("ctrl", "ctrl", True),
]
# Mouse directions: consistent with MOUSE_IDX in get_new_camera_from_keyboard (i,k,j,l,u)
MOUSE_DIRS = {
    "i": (0, -1),   # up
    "k": (0, 1),    # down
    "l": (1, 0),    # right
    "j": (-1, 0),   # left
    "u": (0, 0),    # idle
}


def _layout_from_frame_size(w, h):
    """Compute all layout sizes from the frame size (w, h); everything is relative, no absolute pixels."""
    ref = min(w, h)
    return {
        "key_w": max(20, int(ref * 0.078)),
        "key_h": max(16, int(ref * 0.078)),
        "spacing": max(4, int(ref * 0.016)),
        "bottom_margin": max(8, int(h * 0.028)),
        "horizon_shift": int(w * 0.14),
        "horizon_shift_all": int(w * 0.078),
        "vertical_shift": int(-h * 0.028),
        "wide_extra": max(12, int(ref * 0.062)),
        "radius": max(4, int(ref * 0.012)),
        "first_row_extra": max(4, int(h * 0.028)),
        "small_gap": max(2, int(ref * 0.008)),
        "font_scale": max(0.4, ref / 800.0),
        "font_thickness": max(1, int(ref / 400)),
    }


def sequences_to_key_mouse(mouse_actions, keyboard_actions, key_format, frame_width, frame_height, n_frames=None):
    """Build key_data and mouse_data from mouse_actions and keyboard_actions sequences.

    - mouse_actions: list[str], one per frame, e.g. 'i','k','j','l','u'
    - keyboard_actions: list[str], one per frame, e.g. 'w','s','a','d','space','ctrl','q'
    - key_format: list of (input_key, display_name) or (input_key, display_name, wide)
    - frame_width, frame_height: current frame resolution, used to center the mouse trajectory
    - n_frames: if given, generate exactly this many frames (padding missing frames with 'u' and the last key); otherwise max(len(mouse), len(keyboard))

    Returns:
        key_data: {frame_idx: {display_name: bool}}
        mouse_data: {frame_idx: (x, y)}
    """
    key_data = {}
    mouse_data = {}
    global_scale_factor = 0.1
    mouse_scale_x = 15 * global_scale_factor
    mouse_scale_y = 15 * 4 * global_scale_factor
    mouse_pos = (640 / 2.0, 352 / 2.0)

    n = n_frames if n_frames is not None else max(len(mouse_actions), len(keyboard_actions), 1)
    for i in range(n):
        kb = keyboard_actions[i].lower() if i < len(keyboard_actions) else ""
        for item in key_format:
            if len(item) == 3:
                input_key, display_name, _ = item
            else:
                input_key, display_name = item
            if i not in key_data:
                key_data[i] = {}
            key_data[i][display_name] = (kb == input_key.lower())

        dx, dy = MOUSE_DIRS.get(
            mouse_actions[i].lower() if i < len(mouse_actions) else "u",
            (0, 0)
        )
        mouse_pos = (
            mouse_pos[0] + dx * mouse_scale_x,
            mouse_pos[1] + dy * mouse_scale_y,
        )
        mouse_data[i] = mouse_pos

    # shift to the center of the current frame resolution
    old_cx, old_cy = 640 / 2.0, 352 / 2.0
    new_cx, new_cy = frame_width / 2.0, frame_height / 2.0
    shift_x, shift_y = new_cx - old_cx, new_cy - old_cy
    for idx in mouse_data:
        mx, my = mouse_data[idx]
        mouse_data[idx] = (mx + shift_x, my + shift_y)

    return key_data, mouse_data


def _build_key_positions_and_sizes(key_format, w, h, layout=None):
    """Compute per-key positions and widths from key_format and the frame size (w,h). Layout comes from `layout`, or is computed from (w,h) if None."""
    if layout is None:
        layout = _layout_from_frame_size(w, h)
    kw = layout["key_w"]
    kh = layout["key_h"]
    spacing = layout["spacing"]
    bottom_margin = layout["bottom_margin"]
    horison_shift = layout["horizon_shift"]
    vertical_shift = layout["vertical_shift"]
    horizon_shift_all = layout["horizon_shift_all"]
    wide_extra = layout["wide_extra"]
    first_row_extra = layout["first_row_extra"]
    small_gap = layout["small_gap"]

    key_positions = {}
    key_widths = {}
    if not key_format:
        return key_positions, key_widths
    # first key: centered on top
    input_0, display_0 = key_format[0][:2]
    wide_0 = key_format[0][2] if len(key_format[0]) == 3 else False
    w0 = kw + (wide_extra if wide_0 else 0)
    key_positions[display_0] = (
        w // 2 - w0 // 2 - horison_shift - horizon_shift_all,
        h - bottom_margin - kh * 2 + vertical_shift - first_row_extra,
    )
    key_widths[display_0] = w0
    # remaining keys: one row
    x_start = w // 2 - kw * 2 + small_gap - horison_shift - horizon_shift_all
    for i, item in enumerate(key_format[1:], start=1):
        input_k, display_k = item[:2]
        wide_k = item[2] if len(item) == 3 else False
        key_w = kw + (wide_extra if wide_k else 0)
        key_positions[display_k] = (
            x_start,
            h - bottom_margin - kh + vertical_shift,
        )
        key_widths[display_k] = key_w
        x_start += key_w + spacing
    return key_positions, key_widths


# Draw a rounded rectangle
def draw_rounded_rectangle(image, top_left, bottom_right, color, radius=10, alpha=0.5):
    overlay = image.copy()
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)

    cv2.ellipse(overlay, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, -1)
    cv2.ellipse(overlay, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, -1)
    cv2.ellipse(overlay, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, -1)
    cv2.ellipse(overlay, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, -1)

    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

# Draw keys onto a frame
def draw_keys_on_frame(frame, keys, key_size=None, spacing=None, bottom_margin=None, mode='universal', key_format=None):
    """Draw key states onto the frame. The layout is computed entirely from the frame size (w,h); key_size/spacing/bottom_margin are auto-scaled when None."""
    h, w, _ = frame.shape
    layout = _layout_from_frame_size(w, h)
    kw, kh = layout["key_w"], layout["key_h"]
    if key_size is None:
        key_size = (kw, kh)
    if spacing is None:
        spacing = layout["spacing"]
    if bottom_margin is None:
        bottom_margin = layout["bottom_margin"]

    if key_format is not None:
        key_positions, key_widths = _build_key_positions_and_sizes(key_format, w, h, layout=layout)
        key_icon = {item[1]: item[1] for item in key_format}
    else:
        key_positions, key_widths = _build_key_positions_and_sizes(DEFAULT_KEYBOARD_FORMAT, w, h, layout=layout)
        key_icon = {"W": "W", "A": "A", "S": "S", "D": "D", "space": "space", "ctrl": "ctrl"}

    radius = layout["radius"]
    font_scale = layout["font_scale"]
    font_thickness = layout["font_thickness"]
    for key, (x, y) in key_positions.items():
        is_pressed = keys.get(key, False)
        top_left = (x, y)
        key_w = key_widths.get(key, kw)
        bottom_right = (x + key_w, y + kh)

        color = (0, 255, 0) if is_pressed else (200, 200, 200)
        alpha = 0.8 if is_pressed else 0.5

        draw_rounded_rectangle(frame, top_left, bottom_right, color, radius=radius, alpha=alpha)

        label = key_icon.get(key, key)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text_x = x + (key_w - text_size[0]) // 2
        text_y = y + (kh + text_size[1]) // 2
        cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

# Overlay the mouse icon onto a frame
def overlay_icon(frame, icon, position, scale=1.0, rotation=0):
    x, y = position
    h, w, _ = icon.shape

    # scale the icon
    scaled_width = int(w * scale)
    scaled_height = int(h * scale)
    icon_resized = cv2.resize(icon, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)

    # rotate the icon
    center = (scaled_width // 2, scaled_height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
    icon_rotated = cv2.warpAffine(icon_resized, rotation_matrix, (scaled_width, scaled_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    h, w, _ = icon_rotated.shape
    frame_h, frame_w, _ = frame.shape

    # compute the drawing region
    top_left_x = max(0, int(x - w // 2))
    top_left_y = max(0, int(y - h // 2))
    bottom_right_x = min(frame_w, int(x + w // 2))
    bottom_right_y = min(frame_h, int(y + h // 2))

    icon_x_start = max(0, int(-x + w // 2))
    icon_y_start = max(0, int(-y + h // 2))
    icon_x_end = icon_x_start + (bottom_right_x - top_left_x)
    icon_y_end = icon_y_start + (bottom_right_y - top_left_y)

    # extract the icon region
    icon_region = icon_rotated[icon_y_start:icon_y_end, icon_x_start:icon_x_end]
    alpha = icon_region[:, :, 3] / 255.0
    icon_rgb = icon_region[:, :, :3]

    # extract the matching frame region
    frame_region = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    # blend the icon in
    for c in range(3):
        frame_region[:, :, c] = (1 - alpha) * frame_region[:, :, c] + alpha * icon_rgb[:, :, c]

    # write the region back into the frame
    frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = frame_region


# Process a video
def process_video(
    input_video,
    mouse_actions,
    keyboard_actions,
    mouse_icon_path='images/mouse.png',
    mouse_scale=1.0,
    mouse_rotation=0,
    process_icon=True,
    mode='universal',
    key_format=None,
):
    """Overlay key and mouse icons onto video frames.

    - input_video: (T, H, W, C) uint8 np.ndarray, or a list of (H, W, C) frames (matching the causal_diffusion export format)
    - mouse_actions: list[str], length T, per-frame mouse key such as 'i','k','j','l','u'
    - keyboard_actions: list[str], length T, per-frame keyboard key such as 'w','s','a','d','space','ctrl','q'
    - mouse_scale: extra multiplier on the mouse-icon scale; the base scale is ~5% of the frame's short side (default 1.0; use 0.8 for smaller, 1.2 for larger)
    - key_format: keyboard key format; None uses DEFAULT_KEYBOARD_FORMAT. To add keys, pass a list of (input_key, display_name) or (input_key, display_name, wide)
    """
    # normalize to a list of (H,W,C)
    if isinstance(input_video, np.ndarray):
        if input_video.ndim == 4:
            frames = [input_video[t] for t in range(input_video.shape[0])]
        else:
            frames = [input_video]
    else:
        frames = list(input_video)
    if not frames:
        return []

    frame_height, frame_width = frames[0].shape[0], frames[0].shape[1]
    frame_count = len(frames)
    if key_format is None:
        key_format = DEFAULT_KEYBOARD_FORMAT

    key_data, mouse_data = sequences_to_key_mouse(
        mouse_actions, keyboard_actions, key_format, frame_width, frame_height, n_frames=frame_count
    )
    # default keys per frame (frames absent from key_data get all-False)
    default_keys = {item[1]: False for item in key_format}

    try:
        pil_img = Image.open(mouse_icon_path).convert("RGBA")
        mouse_icon = np.array(pil_img)
        mouse_icon = cv2.cvtColor(mouse_icon, cv2.COLOR_RGBA2BGRA)
    except Exception as e:
        print(f"Error reading mouse icon with PIL: {e}")
        mouse_icon = np.zeros((100, 100, 4), dtype=np.uint8)

    # scale the mouse icon with the frame size (~5% of the short side); mouse_scale is an extra multiplier
    ref = min(frame_width, frame_height)
    target_icon_size = ref * 0.05
    icon_h, icon_w = mouse_icon.shape[:2]
    icon_max = max(icon_w, icon_h)
    base_icon_scale = (target_icon_size / icon_max) if icon_max > 0 else 1.0
    effective_mouse_scale = base_icon_scale * mouse_scale

    out_video = []
    for frame_idx, frame in enumerate(frames):
        frame = np.asarray(frame)
        if process_icon:
            keys = key_data.get(frame_idx, default_keys)
            draw_keys_on_frame(frame, keys, mode=mode, key_format=key_format)
            if mode == 'universal':
                mouse_position = mouse_data.get(frame_idx, (frame_width // 2, frame_height // 2))
                overlay_icon(frame, mouse_icon, mouse_position, scale=effective_mouse_scale, rotation=mouse_rotation)
        if frame.dtype != np.uint8:
            frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)
        else:
            frame_uint8 = frame.copy()
        out_video.append(frame_uint8)
        print(f"Processing frame {frame_idx + 1}/{frame_count}", end="\r")
    return np.asarray(out_video)



# -- trajectory name -> (mouse_key, keyboard_key) mapping, consistent with SimulatedActionDataset.traj_specs --
_TRAJ_KEY_MAP = {
    "move_forward":  ("u", "w"),
    "move_backward": ("u", "s"),
    "move_left":     ("u", "a"),
    "move_right":    ("u", "d"),
    "look_up":       ("i", "q"),
    "look_down":     ("k", "q"),
    "look_left":     ("j", "q"),
    "look_right":    ("l", "q"),
    "static":        ("u", "q"),
}


def draw_eval_action_overlay(video, action_str):
    """Overlay WASD (movement) + IJKL (view) key icons onto an eval video.

    Mouse directions are no longer drawn as a mouse icon; they use the same keyboard-key style as WASD.

    Layout::

        ┌─────── Move ───────┐    ┌─────── Look ───────┐
                [W]                        [I]
           [A] [S] [D]               [J]  [K]  [L]

    Args:
        video: (T, H, W, C) uint8 numpy array
        action_str: trajectory name, e.g. "move_forward", "look_left", "static"
    Returns:
        overlaid_video: uint8 numpy array of the same shape
    """
    mouse_key, keyboard_key = _TRAJ_KEY_MAP.get(action_str, ("u", "q"))
    video = np.asarray(video).copy()
    T, H, W, C = video.shape

    layout = _layout_from_frame_size(W, H)
    kw = layout["key_w"]
    kh = layout["key_h"]
    sp = layout["spacing"]
    radius = layout["radius"]
    font_scale = layout["font_scale"]
    font_thickness = layout["font_thickness"]
    bottom_margin = layout["bottom_margin"]
    first_row_extra = layout["first_row_extra"]

    # definitions of the two key groups: (display_label, input_key, row, col_in_row)
    # row=0 -> top row (single key, centered), row=1 -> bottom row (three keys)
    move_keys = [
        ("W", "w", 0, 0),
        ("A", "a", 1, 0),
        ("S", "s", 1, 1),
        ("D", "d", 1, 2),
    ]
    look_keys = [
        ("I", "i", 0, 0),
        ("J", "j", 1, 0),
        ("K", "k", 1, 1),
        ("L", "l", 1, 2),
    ]

    def _group_positions(group, anchor_x):
        """Compute (x, y, w, h, is_active) for each key of a WASD-style layout."""
        row1_y = H - bottom_margin - kh * 2 - sp - first_row_extra
        row2_y = H - bottom_margin - kh
        positions = []
        for label, key, row, col in group:
            is_active = (key == keyboard_key) or (key == mouse_key)
            if row == 0:
                x = anchor_x + kw + sp // 2 - kw // 2
                y = row1_y
            else:
                x = anchor_x + col * (kw + sp)
                y = row2_y
            positions.append((label, x, y, kw, kh, is_active))
        return positions

    group_width = kw * 3 + sp * 2
    gap_between_groups = kw + sp * 2

    total_width = group_width * 2 + gap_between_groups
    start_x = (W - total_width) // 2

    move_pos = _group_positions(move_keys, start_x)
    look_pos = _group_positions(look_keys, start_x + group_width + gap_between_groups)

    all_keys = move_pos + look_pos

    for t in range(T):
        frame = video[t]
        for label, x, y, w_k, h_k, is_active in all_keys:
            color = (0, 255, 0) if is_active else (200, 200, 200)
            alpha = 0.8 if is_active else 0.5
            draw_rounded_rectangle(frame, (x, y), (x + w_k, y + h_k), color,
                                   radius=radius, alpha=alpha)
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            tx = x + (w_k - text_size[0]) // 2
            ty = y + (h_k + text_size[1]) // 2
            cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 0, 0), font_thickness)

    return video


def draw_interactive_action_overlay(video, mouse_keys, keyboard_keys):
    """Overlay WASD + IJKL key icons frame by frame onto an interactive video.

    Same layout as draw_eval_action_overlay, but each frame can have different active keys.

    Args:
        video: (T, H, W, C) uint8 numpy array
        mouse_keys: list[str] of length T_latent, the mouse key (i/k/j/l/u) for each latent frame
        keyboard_keys: list[str] of length T_latent, the keyboard key (w/a/s/d/q/...) for each latent frame
    Returns:
        overlaid_video: uint8 numpy array of the same shape
    """
    video = np.asarray(video).copy()
    T, H, W, C = video.shape

    n_latent = len(mouse_keys)
    latent_to_pixel = _build_latent_to_pixel_map(T, n_latent)

    layout = _layout_from_frame_size(W, H)
    kw = layout["key_w"]
    kh = layout["key_h"]
    sp = layout["spacing"]
    radius = layout["radius"]
    font_scale = layout["font_scale"]
    font_thickness = layout["font_thickness"]
    bottom_margin = layout["bottom_margin"]
    first_row_extra = layout["first_row_extra"]

    move_labels = [("W", "w", 0, 0), ("A", "a", 1, 0), ("S", "s", 1, 1), ("D", "d", 1, 2)]
    look_labels = [("I", "i", 0, 0), ("J", "j", 1, 0), ("K", "k", 1, 1), ("L", "l", 1, 2)]

    row1_y = H - bottom_margin - kh * 2 - sp - first_row_extra
    row2_y = H - bottom_margin - kh
    group_width = kw * 3 + sp * 2
    gap_between_groups = kw + sp * 2
    total_width = group_width * 2 + gap_between_groups
    start_x = (W - total_width) // 2

    def _key_positions(group, anchor_x):
        positions = []
        for label, key, row, col in group:
            if row == 0:
                x = anchor_x + kw + sp // 2 - kw // 2
                y = row1_y
            else:
                x = anchor_x + col * (kw + sp)
                y = row2_y
            positions.append((label, key, x, y))
        return positions

    move_pos = _key_positions(move_labels, start_x)
    look_pos = _key_positions(look_labels, start_x + group_width + gap_between_groups)
    all_keys = move_pos + look_pos

    for t in range(T):
        lat_idx = latent_to_pixel[t]
        cur_mouse = mouse_keys[lat_idx]
        cur_kb = keyboard_keys[lat_idx]
        frame = video[t]
        for label, key, x, y in all_keys:
            is_active = (key == cur_kb) or (key == cur_mouse)
            color = (0, 255, 0) if is_active else (200, 200, 200)
            alpha = 0.8 if is_active else 0.5
            draw_rounded_rectangle(frame, (x, y), (x + kw, y + kh), color,
                                   radius=radius, alpha=alpha)
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            tx = x + (kw - text_size[0]) // 2
            ty = y + (kh + text_size[1]) // 2
            cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 0, 0), font_thickness)

    return video


def _build_latent_to_pixel_map(num_pixel_frames, num_latent_frames):
    """Map each pixel frame index to its corresponding latent frame index.

    Convention: first latent frame → 1 pixel frame, subsequent → 4 each.
    """
    mapping = []
    for lat_i in range(num_latent_frames):
        n = 1 if lat_i == 0 else 4
        mapping.extend([lat_i] * n)
    if len(mapping) < num_pixel_frames:
        mapping.extend([num_latent_frames - 1] * (num_pixel_frames - len(mapping)))
    return mapping[:num_pixel_frames]
    # video: [num_frames, height, width, channels], uint8, numpy array
    # action_per_frame: [num_frames, action_dim], torch tensor
    # return the video with action icons
    if mode == 'universal':
        # Create a copy of the video to avoid modifying the original
        video_with_icon = video.copy()
        
        # Convert action_per_frame to numpy if it's a torch tensor
        # Convert to float32 first to handle BFloat16 and other unsupported types
        
        # argmax the action_per_frame to get the action index
        action_index = np.argmax(action_per_frame, axis=1)
        
        # Calculate font scale based on video height for better visibility
        font_scale = max(1.0, video.shape[1] / 400.0)  # Scale font with video height
        thickness = max(2, int(font_scale * 2))  # Thickness scales with font
        
        # Add text to each frame of the video
        for i in range(video.shape[0]):
            action = int(action_index[i])
            frame = video_with_icon[i]
            
            # Prepare text
            text = f"Action: {action}"
            
            # Get text size to calculate background rectangle
            # Use FONT_HERSHEY_DUPLEX for a bolder appearance
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness
            )
            
            # Position: top-left corner with some padding
            padding = 10
            x = padding
            y = text_height + padding + baseline
            
            # Draw semi-transparent background rectangle for better visibility
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (x - padding, y - text_height - padding),
                (x + text_width + padding, y + baseline + padding),
                (0, 0, 0),  # Black background
                -1  # Filled rectangle
            )
            # Blend overlay with original frame (alpha blending)
            alpha = 0.7  # Transparency of background
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            
            # Draw white text with black outline for maximum contrast
            # First draw black outline (thicker)
            cv2.putText(
                frame, text,
                (x, y),
                cv2.FONT_HERSHEY_DUPLEX,
                font_scale,
                (0, 0, 0),  # Black color for outline
                thickness + 2,
                cv2.LINE_AA
            )
            # Then draw white text on top
            cv2.putText(
                frame, text,
                (x, y),
                cv2.FONT_HERSHEY_DUPLEX,
                font_scale,
                (255, 255, 255),  # White color
                thickness,
                cv2.LINE_AA
            )
            
            video_with_icon[i] = frame
            return video_with_icon
    else:
        raise ValueError(f"Invalid mode: {mode}")



def visualize_trajectory(poses, output_file="trajectory_vis.html"):
    """
    Visualizes camera trajectory and orientation using Plotly.
    Converts from OpenCV format (Y down, Z forward) to Z-up format for visualization.
    
    Args:
        poses: (N, 4, 4) numpy array representing camera-to-world transformation matrices.
        output_file: Name of the output HTML file.
    """
    import plotly.graph_objects as go  # lazy: only this fn needs plotly

    # Coordinate transformation: (x, y, z) -> (x, z, -y)
    # This maps:
    #   +X (Right) -> +X (Right)
    #   +Y (Down)  -> -Z (Down) => So -Y (Up) -> +Z (Up)
    #   +Z (Forward)-> +Y (Forward)
    def transform_point(p):
        return np.array([p[0], p[2], -p[1]])

    def transform_points(pts):
        # pts: (N, 3)
        return np.stack([pts[:, 0], pts[:, 2], -pts[:, 1]], axis=1)

    # 1. Transform trajectory path
    raw_translations = poses[:, :3, 3] # (N, 3)
    translations = transform_points(raw_translations)
    
    # Create the trace for the trajectory path
    traj_trace = go.Scatter3d(
        x=translations[:, 0],
        y=translations[:, 1],
        z=translations[:, 2],
        mode='lines',
        line=dict(
            color=np.arange(len(translations)), 
            colorscale='Viridis', 
            width=2
        ),
        name='Trajectory Path',
        showlegend=False
    )
    
    # 2. Create Frustums and 3D Markers
    # Calculate scene scale based on trajectory extent
    min_pt = np.min(translations, axis=0)
    max_pt = np.max(translations, axis=0)
    # Use the diagonal of the bounding box as the scene size measure
    scene_size = np.linalg.norm(max_pt - min_pt)
    if scene_size < 1e-6:
        scene_size = 1.0
    
    # Set scale to be a fraction of the scene size (e.g., 5%)
    scale = scene_size * 0.05

    def create_octahedron_mesh(centers, radius, name, intensities=None, colorscale='Viridis', color=None):
        """
        Creates a Mesh3d trace representing octahedrons (diamond shape) at given centers.
        This ensures the markers have a fixed 3D size relative to the scene.
        """
        # Vertices of a unit octahedron
        # 6 vertices
        v_unit = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ]) * radius
        
        # We need to replicate these for each center
        all_x, all_y, all_z = [], [], []
        all_i, all_j, all_k = [], [], []
        all_intensity = []
        
        N_verts = 6
        
        for k, center in enumerate(centers):
            # Vertices
            v_curr = v_unit + center
            all_x.extend(v_curr[:, 0])
            all_y.extend(v_curr[:, 1])
            all_z.extend(v_curr[:, 2])
            
            if intensities is not None:
                all_intensity.extend([intensities[k]] * N_verts)
            
            # Indices
            offset = k * N_verts
            # Faces
            faces = [
                # Top
                [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
                # Bottom
                [0, 5, 2], [2, 5, 1], [1, 5, 3], [3, 5, 0]
            ]
            for face in faces:
                all_i.append(face[0] + offset)
                all_j.append(face[1] + offset)
                all_k.append(face[2] + offset)
                
        mesh_kwargs = dict(
            x=all_x, y=all_y, z=all_z,
            i=all_i, j=all_j, k=all_k,
            name=name,
            flatshading=True
        )
        
        if intensities is not None:
            mesh_kwargs['intensity'] = all_intensity
            mesh_kwargs['colorscale'] = colorscale
            mesh_kwargs['showscale'] = False
        else:
            mesh_kwargs['color'] = color if color else 'blue'

        return go.Mesh3d(**mesh_kwargs)

    # Trajectory points (Small Blue) - Kept as 3D Mesh
    # Radius relative to scale. Frustum width is ~scale/2. 
    traj_markers = create_octahedron_mesh(
        translations, scale * 0.1, 'Images', 
        intensities=np.arange(len(translations)), 
        colorscale='Viridis'
    )

    # Start point (Scatter3d with text/legend)
    start_trace = go.Scatter3d(
        x=[translations[0, 0]],
        y=[translations[0, 1]],
        z=[translations[0, 2]],
        mode='markers+text',
        marker=dict(size=8, color='green', symbol='diamond'),
        text=['Start'],
        textposition="top center",
        name='Start'
    )

    # End point (Scatter3d with text/legend)
    end_trace = go.Scatter3d(
        x=[translations[-1, 0]],
        y=[translations[-1, 1]],
        z=[translations[-1, 2]],
        mode='markers+text',
        marker=dict(size=8, color='red', symbol='diamond'),
        text=['End'],
        textposition="top center",
        name='End'
    )
    
    data = [traj_trace, traj_markers, start_trace, end_trace] 
    
    # Pre-define frustum corners in CAMERA frame (OpenCV: X right, Y down, Z forward)
    # Tips: In OpenCV cam frame, Look direction is +Z. 
    # Frustum should extend into +Z.
    d = scale
    w = scale * 0.5
    h = scale * 0.5
    
    corners_cam = np.array([
        [0, 0, 0],    # Center
        [-w, -h, d],  # Top-Left (in image space? No, -Y is Up in normal space, but here Y is down)
                      # Let's verify OpenCV frustum:
                      # Y is down. -h is "up" in image. +h is "down".
                      # X is right.
        [w, -h, d],
        [w, h, d],
        [-w, h, d]
    ])
    
    # We will accumulate all lines to avoid adding thousands of traces which slows down Plotly
    line_x = []
    line_y = []
    line_z = []
    line_color = []
    
    # Helper to append a line segment separated by None
    def add_line(p1, p2, color_val):
        line_x.extend([p1[0], p2[0], None])
        line_y.extend([p1[1], p2[1], None])
        line_z.extend([p1[2], p2[2], None])
        line_color.extend([color_val, color_val, color_val])

    for i in range(len(poses)):
        # Camera-to-World
        R = poses[i, :3, :3]
        t = poses[i, :3, 3]
        
        # Transform frustum corners to World Frame (Original Coordinates)
        corners_world_orig = (R @ corners_cam.T).T + t
        
        # Now transform World Frame (OpenCV style) to Viz Frame (Z-up)
        corners_viz = transform_points(corners_world_orig)
        
        center = corners_viz[0]
        # Corners 1-4
        c1, c2, c3, c4 = corners_viz[1], corners_viz[2], corners_viz[3], corners_viz[4]
        
        # Edges from center
        add_line(center, c1, i)
        add_line(center, c2, i)
        add_line(center, c3, i)
        add_line(center, c4, i)
        
        # Base rectangle
        add_line(c1, c2, i)
        add_line(c2, c3, i)
        add_line(c3, c4, i)
        add_line(c4, c1, i)

    frustum_trace = go.Scatter3d(
        x=line_x,
        y=line_y,
        z=line_z,
        mode='lines',
        line=dict(
            color=line_color,
            colorscale='Viridis',
            width=1,
            showscale=True,
            colorbar=dict(
                title='Frame Index',
                x=0,
                xanchor='right',
                tickfont=dict(size=8),
            ),
        ),
        name='Cameras',
        showlegend=False
    )
    data.append(frustum_trace)

    # 3. Calculate consistent axis ranges
    # Include frustum points in bounds calculation so they don't get clipped
    frustum_x = np.array([v for v in line_x if v is not None])
    frustum_y = np.array([v for v in line_y if v is not None])
    frustum_z = np.array([v for v in line_z if v is not None])

    all_x = np.concatenate([translations[:, 0], frustum_x])
    all_y = np.concatenate([translations[:, 1], frustum_y])
    all_z = np.concatenate([translations[:, 2], frustum_z])
    
    min_x, max_x = np.min(all_x), np.max(all_x)
    min_y, max_y = np.min(all_y), np.max(all_y)
    min_z, max_z = np.min(all_z), np.max(all_z)
    
    mid_x = (min_x + max_x) / 2
    mid_y = (min_y + max_y) / 2
    mid_z = (min_z + max_z) / 2
    
    range_x = max_x - min_x
    range_y = max_y - min_y
    range_z = max_z - min_z
    
    max_range = max(range_x, range_y, range_z)
    if max_range == 0:
        max_range = 1.0 # Fallback for single point
        
    half_range = max_range / 2 * 1.1 # Add 10% padding
    
    scene_matches = dict(
        xaxis=dict(
            range=[mid_x - half_range, mid_x + half_range],
            title=dict(text='X (Right)', font=dict(size=10)),
            tickfont=dict(size=8),
        ),
        yaxis=dict(
            range=[mid_y - half_range, mid_y + half_range],
            title=dict(text='Y (Forward)', font=dict(size=10)),
            tickfont=dict(size=8),
        ),
        zaxis=dict(
            range=[mid_z - half_range, mid_z + half_range],
            title=dict(text='Z (Up)', font=dict(size=10)),
            tickfont=dict(size=8),
        ),
        aspectmode='cube',  # Forces the box to be a cube
    )

    layout = go.Layout(
        title=dict(text='Trajectory Visualization', font=dict(size=12)),
        scene=scene_matches,
        margin=dict(l=80, r=20, b=20, t=40),
        font=dict(size=10),
    )
    
    fig = go.Figure(data=data, layout=layout)
    
    # 4. Set initial camera view: look along the Y axis, X increasing to the right, tilted slightly upward
    # camera sits on the -Y side (behind), looking at the scene center, slightly above it
    fig.update_layout(scene_camera=dict(
        eye=dict(x=mid_x + 5.0 * half_range, y=mid_y - 20 * half_range, z=mid_z + 10.0 * half_range),
        center=dict(x=mid_x, y=mid_y, z=mid_z),
        up=dict(x=0, y=0, z=1)
    ))
    
    fig.write_html(output_file)
    print(f"Visualization saved to {output_file}")


# ---------------------------------------------------------------------------
# Video-gallery HTML generation (videos + paired pose trajectories + optional label images)
# ---------------------------------------------------------------------------

def _parse_rank_batch(stem: str):
    """
    Parse rank and batch from a video or npy filename.
    e.g.: video_rank00_batch01 -> (0, 1); action_sequence_rank00_batch01 -> (0, 1)
    Returns (rank, batch) or None.
    """
    m = re.search(r"rank(\d+)_batch(\d+)", stem, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def _find_poses_for_video(video_path: Path, npy_names: set) -> Optional[Path]:
    """
    Infer the paired poses file path from a video filename.
    Rule: same-stem match, e.g. video_0.mp4 -> video_0.npy
    """
    stem = video_path.stem
    candidate = f"{stem}.npy"
    if candidate in npy_names:
        return video_path.parent / candidate
    return None


def _find_label_image_for_video(video_path: Path, image_names: set) -> Optional[Path]:
    """
    Infer the matching label image path from a video filename (optional).
    Rule: video_rankXX_batchYY.mp4 -> video_with_icon_rankXX_batchYY.png
    """
    stem = video_path.stem
    if stem.startswith("video_"):
        suffix = stem[6:]
        for ext in VIDEO_GALLERY_IMAGE_EXTENSIONS:
            candidate = f"video_with_icon_{suffix}{ext}"
            if candidate in image_names:
                return video_path.parent / candidate
    for ext in VIDEO_GALLERY_IMAGE_EXTENSIONS:
        if (stem + ext) in image_names:
            return video_path.parent / (stem + ext)
    return None


def generate_video_gallery_html(
    folder: Union[str, Path],
    output_html: Optional[Union[str, Path]] = None,
    title: str = "Video Gallery",
    with_trajectory: bool = True,
    with_label_image: bool = False,
) -> Path:
    """
    Scan a folder for videos and paired poses (.npy), generate a trajectory HTML per video,
    and build a master gallery HTML: video on the left, trajectory (and optional label image) on the right.

    Args:
        folder: folder containing the videos and action_sequence_*.npy (or poses_*.npy) files
        output_html: output HTML path; defaults to folder/index.html
        title: page title
        with_trajectory: whether to render and embed trajectory visualizations from the .npy files
        with_label_image: whether to also show video_with_icon_* images when present

    Returns:
        Path of the generated HTML file
    """
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"not a valid directory: {folder}")

    if output_html is None:
        output_html = folder / "index.html"
    else:
        output_html = Path(output_html).resolve()

    videos = []
    npy_names = set()
    image_names = set()
    for f in folder.iterdir():
        if not f.is_file():
            continue
        if f.suffix.lower() in VIDEO_GALLERY_VIDEO_EXTENSIONS:
            videos.append(f)
        elif f.suffix.lower() == VIDEO_GALLERY_NPY_EXTENSION:
            npy_names.add(f.name)
        elif f.suffix.lower() in VIDEO_GALLERY_IMAGE_EXTENSIONS:
            image_names.add(f.name)

    videos.sort(key=lambda p: p.name)

    def _write_trajectory_html(poses: np.ndarray, out_path: Path) -> None:
        # visualize_trajectory is defined in this same file
        visualize_trajectory(poses, output_file=str(out_path))

    rows = []
    for video_path in videos:
        v_rel = video_path.name
        stem = video_path.stem
        rb = _parse_rank_batch(stem)
        suffix = f"rank{rb[0]:02d}_batch{rb[1]:02d}" if rb else stem

        traj_html_rel = None
        if with_trajectory:
            poses_path = _find_poses_for_video(video_path, npy_names)
            if poses_path is not None:
                try:
                    poses = np.load(poses_path)
                    if poses.ndim == 3 and poses.shape[1:] == (4, 4):
                        traj_name = f"traj_{suffix}.html"
                        traj_path = folder / traj_name
                        _write_trajectory_html(poses, traj_path)
                        traj_html_rel = traj_name
                except Exception as e:
                    print(f"Warning: could not load or visualize {poses_path}: {e}")

        if traj_html_rel is not None:
            traj_block = f'<iframe src="{traj_html_rel}" class="traj-iframe" title="Trajectory"></iframe>'
        else:
            traj_block = '<span class="no-traj">no trajectory data</span>'

        if with_label_image:
            label_path = _find_label_image_for_video(video_path, image_names)
            img_block = f'<img src="{label_path.name}" alt="label" class="label-img" />' if label_path is not None else ""
        else:
            img_block = ""

        row_parts = [
            f'<div class="media"><video src="{v_rel}" controls loop muted playsinline></video></div>',
            f'<div class="traj">{traj_block}</div>',
        ]
        if with_label_image and img_block:
            row_parts.append(f'<div class="label">{img_block}</div>')
        row_parts.append(f'<div class="caption">{v_rel}</div>')
        rows.append('<div class="item">' + "".join(row_parts) + "</div>")

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ font-family: system-ui, sans-serif; margin: 0; padding: 20px; background: #1a1a1a; color: #e0e0e0; }}
        h1 {{ margin-top: 0; }}
        .gallery {{ display: flex; flex-direction: column; gap: 32px; max-width: 1600px; margin: 0 auto; }}
        .item {{
            display: flex;
            align-items: flex-start;
            gap: 20px;
            padding: 16px;
            background: #252525;
            border-radius: 12px;
            flex-wrap: wrap;
        }}
        .item .media {{ flex: 0 0 auto; }}
        .item .media video {{
            max-width: 480px;
            max-height: 360px;
            width: 100%;
            height: auto;
            border-radius: 8px;
            background: #000;
        }}
        .item .traj {{ flex: 0 0 auto; min-width: 320px; }}
        .item .traj .traj-iframe {{
            width: 420px;
            height: 420px;
            border: 1px solid #444;
            border-radius: 8px;
            background: #111;
        }}
        .item .traj .no-traj {{ color: #888; font-style: italic; }}
        .item .label {{ flex: 0 0 auto; }}
        .item .label .label-img {{
            max-width: 240px;
            max-height: 360px;
            width: 100%;
            height: auto;
            object-fit: contain;
            border-radius: 8px;
            border: 1px solid #444;
        }}
        .item .caption {{
            width: 100%;
            font-size: 12px;
            color: #999;
            margin-top: 4px;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p>{len(rows)} videos - generated from <code>{folder.name}</code></p>
    <div class="gallery">
        {"".join(rows)}
    </div>
</body>
</html>
"""
    output_html.write_text(html_content, encoding="utf-8")
    return output_html