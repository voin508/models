import os
import io
import tempfile
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import streamlit as st
import plotly.graph_objects as go

from core.core_lut import LoRIA3DLUT, cp_residual_to_lut, create_identity_lut
from export.export_cube import write_cube


def _pick_device(user_choice: str) -> str:
    if user_choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if user_choice == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return user_choice


@st.cache_resource(show_spinner=False)
def load_model_from_ckpt(ckpt_path: str, device: str) -> Tuple[LoRIA3DLUT, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]
    G = cfg["model"]["G"]
    K = cfg["model"]["K"]
    R = cfg["model"]["R"]
    model = LoRIA3DLUT(G=G, K=K, R=R).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model, cfg


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t


def tensor_to_image_np(t: torch.Tensor) -> np.ndarray:
    x = t.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return x


@torch.no_grad()
def compute_image_adaptive_lut(
    model: LoRIA3DLUT,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # x: [1,3,H,W] in [0,1]
    x_lr = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
    alpha = model.weight_pred(x_lr)  # [1,K]
    u, v, w, c = model.resid_pred(x_lr)  # [1,R,G], [1,R,3]
    fused = model.fuse_bases(alpha)  # [1,G,G,G,3]
    delta = cp_residual_to_lut(u, v, w, c)  # [1,G,G,G,3]
    # CRITICAL: Apply the same amplification as in training (core_lut.py line 176)
    delta = delta * 1.0
    Lstar = fused + delta  # [1,G,G,G,3]
    return Lstar, alpha, delta, fused, u, v, w, c


@torch.no_grad()
def compute_image_adaptive_lut_with_ablation(
    model: LoRIA3DLUT,
    x: torch.Tensor,
    alpha_scale: Optional[torch.Tensor] = None,
    residual_scale: float = 1.0,
    delta_amplifier: float = 1.0,
    rank_scales: Optional[torch.Tensor] = None,
    enable_bases: Optional[torch.Tensor] = None,
    enable_residual: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute LUT with ablation-style controls.
    
    Args:
        model: LoRIA3DLUT model
        x: Input image [1,3,H,W]
        alpha_scale: Manual alpha weights [K] to override predictions
        residual_scale: Global scale for residual delta
        delta_amplifier: Amplification factor for delta (default 1.0)
        rank_scales: Scale each CP rank contribution [R]
        enable_bases: Enable/disable each base [K]
        enable_residual: Enable/disable residual correction
    
    Returns:
        Lstar: Final LUT
        alpha: Alpha weights used
        delta: Residual delta
        fused: Fused base LUTs
    """
    x_lr = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
    
    # Get predicted weights
    alpha_pred = model.weight_pred(x_lr)  # [1,K]
    
    # Apply manual alpha scales if provided
    if alpha_scale is not None:
        alpha = alpha_scale.unsqueeze(0)  # [1,K]
    else:
        alpha = alpha_pred
    
    # Apply base enable/disable masks
    if enable_bases is not None:
        alpha = alpha * enable_bases.unsqueeze(0)  # [1,K]
    
    # Fused bases
    fused = model.fuse_bases(alpha)  # [1,G,G,G,3]
    
    # Compute residual if enabled
    if enable_residual:
        u, v, w, c = model.resid_pred(x_lr)  # [1,R,G], [1,R,3]
        
        # Apply rank scaling if provided
        if rank_scales is not None:
            rank_scales = rank_scales.view(1, -1, 1, 1, 1)  # [1,R,1,1,1]
            u = u * rank_scales
            v = v * rank_scales
            w = w * rank_scales
        
        delta = cp_residual_to_lut(u, v, w, c)  # [1,G,G,G,3]
        delta = delta * delta_amplifier  # Amplification
        delta = delta * residual_scale  # Global scale
    else:
        delta = torch.zeros_like(fused)
    
    Lstar = fused + delta
    return Lstar, alpha, delta, fused


@torch.no_grad()
def apply_lut_to_image(model: LoRIA3DLUT, x: torch.Tensor, Lstar: torch.Tensor) -> torch.Tensor:
    return model.apply_lut(x, Lstar)


def make_lut_scatter3d(
    lut_np: np.ndarray,
    stride: int = 2,
    slice_axis: Optional[str] = None,
    slice_index: int = 0,
    point_size: int = 3,
    point_opacity: float = 0.7,
    show_frame: bool = True,
    camera_preset: str = "Free Rotation",
    title: str = "3D LUT",
    viz_mode: str = "Standard",
    identity_lut: Optional[np.ndarray] = None,
) -> go.Figure:
    # lut_np: [G,G,G,3] in [0,1]
    G = lut_np.shape[0]
    coords = range(0, G, max(1, int(stride)))
    xs, ys, zs, colors, hover_texts = [], [], [], [], []

    # Determine if we're showing deviation
    show_deviation = (viz_mode == "Deviation" and identity_lut is not None)
    
    for r in coords:
        for g in coords:
            for b in coords:
                if slice_axis == "R" and r != slice_index:
                    continue
                if slice_axis == "G" and g != slice_index:
                    continue
                if slice_axis == "B" and b != slice_index:
                    continue
                
                # Input position (normalized)
                r_in = r / (G - 1)
                g_in = g / (G - 1)
                b_in = b / (G - 1)
                
                xs.append(r_in)
                ys.append(g_in)
                zs.append(b_in)
                
                # Output color
                v = np.clip(lut_np[r, g, b], 0.0, 1.0)
                
                if show_deviation:
                    # Show deviation magnitude
                    dev = lut_np[r, g, b] - identity_lut[r, g, b]
                    mag = np.linalg.norm(dev)
                    # Color by deviation: gray=small, orange=medium, red=large
                    if mag > 0.3:
                        color_rgb = [255, 0, 0]  # Red
                    elif mag > 0.1:
                        color_rgb = [255, 128, 0]  # Orange
                    else:
                        color_rgb = [128, 128, 128]  # Gray
                    colors.append(f"rgb({color_rgb[0]},{color_rgb[1]},{color_rgb[2]})")
                    hover_texts.append(f"In: ({r_in:.2f},{g_in:.2f},{b_in:.2f})<br>Out: ({v[0]:.2f},{v[1]:.2f},{v[2]:.2f})<br>Deviation: {mag:.3f}")
                else:
                    v_int = (v * 255.0).astype(int)
                    colors.append(f"rgb({v_int[0]},{v_int[1]},{v_int[2]})")
                    hover_texts.append(f"In: ({r_in:.2f},{g_in:.2f},{b_in:.2f})<br>Out: ({v[0]:.2f},{v[1]:.2f},{v[2]:.2f})")

    # Create scatter plot
    data = [go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="markers",
        marker=dict(size=point_size, color=colors, opacity=point_opacity, line=dict(width=0)),
        text=hover_texts,
        hoverinfo='text',
        name='LUT Points'
    )]
    
    # Add cube frame if requested
    if show_frame:
        edges = [
            # Bottom face
            ([0,1],[0,0],[0,0]), ([1,1],[0,1],[0,0]), ([1,0],[1,1],[0,0]), ([0,0],[1,0],[0,0]),
            # Top face
            ([0,1],[0,0],[1,1]), ([1,1],[0,1],[1,1]), ([1,0],[1,1],[1,1]), ([0,0],[1,0],[1,1]),
            # Vertical edges
            ([0,0],[0,0],[0,1]), ([1,1],[0,0],[0,1]), ([1,1],[1,1],[0,1]), ([0,0],[1,1],[0,1])
        ]
        
        for edge in edges:
            data.append(go.Scatter3d(
                x=edge[0], y=edge[1], z=edge[2],
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Add diagonal identity line in Standard mode
    if viz_mode == "Standard":
        data.append(go.Scatter3d(
            x=[0, 1], y=[0, 1], z=[0, 1],
            mode='lines',
            line=dict(color='yellow', width=4, dash='dash'),
            name='Identity Line',
            hoverinfo='name'
        ))

    fig = go.Figure(data=data)
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        scene=dict(
            xaxis=dict(title='R (Red)', range=[0, 1], showbackground=True, backgroundcolor='rgb(230, 230, 230)'),
            yaxis=dict(title='G (Green)', range=[0, 1], showbackground=True, backgroundcolor='rgb(230, 230, 230)'),
            zaxis=dict(title='B (Blue)', range=[0, 1], showbackground=True, backgroundcolor='rgb(230, 230, 230)'),
            camera=get_camera_settings(camera_preset),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=600,
        showlegend=(viz_mode == "Standard"),
        hovermode='closest'
    )
    
    return fig


def make_residual_scatter3d(
    delta_np: np.ndarray,
    stride: int = 2,
    slice_axis: Optional[str] = None,
    slice_index: int = 0,
    point_size: int = 3,
    point_opacity: float = 0.7,
    show_frame: bool = True,
    camera_preset: str = "Free Rotation",
) -> go.Figure:
    # delta_np: [G,G,G,3]
    mag = np.linalg.norm(delta_np, axis=-1)  # [G,G,G]
    G = mag.shape[0]
    coords = range(0, G, max(1, int(stride)))
    xs, ys, zs, vals, hover_texts = [], [], [], [], []
    for r in coords:
        for g in coords:
            for b in coords:
                if slice_axis == "R" and r != slice_index:
                    continue
                if slice_axis == "G" and g != slice_index:
                    continue
                if slice_axis == "B" and b != slice_index:
                    continue
                
                r_in = r / (G - 1)
                g_in = g / (G - 1)
                b_in = b / (G - 1)
                
                xs.append(r_in)
                ys.append(g_in)
                zs.append(b_in)
                vals.append(mag[r, g, b])
                hover_texts.append(f"Pos: ({r_in:.2f},{g_in:.2f},{b_in:.2f})<br>Magnitude: {mag[r, g, b]:.4f}")
    
    data = [go.Scatter3d(
        x=xs, y=ys, z=zs, mode="markers",
        marker=dict(size=point_size, color=vals, colorscale="Viridis", showscale=True, 
                    opacity=point_opacity, line=dict(width=0)),
        text=hover_texts,
        hoverinfo='text',
        name='Residual Magnitude'
    )]
    
    # Add cube frame if requested
    if show_frame:
        edges = [
            # Bottom face
            ([0,1],[0,0],[0,0]), ([1,1],[0,1],[0,0]), ([1,0],[1,1],[0,0]), ([0,0],[1,0],[0,0]),
            # Top face
            ([0,1],[0,0],[1,1]), ([1,1],[0,1],[1,1]), ([1,0],[1,1],[1,1]), ([0,0],[1,0],[1,1]),
            # Vertical edges
            ([0,0],[0,0],[0,1]), ([1,1],[0,0],[0,1]), ([1,1],[1,1],[0,1]), ([0,0],[1,1],[0,1])
        ]
        
        for edge in edges:
            data.append(go.Scatter3d(
                x=edge[0], y=edge[1], z=edge[2],
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    fig = go.Figure(data=data)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='R (Red)', range=[0, 1], showbackground=True, backgroundcolor='rgb(230, 230, 230)'),
            yaxis=dict(title='G (Green)', range=[0, 1], showbackground=True, backgroundcolor='rgb(230, 230, 230)'),
            zaxis=dict(title='B (Blue)', range=[0, 1], showbackground=True, backgroundcolor='rgb(230, 230, 230)'),
            camera=get_camera_settings(camera_preset),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=600,
        showlegend=False,
        hovermode='closest'
    )
    return fig


def make_residual_plane_heatmap(delta_np: np.ndarray, axis: str, idx: int) -> go.Figure:
    # delta_np: [G,G,G,3]
    mag = np.linalg.norm(delta_np, axis=-1)  # [G,G,G]
    G = mag.shape[0]
    idx = int(np.clip(idx, 0, G - 1))
    if axis == "R":
        plane = mag[idx, :, :]  # GxG over G,B
        xlab, ylab = "B", "G"
    elif axis == "G":
        plane = mag[:, idx, :]  # GxG over R,B
        xlab, ylab = "B", "R"
    else:  # "B"
        plane = mag[:, :, idx]  # GxG over R,G
        xlab, ylab = "G", "R"
    fig = go.Figure(data=go.Heatmap(z=plane, colorscale="Viridis"))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30), height=500,
                      xaxis_title=xlab, yaxis_title=ylab)
    return fig


def cube_bytes_from_lut(lut_np: np.ndarray, title: str = "LoR-IA3DLUT") -> bytes:
    # Reuse write_cube -> temp file -> read bytes for download
    with tempfile.NamedTemporaryFile(delete=False, suffix=".cube") as tmp:
        tmp_path = tmp.name
    try:
        write_cube(lut_np, tmp_path, title=title)
        with open(tmp_path, "rb") as f:
            data = f.read()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return data


def create_identity_lut_for_viz(G: int) -> torch.Tensor:
    """Create a perfect identity LUT for reference"""
    lut = torch.zeros(1, G, G, G, 3)
    for i in range(G):
        for j in range(G):
            for k in range(G):
                lut[0, i, j, k, 0] = i / (G - 1)
                lut[0, i, j, k, 1] = j / (G - 1)
                lut[0, i, j, k, 2] = k / (G - 1)
    return lut


def get_camera_settings(preset: str) -> dict:
    """Get camera settings for different view presets"""
    cameras = {
        "Free Rotation": dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        "Front": dict(eye=dict(x=0, y=0, z=2.5)),
        "Side": dict(eye=dict(x=2.5, y=0, z=0)),
        "Top": dict(eye=dict(x=0, y=2.5, z=0)),
        "Diagonal": dict(eye=dict(x=2, y=2, z=2))
    }
    return cameras.get(preset, cameras["Free Rotation"])


def main():
    st.set_page_config(page_title="🎨 LoR-IA3DLUT Viewer", layout="wide")
    st.title("🎨 LoR-IA3DLUT Viewer")
    st.markdown("**Low-Rank Image-Adaptive 3D LUT** | Interactive visualization and analysis tool")

    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Get all checkpoint files
        saved_model_dir = "saved_model"
        ckpt_files = []
        if os.path.isdir(saved_model_dir):
            ckpt_files = sorted([f for f in os.listdir(saved_model_dir) if f.endswith('.ckpt')])
        
        if ckpt_files:
            default_index = 0
            if "best.ckpt" in ckpt_files:
                default_index = ckpt_files.index("best.ckpt")
            
            selected_ckpt = st.selectbox(
                "Model Selection",
                options=ckpt_files,
                index=default_index,
                help="Select a trained model checkpoint"
            )
            ckpt_path = os.path.join(saved_model_dir, selected_ckpt)
        else:
            st.warning("⚠️ No checkpoint files found in saved_model/")
            ckpt_path = st.text_input("Checkpoint path", value="saved_model/best.ckpt")
        
        device_choice = st.radio("Device", options=["auto", "cuda", "cpu"], index=0, horizontal=True)
        
        st.markdown("---")
        st.subheader("📊 Visualization Settings")
        viz_stride = st.slider("3D Viz Stride", min_value=1, max_value=4, value=2, step=1,
                               help="Lower = more points, slower rendering")
        point_size = st.slider("Point Size", min_value=1, max_value=10, value=3, step=1)
        point_opacity = st.slider("Point Opacity", min_value=0.1, max_value=1.0, value=0.7, step=0.1,
                                   help="1.0 = fully opaque")
        
        # Visualization mode
        viz_mode = st.radio(
            "Visualization Mode",
            ["Standard", "Slice", "Deviation"],
            help="Standard: full LUT | Slice: specific plane | Deviation: difference from identity"
        )
        
        # Camera presets
        st.subheader("📷 View Presets")
        view_preset = st.radio(
            "Camera Angle",
            ["Free Rotation", "Front", "Side", "Top", "Diagonal"],
            help="Quick switch to different viewing angles"
        )
        
        show_cube_frame = st.checkbox("Show Cube Frame", value=True)
        show_identity_ref = st.checkbox("Show Identity Reference", value=False,
                                        help="Show ideal identity LUT for comparison")
        
        st.markdown("---")
        use_preview_downscale = st.checkbox("Downscale preview to 1280px (faster)", value=True)
        st.caption("💡 Upload an image to begin.")

    uploaded = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png", "tif", "tiff"]) 

    if not ckpt_path or not os.path.isfile(ckpt_path):
        st.info("📂 Select a valid checkpoint (e.g., saved_model/best.ckpt) to proceed.")
        return

    if uploaded is None:
        st.info("📷 Upload an input image to visualize the LUT and preview the enhancement.")
        return

    # Prepare image tensor
    try:
        pil_img = Image.open(uploaded).convert("RGB")
    except Exception as e:
        st.error(f"❌ Failed to read image: {e}")
        return

    x = pil_to_tensor(pil_img)  # [1,3,H,W] in [0,1]

    # Device and model
    device = _pick_device(device_choice)
    with st.spinner("⏳ Loading model..."):
        try:
            model, cfg = load_model_from_ckpt(ckpt_path, device)
            st.sidebar.success("✅ Model loaded successfully")
        except Exception as e:
            st.error(f"❌ Failed to load checkpoint: {e}")
            return

    x = x.to(device)

    # Compute LUT
    with st.spinner("🧮 Predicting image-adaptive LUT..."):
        try:
            Lstar, alpha, delta, fused, u, v, w, c = compute_image_adaptive_lut(model, x)
            # Create identity LUT for reference
            G = Lstar.shape[1]
            identity_lut_tensor = create_identity_lut_for_viz(G)
            identity_lut_np = identity_lut_tensor.squeeze(0).cpu().numpy()
        except Exception as e:
            st.error(f"❌ Failed to compute LUT: {e}")
            return

    # Preview (apply LUT)
    with st.spinner("✨ Applying LUT to image..."):
        x_for_preview = x
        if use_preview_downscale:
            H, W = x.shape[-2:]
            max_side = max(H, W)
            if max_side > 1280:
                scale = 1280 / max_side
                new_h = max(1, int(round(H * scale)))
                new_w = max(1, int(round(W * scale)))
                x_for_preview = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        try:
            pred = apply_lut_to_image(model, x_for_preview, Lstar)
        except Exception as e:
            st.error(f"❌ Failed to apply LUT: {e}")
            return

    # Overview - Always visible at top
    st.markdown("---")
    st.header("🖼️ Overview")
    
    # Quick stats
    alpha_np = alpha.squeeze(0).detach().cpu().numpy()
    delta_np = delta.squeeze(0).detach().cpu().numpy()
    delta_mag = np.linalg.norm(delta_np, axis=-1)
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.metric("Bases", f"{alpha.shape[1]}", help="Number of base LUTs")
    with col_stat2:
        st.metric("Alpha Sum", f"{alpha.sum().item():.3f}", help="Total alpha weights")
    with col_stat3:
        st.metric("Residual Magnitude", f"{delta_mag.mean():.4f}", help="Average residual correction")
    with col_stat4:
        st.metric("LUT Size", f"{Lstar.shape[1]}×{Lstar.shape[1]}×{Lstar.shape[1]}", help="Grid size")
    
    st.markdown("---")
    st.markdown("💡 **Tip**: Compare input and output side-by-side. Drag to rotate 3D plots, scroll to zoom.")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Input Image")
        st.image(tensor_to_image_np(x_for_preview), channels="RGB", use_container_width=True)
    with col2:
        st.subheader("✨ Enhanced Output")
        st.image(tensor_to_image_np(pred), channels="RGB", use_container_width=True)
    
    # Show identity reference if requested
    if show_identity_ref:
        st.markdown("---")
        st.subheader("📐 Ideal Identity LUT Reference")
        st.info("💡 An identity LUT doesn't change colors at all (Input = Output). All points lie on the diagonal line.")
        
        # Get slice settings based on viz mode
        slice_axis_ref = None
        slice_idx_ref = 0
        if viz_mode == "Slice":
            slice_axis_ref = st.selectbox("Slice Axis (Identity)", options=["R", "G", "B"], index=0, key="id_axis")
            slice_idx_ref = st.slider("Slice Position (Identity)", min_value=0, max_value=int(G - 1), 
                                      value=int(G // 2), step=1, key="id_idx")
        
        with st.spinner("🔄 Rendering identity LUT..."):
            fig_identity = make_lut_scatter3d(
                identity_lut_np, stride=viz_stride, slice_axis=slice_axis_ref, slice_index=int(slice_idx_ref),
                point_size=point_size, point_opacity=point_opacity, show_frame=show_cube_frame,
                camera_preset=view_preset, title="Ideal Identity LUT | Input = Output",
                viz_mode=viz_mode, identity_lut=identity_lut_np
            )
        st.plotly_chart(fig_identity, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})

    # Tabs for organized visualization
    st.markdown("---")
    st.header("📊 Analysis & Controls")
    tabs = st.tabs(["🎨 Bases & Alpha", "📐 Residual", "🔍 Factors", "🧊 3D Compare", "🔬 Ablation Study", "💾 Export"])

    # Bases & Alpha
    with tabs[0]:
        st.markdown("🎨 **Base LUTs** are learned color transformations. **Alpha weights** control their contribution.")
        st.subheader("📊 Base Weights (Alpha)")
        
        alpha_np = alpha.squeeze(0).detach().cpu().numpy()
        st.bar_chart(alpha_np)
        st.caption(f"Sum(alpha) ≈ {alpha.sum().item():.3f} | K={alpha.shape[1]} bases")

        st.markdown("---")
        st.subheader("🔍 Inspect Individual Base LUT")
        K = alpha.shape[1]
        col_base_sel, col_base_weight = st.columns([3, 1])
        with col_base_sel:
            if int(K - 1) >= 1:
                sel_base = st.slider("Base index", min_value=0, max_value=int(K - 1), value=0, step=1)
            else:
                sel_base = 0
                st.caption("Only one base available")
        with col_base_weight:
            st.metric("Alpha[k]", f"{alpha_np[sel_base]:.4f}")
        
        with torch.no_grad():
            base_np = model.bases[sel_base].detach().clamp(0, 1).cpu().numpy()
        
        # Slice controls based on viz mode
        slice_axis_b = None
        slice_idx_b = 0
        if viz_mode == "Slice":
            slice_axis_b = st.selectbox("Slice Axis", options=["R", "G", "B"], index=0, key="base_axis")
            G_ = base_np.shape[0]
            slice_idx_b = st.slider("Slice Position", min_value=0, max_value=int(G_ - 1), 
                                    value=int(G_ // 2), step=1, key="base_idx")
        
        with st.spinner("🔄 Rendering base LUT 3D visualization..."):
            fig_base = make_lut_scatter3d(
                base_np, stride=viz_stride, slice_axis=slice_axis_b, slice_index=int(slice_idx_b),
                point_size=point_size, point_opacity=point_opacity, show_frame=show_cube_frame,
                camera_preset=view_preset, title=f"Base LUT #{sel_base} | Weight={alpha_np[sel_base]:.4f}",
                viz_mode=viz_mode, identity_lut=identity_lut_np
            )
        st.plotly_chart(fig_base, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})

    # Residual
    with tabs[1]:
        st.markdown("📐 **Residual** is the low-rank correction added to the fused base LUTs for image-specific fine-tuning.")
        st.subheader("📊 Residual Statistics")
        delta_np = delta.squeeze(0).detach().cpu().numpy()
        mag = np.linalg.norm(delta_np, axis=-1)
        
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.metric("Mean Abs", f"{np.mean(np.abs(delta_np)):.5f}")
        with col_s2:
            st.metric("Mag Mean", f"{mag.mean():.5f}")
        with col_s3:
            st.metric("Mag Std", f"{mag.std():.5f}")
        with col_s4:
            st.metric("Mag Max", f"{mag.max():.5f}")
        
        st.markdown("---")
        st.subheader("🔥 Residual Heatmap (2D Slice)")
        show_heatmap = st.checkbox("Show slice heatmap", value=True, key="res_slice")
        
        if show_heatmap:
            axis_r = st.selectbox("Slice Axis", options=["R", "G", "B"], index=0, key="res_axis")
            idx_r = st.slider("Slice Position", min_value=0, max_value=int(mag.shape[0] - 1), 
                            value=int(mag.shape[0] // 2), step=1, key="res_idx")
            with st.spinner("🔄 Rendering residual heatmap..."):
                fig_hm = make_residual_plane_heatmap(delta_np, axis=axis_r, idx=int(idx_r))
            st.plotly_chart(fig_hm, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})

        st.markdown("---")
        st.subheader("🧊 Residual 3D Scatter (Magnitude)")
        st.info("💡 Brighter points = larger residual correction. Rotate to explore spatial distribution.")
        
        slice_axis3 = None
        slice_idx3 = 0
        if viz_mode == "Slice":
            slice_axis3 = st.selectbox("Slice Axis (3D)", options=["R", "G", "B"], index=0, key="res_axis3")
            slice_idx3 = st.slider("Slice Position (3D)", min_value=0, max_value=int(mag.shape[0] - 1), 
                                   value=int(mag.shape[0] // 2), step=1, key="res_idx3")
        
        with st.spinner("🔄 Rendering residual 3D scatter..."):
            fig_res3 = make_residual_scatter3d(
                delta_np, stride=viz_stride, slice_axis=slice_axis3, slice_index=int(slice_idx3),
                point_size=point_size, point_opacity=point_opacity, show_frame=show_cube_frame,
                camera_preset=view_preset
            )
        st.plotly_chart(fig_res3, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})

        

    # Factors
    with tabs[2]:
        st.markdown("🔍 **Low-rank factors** (u, v, w, c) define the CP decomposition for the residual LUT.")
        st.subheader("📐 CP Decomposition Factors")
        st.info("💡 Residual = Σ (u[r] ⊗ v[r] ⊗ w[r]) · c[r]  |  Select rank to inspect individual components")
        
        Rnk = u.shape[1]
        col_rank, col_rank_info = st.columns([3, 1])
        with col_rank:
            if int(Rnk - 1) >= 1:
                r_sel = st.slider("Rank index (r)", min_value=0, max_value=int(Rnk - 1), value=0, step=1)
            else:
                r_sel = 0
                st.caption("Only one rank available")
        with col_rank_info:
            st.metric("Total Rank", f"{Rnk}")
        
        u_np = u.squeeze(0).detach().cpu().numpy()  # [R,G]
        v_np = v.squeeze(0).detach().cpu().numpy()
        w_np = w.squeeze(0).detach().cpu().numpy()
        c_np = c.squeeze(0).detach().cpu().numpy()  # [R,3]
        
        st.markdown("---")
        st.subheader(f"📊 1D Factor Curves (Rank {r_sel})")
        
        line_u = go.Figure(data=[go.Scatter(y=u_np[r_sel], mode="lines+markers", 
                                            line=dict(color='rgb(255,100,100)', width=2),
                                            marker=dict(size=4))])
        line_u.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0), title="u[r] - Red axis factor",
                            yaxis_title="Value", xaxis_title="Position")
        
        line_v = go.Figure(data=[go.Scatter(y=v_np[r_sel], mode="lines+markers",
                                            line=dict(color='rgb(100,255,100)', width=2),
                                            marker=dict(size=4))])
        line_v.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0), title="v[r] - Green axis factor",
                            yaxis_title="Value", xaxis_title="Position")
        
        line_w = go.Figure(data=[go.Scatter(y=w_np[r_sel], mode="lines+markers",
                                            line=dict(color='rgb(100,100,255)', width=2),
                                            marker=dict(size=4))])
        line_w.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0), title="w[r] - Blue axis factor",
                            yaxis_title="Value", xaxis_title="Position")
        
        colu, colv, colw = st.columns(3)
        colu.plotly_chart(line_u, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})
        colv.plotly_chart(line_v, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})
        colw.plotly_chart(line_w, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})
        
        st.markdown("---")
        st.subheader(f"🎨 RGB Coefficients c[r={r_sel}]")
        
        colors_bar = ['rgb(255,100,100)', 'rgb(100,255,100)', 'rgb(100,100,255)']
        bar_c = go.Figure(data=[go.Bar(x=["R", "G", "B"], y=c_np[r_sel].tolist(),
                                       marker=dict(color=colors_bar))])
        bar_c.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), 
                           title=f"c[{r_sel}] RGB channel coefficients",
                           yaxis_title="Coefficient Value", xaxis_title="Channel")
        st.plotly_chart(bar_c, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})

    # 3D Compare
    with tabs[3]:
        st.markdown("🧊 **Compare** the fused base LUT (left) with the final LUT after residual correction (right).")
        st.subheader("📊 Fused vs Final LUT")
        st.info("💡 Fused = weighted sum of base LUTs | Final = Fused + Low-rank Residual")
        
        G = Lstar.shape[1]
        
        # Slice controls based on viz mode
        slice_axis_cmp = None
        slice_index_cmp = 0
        if viz_mode == "Slice":
            slice_axis_cmp = st.selectbox("Slice Axis", options=["R", "G", "B"], index=0, key="cmp_axis")
            slice_index_cmp = st.slider("Slice Position", min_value=0, max_value=int(G - 1), 
                                       value=int(G // 2), step=1, key="cmp_idx")
        
        fused_np = fused.squeeze(0).detach().clamp(0, 1).cpu().numpy()
        lut_np = Lstar.squeeze(0).detach().clamp(0, 1).cpu().numpy()
        
        colf, coll = st.columns(2)
        
        with colf:
            st.markdown("**🎨 Fused LUT (Base Combination)**")
            with st.spinner("🔄 Rendering fused LUT..."):
                fig_fused = make_lut_scatter3d(
                    fused_np, stride=viz_stride, slice_axis=slice_axis_cmp, slice_index=int(slice_index_cmp),
                    point_size=point_size, point_opacity=point_opacity, show_frame=show_cube_frame,
                    camera_preset=view_preset, title="Fused LUT (Bases Only)",
                    viz_mode=viz_mode, identity_lut=identity_lut_np
                )
            st.plotly_chart(fig_fused, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})
        
        with coll:
            st.markdown("**✨ Final LUT (Fused + Residual)**")
            with st.spinner("🔄 Rendering final LUT..."):
                fig_lut = make_lut_scatter3d(
                    lut_np, stride=viz_stride, slice_axis=slice_axis_cmp, slice_index=int(slice_index_cmp),
                    point_size=point_size, point_opacity=point_opacity, show_frame=show_cube_frame,
                    camera_preset=view_preset, title="Final LUT (Complete)",
                    viz_mode=viz_mode, identity_lut=identity_lut_np
                )
            st.plotly_chart(fig_lut, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})
        
        # Show deviation statistics
        st.markdown("---")
        with st.expander("📊 Deviation Statistics"):
            diff = lut_np - fused_np
            diff_mag = np.linalg.norm(diff, axis=-1)
            
            col_d1, col_d2, col_d3, col_d4 = st.columns(4)
            with col_d1:
                st.metric("Residual Mean Abs", f"{np.mean(np.abs(diff)):.5f}")
            with col_d2:
                st.metric("Residual Mag Mean", f"{diff_mag.mean():.5f}")
            with col_d3:
                st.metric("Residual Mag Std", f"{diff_mag.std():.5f}")
            with col_d4:
                st.metric("Residual Mag Max", f"{diff_mag.max():.5f}")

    # Ablation Study Tab
    with tabs[4]:
        st.markdown("🔬 **Ablation Study Controls** - Manipulate model parameters and see real-time effects.")
        st.info("💡 Adjust parameters below and see how they affect the image enhancement.")
        
        st.subheader("🎛️ Global Controls")
        
        col_enable1, col_enable2 = st.columns(2)
        with col_enable1:
            enable_residual = st.checkbox("Enable Residual Correction", value=True, 
                                          help="Toggle residual correction on/off")
            delta_amplifier = st.slider("Delta Amplifier", min_value=0.0, max_value=100.0, 
                                       value=1.0, step=1.0,
                                       help="Amplification factor for residual delta (default: 1.0)")
        with col_enable2:
            residual_scale = st.slider("Residual Scale", min_value=0.0, max_value=2.0,
                                      value=1.0, step=0.1,
                                      help="Global scale for residual contribution")
        
        st.markdown("---")
        st.subheader("🎨 Alpha Weights (Base Contribution)")
        
        with st.expander("📊 Manual Alpha Weights", expanded=False):
            K = alpha.shape[1]
            # Get predicted weights for comparison
            x_lr = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
            alpha_pred = model.weight_pred(x_lr)  # [1,K]
            alpha_pred_np = alpha_pred.squeeze(0).detach().cpu().numpy()
            alpha_scales = []
            
            use_manual_alpha = st.checkbox("Use Manual Alpha Weights", value=False,
                                           help="Override predicted alpha weights with manual values")
            
            if use_manual_alpha:
                st.caption(f"Adjust contribution of each of {K} base LUTs")
                cols = st.columns(min(K, 4))
                for i in range(K):
                    with cols[i % 4]:
                        scale = st.slider(f"Base {i}", min_value=-2.0, max_value=2.0,
                                         value=float(alpha_pred_np[i]), step=0.1, key=f"alpha_{i}")
                        alpha_scales.append(scale)
                alpha_scale_tensor = torch.tensor(alpha_scales, dtype=torch.float32).to(device)
            else:
                alpha_scale_tensor = None
            
            # Base enable/disable controls
            st.markdown("---")
            st.caption("Enable/Disable Individual Bases")
            enable_bases_list = []
            for i in range(K):
                enabled = st.checkbox(f"Base {i}", value=True, key=f"enable_base_{i}")
                enable_bases_list.append(1.0 if enabled else 0.0)
            enable_bases_tensor = torch.tensor(enable_bases_list, dtype=torch.float32).to(device)
        
        st.markdown("---")
        st.subheader("🔬 CP Rank Controls")
        
        with st.expander("🎚️ Rank Contribution Scales", expanded=False):
            R = u.shape[1]
            
            use_rank_scales = st.checkbox("Use Manual Rank Scales", value=False,
                                          help="Scale individual CP rank contributions")
            
            rank_scales = None
            if use_rank_scales:
                st.caption(f"Adjust contribution of each of {R} CP ranks")
                cols = st.columns(min(R, 4))
                rank_scale_list = []
                for i in range(R):
                    with cols[i % 4]:
                        scale = st.slider(f"Rank {i}", min_value=0.0, max_value=2.0,
                                         value=1.0, step=0.1, key=f"rank_{i}")
                        rank_scale_list.append(scale)
                rank_scales = torch.tensor(rank_scale_list, dtype=torch.float32).to(device)
        
        # Recompute LUT with ablation parameters
        with st.spinner("🔄 Computing LUT with ablation parameters..."):
            Lstar_ablation, alpha_ablation, delta_ablation, fused_ablation = compute_image_adaptive_lut_with_ablation(
                model, x,
                alpha_scale=alpha_scale_tensor if use_manual_alpha else None,
                residual_scale=residual_scale,
                delta_amplifier=delta_amplifier,
                rank_scales=rank_scales if use_rank_scales else None,
                enable_bases=enable_bases_tensor,
                enable_residual=enable_residual,
            )
        
        # Apply and display
        with st.spinner("✨ Applying modified LUT..."):
            pred_ablation = apply_lut_to_image(model, x_for_preview, Lstar_ablation)
        
        st.subheader("📊 Results Comparison")
        
        col_orig, col_modified = st.columns(2)
        with col_orig:
            st.markdown("**🔵 Original (Predicted)**")
            st.image(tensor_to_image_np(pred), channels="RGB", use_container_width=True)
        
        with col_modified:
            st.markdown("**🔬 Modified (Ablation)**")
            st.image(tensor_to_image_np(pred_ablation), channels="RGB", use_container_width=True)
        
        # Show alpha comparison
        st.markdown("---")
        st.subheader("📊 Alpha Weights Comparison")
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            alpha_orig_np = alpha.squeeze(0).detach().cpu().numpy()
            st.bar_chart(alpha_orig_np)
            st.caption("Original Alpha Weights")
        with col_a2:
            alpha_abl_np = alpha_ablation.squeeze(0).detach().cpu().numpy()
            st.bar_chart(alpha_abl_np)
            st.caption("Ablation Alpha Weights")
        
        # Visualize modified LUT
        st.markdown("---")
        st.subheader("🧊 Modified LUT 3D Visualization")
        
        lut_ablation_np = Lstar_ablation.squeeze(0).detach().clamp(0, 1).cpu().numpy()
        
        # Slice controls
        slice_axis_abl = None
        slice_idx_abl = 0
        if viz_mode == "Slice":
            slice_axis_abl = st.selectbox("Slice Axis", options=["R", "G", "B"], 
                                         index=0, key="abl_axis")
            slice_idx_abl = st.slider("Slice Position", min_value=0, max_value=int(G - 1), 
                                     value=int(G // 2), step=1, key="abl_idx")
        
        with st.spinner("🔄 Rendering modified LUT 3D..."):
            fig_abl = make_lut_scatter3d(
                lut_ablation_np, stride=viz_stride, 
                slice_axis=slice_axis_abl, slice_index=int(slice_idx_abl),
                point_size=point_size, point_opacity=point_opacity, 
                show_frame=show_cube_frame,
                camera_preset=view_preset, 
                title="Modified LUT (Ablation)",
                viz_mode=viz_mode, 
                identity_lut=identity_lut_np
            )
        st.plotly_chart(fig_abl, use_container_width=True, 
                       config={'displayModeBar': True, 'displaylogo': False})
        
        # Statistics
        with st.expander("📊 Ablation Statistics"):
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                delta_mag_abl = torch.norm(delta_ablation, dim=-1).mean().item()
                st.metric("Delta Magnitude", f"{delta_mag_abl:.5f}")
            with col_s2:
                fused_mag_abl = torch.norm(fused_ablation, dim=-1).mean().item()
                st.metric("Fused Magnitude", f"{fused_mag_abl:.5f}")
            with col_s3:
                alpha_sum_abl = alpha_ablation.sum().item()
                st.metric("Alpha Sum", f"{alpha_sum_abl:.3f}")

    # Export
    with tabs[5]:
        st.markdown("💾 **Export** your image-adaptive LUT for use in other applications.")
        st.subheader("📦 Export Options")
        
        G = Lstar.shape[1]
        lut_np = Lstar.squeeze(0).detach().clamp(0, 1).cpu().numpy()
        
        st.info(f"🔢 LUT size: {G}×{G}×{G} = {G**3:,} color mappings")
        
        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            st.markdown("#### 📄 .cube Format")
            st.caption("Compatible with Adobe Premiere, DaVinci Resolve, Final Cut Pro, Photoshop (via plugins)")
            cube_bytes = cube_bytes_from_lut(lut_np, title="LoR-IA3DLUT")
            st.download_button(
                label="⬇️ Download .cube",
                data=cube_bytes,
                file_name=f"lor_ia3dlut_{G}.cube",
                mime="text/plain",
                use_container_width=True
            )
        
        with col_export2:
            st.markdown("#### 🐍 .npy Format")
            st.caption("NumPy array format for Python/PyTorch workflows")
            buf = io.BytesIO()
            np.save(buf, lut_np)
            buf.seek(0)
            st.download_button(
                label="⬇️ Download .npy",
                data=buf.getvalue(),
                file_name=f"lor_ia3dlut_{G}.npy",
                mime="application/octet-stream",
                use_container_width=True
            )
        
        st.markdown("---")
        st.subheader("🖼️ Export Enhanced Image")
        
        # Export full resolution output
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            if st.button("🎨 Generate Full Resolution Output", use_container_width=True):
                with st.spinner("🔄 Applying LUT to full resolution image..."):
                    pred_full = apply_lut_to_image(model, x, Lstar)
                    pred_full_np = tensor_to_image_np(pred_full)
                    pred_full_img = Image.fromarray((pred_full_np * 255).astype(np.uint8))
                    
                    # Save to buffer
                    img_buf = io.BytesIO()
                    pred_full_img.save(img_buf, format="PNG", quality=95)
                    img_buf.seek(0)
                    
                    st.session_state['full_res_output'] = img_buf.getvalue()
                    st.success("✅ Full resolution output generated!")
        
        with col_img2:
            if 'full_res_output' in st.session_state:
                st.download_button(
                    label="⬇️ Download Enhanced Image",
                    data=st.session_state['full_res_output'],
                    file_name="enhanced_output.png",
                    mime="image/png",
                    use_container_width=True
                )
    
    # User Guide (at the bottom)
    st.markdown("---")
    with st.expander("💡 User Guide & Tips", expanded=False):
        st.markdown("""
        ### 🎨 Visualization Modes
        
        - **Standard Mode**: Show complete LUT cube. Yellow dashed line indicates identity (Input=Output).
        - **Slice Mode**: Show only a specific plane (easier to see internal structure).
        - **Deviation Mode**: Highlight difference from identity LUT (Red=large change, Gray=small change).
        
        ### 🖱️ 3D Interaction Controls
        
        - **Drag** to rotate the view
        - **Scroll** to zoom in/out
        - **Double-click** to reset view
        - **Shift+Drag** to pan
        - **📸 Camera icon** (top-right) to export high-resolution screenshot
        
        ### 📐 Understanding the Visualizations
        
        #### Overview Tab
        - Compare input and enhanced output images
        - Optional: view ideal identity LUT reference
        
        #### Bases & Alpha Tab
        - **Alpha weights**: Contribution of each base LUT to the final result
        - **Base LUTs**: Learned color transformations (K bases total)
        - Higher alpha = stronger influence on final output
        
        #### Residual Tab
        - **Residual**: Low-rank correction added for image-specific adaptation
        - **Heatmap**: 2D slice showing residual magnitude distribution
        - **3D Scatter**: Full 3D view of residual magnitude (brighter = larger correction)
        
        #### Factors Tab
        - **u, v, w**: 1D factor curves for CP decomposition along R, G, B axes
        - **c**: RGB coefficients for each rank
        - Residual = Σ (u[r] ⊗ v[r] ⊗ w[r]) · c[r]
        
        #### 3D Compare Tab
        - **Fused LUT**: Weighted combination of base LUTs
        - **Final LUT**: Fused + Residual correction
        - Compare side-by-side to see residual impact
        
        #### Export Tab
        - **.cube**: Standard format for video/photo editing software
        - **.npy**: Python/NumPy format for further processing
        - Full resolution image export available
        
        ### 🔍 Tips for Better Visualization
        
        1. **Lower opacity** (0.3-0.5) to see through outer points
        2. **Use Slice mode** to view specific planes
        3. **Rotate to different angles** using camera presets
        4. **Adjust stride** (1-4) to balance detail vs. performance
        5. **Enable cube frame** to better understand spatial structure
        
        ### 🎯 Color Interpretation
        
        - **Point position** = Input RGB color
        - **Point color** = Output RGB color
        - Points on the diagonal = no color change
        - Points far from diagonal = significant color transformation
        
        ### ⚡ Performance Tips
        
        - Use stride=2-3 for faster rendering
        - Enable "Downscale preview" for large images
        - Lower point count in Slice mode for complex visualizations
        """)
    
    st.markdown("---")
    st.caption("🎨 LoR-IA3DLUT Viewer | Low-Rank Image-Adaptive 3D LUT Visualization Tool")


if __name__ == "__main__":
    main()
