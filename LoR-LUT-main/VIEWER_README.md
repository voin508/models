### 🎨 LoR-IA3DLUT Viewer

An interactive Streamlit app for visualizing and exporting image-adaptive 3D LUTs with advanced low-rank decomposition analysis.

#### ✨ Features

**Enhanced Visualization:**
- 🎨 Multiple visualization modes: Standard, Slice, Deviation
- 📷 Camera angle presets: Free Rotation, Front, Side, Top, Diagonal
- 📐 Optional identity LUT reference for comparison
- 🧊 Customizable cube frame and point rendering
- 🎯 Interactive 3D plots with hover information

**Comprehensive Analysis Tabs:**
- **🖼️ Overview**: Side-by-side input/output comparison with optional identity reference
- **🎨 Bases & Alpha**: Visualize base LUT weights and inspect individual base LUTs
- **📐 Residual**: Statistical analysis, 2D heatmaps, and 3D magnitude visualization
- **🔍 Factors**: CP decomposition factors (u, v, w, c) with interactive rank selection
- **🧊 3D Compare**: Side-by-side comparison of fused vs final LUT
- **💾 Export**: Download `.cube`, `.npy`, and full-resolution enhanced images

**Flexible Controls:**
- Point size and opacity adjustment
- 3D visualization stride (performance vs. detail)
- Slice mode with axis and position selection
- Deviation mode for identity comparison
- Multiple camera angle presets

#### 📦 Install
```bash
pip install -r requirements.txt
```

#### 🚀 Run
```bash
streamlit run viewer_app.py
```

#### 💡 Usage Tips

**Visualization Modes:**
- **Standard**: Full LUT cube with identity line reference
- **Slice**: View specific plane for detailed analysis
- **Deviation**: Highlight differences from identity (Red=large, Gray=small)

**3D Interaction:**
- Drag to rotate, scroll to zoom
- Double-click to reset view
- Shift+Drag to pan
- Use camera icon (top-right) for high-res screenshots

**Performance:**
- Adjust stride (1-4) to balance detail vs speed
- Enable "Downscale preview" for large images
- Lower opacity (0.3-0.5) to see through outer points
- Use Slice mode to reduce point count

**Export Options:**
- `.cube`: Compatible with Adobe Premiere, DaVinci Resolve, Final Cut Pro
- `.npy`: For Python/PyTorch workflows
- Full-resolution enhanced images (PNG)

#### 🔧 Configuration

**Sidebar Settings:**
- Checkpoint path (default: `saved_model/best.ckpt`)
- Device selection (auto/cuda/cpu)
- Point size and opacity
- Visualization mode
- Camera presets
- Cube frame toggle

#### 📊 Understanding the Model

**LoR-IA3DLUT Architecture:**
```
Image → [Weight Predictor] → α (alpha weights)
     → [Residual Predictor] → u, v, w, c (CP factors)

Final LUT = Σ(α[k] · Base[k]) + Σ(u[r] ⊗ v[r] ⊗ w[r]) · c[r]
            \_____________/     \___________________________/
              Fused Base              Low-rank Residual
```

**Key Components:**
- **Base LUTs**: K learnable 3D color transformations
- **Alpha weights**: Image-adaptive mixing coefficients
- **Residual**: Low-rank CP decomposition for fine-tuning
- **Factors (u,v,w,c)**: 1D vectors defining the residual structure

#### 🎯 Color Interpretation
- Point position = Input RGB color
- Point color = Output RGB color
- Diagonal line = No color change (identity)
- Distance from diagonal = Transformation magnitude

#### 📝 Notes
- Device auto-selects GPU if available (override in sidebar)
- Preview optionally downscales for speed (exports unaffected)
- All 3D plots support interactive rotation and zoom
- User guide available in expandable section at bottom


