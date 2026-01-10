# PASTA Web Interface Guide

## 📋 Overview

PASTA Web Interface provides a user-friendly GUI for spatial phenotype prediction on pathology images. No coding or JSON configuration required - complete the entire prediction workflow through your browser.

## 🚀 Quick Start

### 1. Install

Please follow the introduction in [README.md](README.md).

### 2. Launch Web Interface

```bash
python web_ui.py
```

Launch parameters:
- `--host`: Host address (default: 0.0.0.0)
- `--port`: Port number (default: 7860)

Examples:

```bash
# Simply run 
python web_ui.py --port 7860

# Specify GPU device
CUDA_VISIBLE_DEVICES=0 python web_ui.py
```

### 3. Access Interface

Open browser and visit:
- Local access: `http://localhost:7860`
- LAN access: `http://<your-ip>:7860`

## 📖 Usage Instructions

### Interface Layout

The interface is divided into left and right panels:

**Left Panel - Input Configuration**
1. **File Upload**: Upload WSI image files
2. **Model Configuration**: Select backbone model and pathway configuration
3. **Prediction Parameters**: Set prediction mode and downsampling factor
4. **Advanced Options**: Adjust patch size, step size, and other advanced parameters

**Right Panel - Prediction Results**
1. **Status Display**: Real-time task progress
2. **Heatmaps**: View prediction heatmaps for each pathway
3. **Overlay Images**: View predictions overlaid on H&E images
4. **Statistics**: View prediction statistics
5. **Download**: Download complete results in h5ad format

### Operation Steps

#### Step 1: Upload WSI File

Supported formats:
- `.svs` - Aperio format
- `.tif` / `.tiff` - TIFF format
- `.mrxs` - MIRAX format
- `.ndpi` - Hamamatsu format

Click the "Upload WSI File" area and select your pathology image file.

#### Step 2: Configure Model

**Backbone Model**
- System automatically scans model weight files in `model/` directory
- Select corresponding backbone model (e.g., UNI, Virchow2, Gigapath)
- Then select specific weight file

**Pathway Configuration**
- `default_14`: 14 tumor microenvironment pathways
- `default_16`: 16 tumor microenvironment pathways
- `313_Xenium`: 313 gene pathways
- `100_rep_genes`: 100 representative genes
- Other (check the Hugging Face repo for all available models)

**Specify Pathways (Optional)**
- To predict specific pathways only, enter them here
- Comma-separated, e.g., `CAF, T-cells, B-cells`
- Leave empty to predict all pathways

#### Step 3: Set Prediction Parameters

**Prediction Mode**
- `pixel`: High-resolution pixel-level prediction (recommended for detailed analysis)
- `spot`: Fast spot-level prediction (recommended for batch processing)

**Downsampling Factor** (1-20)
- Higher values result in lower output resolution but faster processing
- Recommended values:
  - Detailed analysis: 4-6
  - Regular analysis: 10
  - Quick preview: 15-20

**Include TLS Prediction**
- Check this option to additionally predict Tertiary Lymphoid Structures (TLS)

#### Step 4: Advanced Options (Optional)

- **Patch Size**: Size of extracted image patches (default 256)
- **Step Size**: Step size between patches (default 128, smaller values produce denser predictions)
- **Image Size**: Output image dimensions (default 10)
- **Device**: GPU device (e.g., cuda:0, cuda:1) or cpu
- **Save TIFF**: Whether to save QuPath-compatible TIFF files
- **Colormap**: Color scheme for heatmaps

#### Step 5: Submit and Monitor

1. Click "🚀 Start Prediction" button
2. After task submission, system displays task ID
3. Progress bar updates in real-time (auto-refresh every 3 seconds)
4. Manually click "🔄 Refresh Status" button if needed

#### Step 6: View and Download Results

After prediction completes, view results in right-side tabs:

**Heatmaps Tab**
- Displays prediction heatmaps for each pathway
- Click images to enlarge

**Overlay Tab**
- Shows predictions overlaid on original H&E images
- More intuitive understanding of spatial distribution

**Statistics Tab**
- Displays statistics for each pathway (mean, std, min, max)
- Shows processing time and other information

**Download Tab**
- Download complete results in h5ad format
- Can be analyzed further with Scanpy, Squidpy, and other tools

## 🔧 Technical Details

### File Management

All task files are stored in `results/web_tasks/` directory:

```
results/web_tasks/
├── <timestamp>_<filename>/
│   ├── wsi/              # Original WSI files
│   ├── patches/          # Extracted patches
│   ├── masks/            # Tissue segmentation masks
│   ├── edge_info/        # Edge information
│   └── predictions/      # Prediction results
│       └── <sample_id>/
│           ├── plots/           # Heatmaps
│           ├── plots_overlay/   # Overlay images
│           └── *.h5ad          # AnnData files
```

### Automatic Cleanup

- System automatically cleans up tasks older than 24 hours on startup
- Manually delete folders in `results/web_tasks/` directory if needed

## 🐳 Docker Usage

### Build Image with Web Interface

To use Web Interface in Docker, expose the port in Dockerfile:

```dockerfile
# Add to existing Dockerfile
EXPOSE 7860

# Launch command
CMD ["python", "web_ui.py", "--host", "0.0.0.0", "--port", "7860"]
```

### Run Container

```bash
# With GPU support
docker run --gpus all -p 7860:7860 -it mengflz/pasta:latest python web_ui.py

# Using Podman
podman run --device nvidia.com/gpu=all -p 7860:7860 -it mengflz/pasta:latest python web_ui.py
```

Then visit `http://localhost:7860`

## 💡 Usage Tips

### 1. Model File Organization

For proper model file recognition, use filenames containing model names:

```
model/
├── UNI_pos_embed.pt
├── Virchow2_no_pos_embed.pt
├── gigapath_weights.pt
└── Phikonv2_no_pos_embed.pt
```

### 2. Large File Processing

For very large WSI files (>5GB):
- Use higher downsampling factors (8-16)
- Consider spot mode instead of pixel mode
- Ensure sufficient disk space (at least 3x the WSI file size)

### 3. Batch Processing

While the Web Interface is designed for single-file processing, for batch processing:
1. Open multiple browser tabs to submit tasks simultaneously
2. Or use command-line tools for batch processing (see README.md)

### 4. Remote Access

For remote access to Web Interface, use SSH port forwarding:

```bash
# Execute on local machine
ssh -L 7860:localhost:7860 user@remote-server
```
