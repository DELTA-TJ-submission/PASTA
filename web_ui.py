#!/usr/bin/env python3
"""
Gradio Web Interface for PASTA
"""

import os
import sys
import glob
import shutil
import time
import gradio as gr
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pasta.task_manager import get_task_manager
from pasta.web_utils import (
    scan_model_files, 
    get_pathway_choices, 
    get_pathway_names,
    create_temp_config,
    format_results_for_display,
    cleanup_old_tasks,
    get_result_statistics
)

# Initialize task manager
task_manager = get_task_manager(max_workers=2)

WEB_TASKS_DIR = "results/web_tasks"
os.makedirs(WEB_TASKS_DIR, exist_ok=True)

# Path to UI logo
ICON_PATH = Path(__file__).parent / ".github" / "PASTA_icon.png"


def run_prediction_pipeline(task_id: str, wsi_path: str, config: dict):
    """
    Run the full prediction pipeline with progress updates.
    
    Args:
        task_id: Task identifier
        wsi_path: Path to WSI file
        config: Configuration dictionary
        
    Returns:
        Dictionary of result paths
    """
    
    try:
        from pasta.utils import setup_huggingface, ensure_model_weights
        # Setup HuggingFace if needed
        if config['huggingface']['endpoint']:
            setup_huggingface(config['huggingface'])
        
        from pasta.create_patches_fp import seg_and_patch
        from pasta.process_patch_h5 import process_file
        from pasta.inference import run_inference_pipeline
        
        # Step 1: Extract patches 
        task_manager.update_progress(task_id, 0.05, "Starting patch extraction...")
        
        # Create output directories
        for subdir in ['patches', 'masks', 'stitches']:
            os.makedirs(os.path.join(config['patch_extraction']['save_dir'], subdir), exist_ok=True)
        
        seg_and_patch(**config['patch_extraction'])
        task_manager.update_progress(task_id, 0.30, "Patch extraction completed")
        
        # Step 2: Process H5 files (30-60%)
        task_manager.update_progress(task_id, 0.35, "Processing H5 files...")
        
        h5_files = glob.glob(f"{config['h5_processing']['h5_path']}/*.h5")
        if not h5_files:
            raise ValueError("No H5 files generated from patch extraction")
        
        for i, h5_file in enumerate(h5_files):
            process_file(
                h5_file,
                config['h5_processing']['patch_size'],
                config['h5_processing']['slide_path'],
                config['h5_processing']['file_type'],
                config['h5_processing']['mask_path'],
                config['h5_processing']['save_edge'],
                config['h5_processing']['edge_info_path'],
                config['h5_processing']['blank']
            )
            progress = 0.35 + (0.25 * (i + 1) / len(h5_files))
            task_manager.update_progress(task_id, progress, f"Processing H5 file {i+1}/{len(h5_files)}")
        
        task_manager.update_progress(task_id, 0.60, "H5 file processing completed")
        
        # Step 3: Run inference (60-100%)
        task_manager.update_progress(task_id, 0.65, "Loading model...")

        # Ensure HuggingFace is set up before loading model
        if config['huggingface']['endpoint']:
            os.environ["HF_ENDPOINT"] = config['huggingface']['endpoint']
        if config['huggingface']['token']:
            os.environ["HF_TOKEN"] = config['huggingface']['token']
            try:
                from huggingface_hub import login
                login(token=config['huggingface']['token'])
            except Exception as e:
                print(f"Warning: HF login failed: {e}")

        # Ensure model weights exist (auto-download when missing), mirroring demo.py behaviour
        model_path = config.get('inference', {}).get('model_path')
        if model_path:
            try:
                ensure_model_weights(model_path)
            except Exception as e:
                # Do not hard-fail here; let downstream code surface a clear error if needed
                print(f"Warning: automatic model download failed for '{model_path}': {e}")

        run_inference_pipeline(config['inference'])
        
        task_manager.update_progress(task_id, 0.95, "Generating visualizations...")
        
        # Collect results
        result_paths = {
            "prediction_dir": config['inference']['output_path'],
            "task_dir": config['patch_extraction']['save_dir']
        }
        
        task_manager.update_progress(task_id, 1.0, "Prediction completed!")
        
        return result_paths
        
    except Exception as e:
        task_manager.update_progress(task_id, -1, f"Error: {str(e)}")
        raise


def submit_prediction(
    wsi_file,
    backbone_model,
    model_file,
    pathway_config,
    selected_pathways_text,
    prediction_mode,
    downsample_size,
    include_tls,
    patch_size,
    step_size,
    figsize,
    device,
    save_tiffs,
    cmap,
    hf_token,
    hf_endpoint
):
    """Handle prediction submission from UI."""
    
    if wsi_file is None:
        return "❌ Please upload a WSI file", "", 0, "", None, None, None, None, None
    
    # Save uploaded file
    wsi_filename = os.path.basename(wsi_file.name)
    task_id = f"{int(time.time())}_{wsi_filename.split('.')[0]}"
    task_output_dir = os.path.join(WEB_TASKS_DIR, task_id)
    os.makedirs(task_output_dir, exist_ok=True)
    
    # Copy WSI file to task directory
    wsi_save_path = os.path.join(task_output_dir, "wsi", wsi_filename)
    os.makedirs(os.path.dirname(wsi_save_path), exist_ok=True)
    shutil.copy(wsi_file.name, wsi_save_path)
    
    # Parse selected pathways
    selected_pathways = None
    if selected_pathways_text.strip():
        selected_pathways = [p.strip() for p in selected_pathways_text.split(",")]
    
    # Determine file type
    file_type = os.path.splitext(wsi_filename)[1]
    
    # Create configuration
    config = create_temp_config(
        wsi_path=wsi_save_path,
        task_output_dir=task_output_dir,
        backbone_model=backbone_model,
        model_path=model_file,
        pathway_config=pathway_config,
        selected_pathways=selected_pathways,
        prediction_mode=prediction_mode,
        downsample_size=downsample_size,
        include_tls=include_tls,
        patch_size=patch_size,
        step_size=step_size,
        figsize=figsize,
        device=device,
        file_type=file_type,
        save_tiffs=save_tiffs,
        cmap=cmap,
        hf_token=hf_token,
        hf_endpoint=hf_endpoint
    )
    
    # Create task
    task_id_short = task_manager.create_task(wsi_filename, config)
    
    # Submit for execution
    task_manager.submit_task(
        task_id_short,
        run_prediction_pipeline,
        wsi_save_path,
        config
    )
    
    return (
        f"✅ Task submitted! Task ID: {task_id_short}",
        task_id_short,
        0.0,
        "Waiting for processing...",
        None,
        None,
        None,
        None,
        None  # stats_table
    )


def check_task_status(task_id):
    """Check status of a running task."""
    
    if not task_id:
        return 0, "Please submit a task", None, None, None, None, None
    
    task = task_manager.get_task(task_id)
    
    if task is None:
        return 0, "Task not exists", None, None, None, None, None
    
    progress = task.progress
    status_text = f"Status: {task.status} | {task.current_step}"
    
    if task.status == "failed":
        status_text = f"❌ Failed: {task.error_message}"
        return 0, status_text, None, None, None, None, None
    
    if task.status == "completed":
        # Format results
        heatmaps, overlays, h5ad_path, summary = format_results_for_display(task.result_paths)
        
        # Get statistics if h5ad exists
        stats_df = None
        if h5ad_path and os.path.exists(h5ad_path):
            try:
                stats_df = get_result_statistics(h5ad_path)
                # Convert DataFrame to list of lists for Gradio Dataframe
                # Since headers are already set in the UI, we only need values
                if stats_df is not None and not stats_df.empty:
                    stats_df = stats_df.values.tolist()
            except Exception as e:
                print(f"Error loading statistics: {e}")
                stats_df = None
        
        elapsed = task.end_time - task.start_time if task.end_time else 0
        status_text = f"✅ Completed! Time: {elapsed:.1f} seconds"
        
        return (
            1.0,
            status_text,
            heatmaps if heatmaps else None,
            overlays if overlays else None,
            h5ad_path,
            summary + f"\n\nTime: {elapsed:.1f} seconds",
            stats_df
        )
    
    return progress, status_text, None, None, None, None, None


def update_model_files(backbone):
    """Update available model files based on selected backbone."""
    model_mapping = scan_model_files('/workspace/model')
    
    if backbone in model_mapping:
        choices = model_mapping[backbone]
        return (
            gr.Dropdown(
                choices=choices, 
                value=choices[0] if choices else None,
                allow_custom_value=True,
                interactive=True
            ),
            f"✓ Found {len(choices)} {backbone} model files"
        )
    # Return empty dropdown but allow custom input
    return (
        gr.Dropdown(
            choices=[], 
            value=None,
            allow_custom_value=True,
            interactive=True
        ),
        f"⚠️ No {backbone} model files found, please manually input model path (e.g. model/{backbone}_pos_embed.pt)"
    )


def update_pathway_list(pathway_config):
    """Update pathway list display."""
    pathway_names = get_pathway_names("pasta/configs/pathways.json", pathway_config)
    if pathway_names:
        return f"Available pathways ({len(pathway_names)}):\n" + "\n".join([f"- {name}" for name in pathway_names[:10]]) + \
               (f"\n...and {len(pathway_names) - 10} more" if len(pathway_names) > 10 else "")
    return "Unable to load pathway list"


# Custom CSS for beautiful UI
custom_css = """
#header-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.8em;
    margin-top: 1em;
    margin-bottom: 1em;
    padding: 1em 0;
}

#main-logo {
    flex-shrink: 0;
}

#main-logo img {
    width: 120px;
    height: 120px;
    object-fit: contain;
    filter: drop-shadow(0 4px 6px rgba(0, 0, 0, 0.1));
    transition: transform 0.3s ease;
}

#title-group {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
}

#main-title {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5em;
    font-weight: bold;
    margin: 0;
    line-height: 1.2;
}

#subtitle {
    color: #666;
    font-size: 1.1em;
    margin-top: 0.3em;
    font-weight: 300;
    letter-spacing: 0.5px;
}

.gradio-container {
    max-width: 1400px !important;
}

#submit-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    font-size: 1.1em;
    font-weight: bold;
}

#status-box {
    border-left: 4px solid #667eea;
    padding-left: 1em;
}

.tab-nav button {
    font-size: 1.05em;
}
"""


def build_interface():
    """Build the Gradio interface."""
    
    # All supported backbone models (for dropdown)
    all_backbone_choices = [
        "UNI", "UNIv2", "CONCH", "Virchow", "Virchow2", 
        "gigapath", "Phikon", "Phikonv2", "Kaiko-B", "Kaiko-L",
        "H-optimus-0", "H-optimus-1", "Hibou-B", "Hibou-L", "PLIP"
    ]
    
    # Scan available models
    model_mapping = scan_model_files()
    found_backbones = list(model_mapping.keys())
    
    # Use all backbones as choices, but mark which ones have files
    backbone_choices = all_backbone_choices
    
    # Get pathway choices
    pathway_choices = get_pathway_choices()
    
    # Create model status message
    if not found_backbones:
        model_status_msg = "⚠️ No model files found. Please put model weight files (.pt or .pth) into model/ directory"
    else:
        model_status_msg = f"✓ Found {len(found_backbones)} model categories, total {sum(len(files) for files in model_mapping.values())} files"
    
    with gr.Blocks(title="PASTA Prediction System", css=custom_css, theme=gr.themes.Soft()) as demo:
        
        # Header with logo and title in one row
        with gr.Row(elem_id="header-container"):
            if ICON_PATH.exists():
                gr.Image(value=str(ICON_PATH), show_label=False, container=False, elem_id="main-logo", height=120, width=120)
            with gr.Column(elem_id="title-group", scale=1):
                gr.Markdown("# 🔬 PASTA Spatial Pathology Prediction System", elem_id="main-title")
                gr.Markdown("A plug-and-play model paradigm for gigapixel vritual tissue phenotyping", elem_id="subtitle")
        
        # Hidden state for task ID
        task_id_state = gr.State("")
        
        with gr.Row():
            # Left column - Input
            with gr.Column(scale=1):
                gr.Markdown("## 📤 Input Configuration")
                
                wsi_file = gr.File(
                    label="Upload WSI File",
                    file_types=[".svs", ".tif", ".tiff", ".mrxs", ".ndpi"],
                    type="filepath"
                )
                
                with gr.Accordion("🎯 Model Configuration", open=True):
                    # Model status message
                    model_status = gr.Markdown(model_status_msg)
                    
                    backbone_model = gr.Dropdown(
                        choices=backbone_choices,
                        label="Backbone Model",
                        value=backbone_choices[0] if backbone_choices else None,
                        info="Select the backbone model",
                        allow_custom_value=False
                    )
                    
                    # Model file input - allow both dropdown and text input
                    model_file_choices = model_mapping.get(backbone_choices[0], []) if backbone_choices and model_mapping.get(backbone_choices[0]) else []
                    
                    model_file = gr.Dropdown(
                        choices=model_file_choices,
                        label="Model Weight File",
                        value=model_file_choices[0] if model_file_choices else None,
                        info="Select or manually input model file path (e.g. model/UNI_pos_embed.pt)",
                        allow_custom_value=True,
                        interactive=True
                    )
                    
                    pathway_config = gr.Dropdown(
                        choices=[choice[1] for choice in pathway_choices],
                        label="Pathway Configuration",
                        value=pathway_choices[0][1] if pathway_choices else "default_14",
                        info="Select the pathways to predict"
                    )
                    
                    pathway_list_display = gr.Textbox(
                        label="Pathway List",
                        value=update_pathway_list(pathway_choices[0][1] if pathway_choices else "default_14"),
                        lines=6,
                        interactive=False
                    )
                    
                    selected_pathways = gr.Textbox(
                        label="Specified Pathways (Optional)",
                        placeholder="Leave empty to predict all, or input pathways separated by commas, e.g. CAF, T-cells, B-cells",
                        lines=2
                    )
                
                with gr.Accordion("⚙️ Prediction Parameters", open=True):
                    prediction_mode = gr.Radio(
                        choices=["pixel", "spot"],
                        label="Prediction Mode",
                        value="pixel",
                        info="pixel: High-resolution pixel-level prediction | spot: Fast spot-level prediction"
                    )
                    
                    downsample_size = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=10,
                        step=1,
                        label="Downsample Factor",
                        info="The larger the value, the lower the output resolution, the faster the processing speed"
                    )
                    
                    include_tls = gr.Checkbox(
                        label="Include TLS Prediction",
                        value=False,
                        info="Predict tertiary lymphoid structure (TLS) scores"
                    )
                
                with gr.Accordion("🔧 Advanced Options", open=False):
                    patch_size = gr.Slider(
                        minimum=128,
                        maximum=512,
                        value=256,
                        step=64,
                        label="Patch Size"
                    )
                    
                    step_size = gr.Slider(
                        minimum=64,
                        maximum=512,
                        value=128,
                        step=64,
                        label="Step Size"
                    )
                    
                    figsize = gr.Slider(
                        minimum=5,
                        maximum=20,
                        value=10,
                        step=1,
                        label="Image Size"
                    )
                    
                    device = gr.Textbox(
                        label="Device",
                        value="cuda:0",
                        info="e.g. cuda:0, cuda:1, cpu"
                    )
                    
                    save_tiffs = gr.Checkbox(
                        label="Save TIFF Files (For QuPath usage)",
                        value=False
                    )
                    
                    cmap = gr.Dropdown(
                        choices=["turbo", "viridis", "plasma", "inferno", "magma", "jet"],
                        label="Color Mapping",
                        value="turbo"
                    )
                    
                    gr.Markdown("### 🤗 HuggingFace Configuration (For model downloading)")
                    hf_token = gr.Textbox(
                        label="HF Token (Optional)",
                        placeholder="Leave empty to use environment variable HF_TOKEN, or input your HuggingFace token",
                        type="password",
                        info="Some models (e.g. UNI, Virchow) require HF token to download"
                    )
                    
                    hf_endpoint = gr.Dropdown(
                        choices=["https://hf-mirror.com", "https://huggingface.co"],
                        label="HF Endpoint",
                        value="https://hf-mirror.com",
                        info="Recommended to use mirror: https://hf-mirror.com (Faster access in China)"
                    )
                
                submit_btn = gr.Button("🚀 Start Prediction", variant="primary", size="lg", elem_id="submit-btn")
            
            # Right column - Results
            with gr.Column(scale=1):
                gr.Markdown("## 📊 Prediction Results")
                
                status_msg = gr.Textbox(
                    label="Submission Status",
                    value="Waiting for submission...",
                    interactive=False
                )
                
                with gr.Group(elem_id="status-box"):
                    progress_bar = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=0,
                        label="Progress",
                        interactive=False
                    )
                    
                    status_text = gr.Textbox(
                        label="Current Status",
                        value="",
                        interactive=False
                    )
                
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh Status", size="sm", scale=1)
                    auto_refresh = gr.Checkbox(label="Auto Refresh", value=False, scale=1)
                
                with gr.Tabs():
                    with gr.TabItem("🎨 Heatmaps"):
                        heatmap_gallery = gr.Gallery(
                            label="Prediction Heatmaps",
                            columns=2,
                            height="auto"
                        )
                    
                    with gr.TabItem("🖼️ Overlays"):
                        overlay_gallery = gr.Gallery(
                            label="H&E Overlays",
                            columns=2,
                            height="auto"
                        )
                    
                    with gr.TabItem("📈 Statistics"):
                        summary_text = gr.Markdown("Waiting for results...")
                        stats_table = gr.Dataframe(
                            label="Pathway Statistics",
                            headers=["Pathway", "Mean", "Std", "Min", "Max"],
                            interactive=False,
                            wrap=True
                        )
                    
                    with gr.TabItem("💾 Download"):
                        h5ad_file = gr.File(
                            label="Download H5AD File",
                            interactive=False
                        )
        
        # Event handlers
        backbone_model.change(
            fn=update_model_files,
            inputs=[backbone_model],
            outputs=[model_file, model_status]
        )
        
        pathway_config.change(
            fn=update_pathway_list,
            inputs=[pathway_config],
            outputs=[pathway_list_display]
        )
        
        submit_btn.click(
            fn=submit_prediction,
            inputs=[
                wsi_file, backbone_model, model_file, pathway_config,
                selected_pathways, prediction_mode, downsample_size,
                include_tls, patch_size, step_size, figsize, device,
                save_tiffs, cmap, hf_token, hf_endpoint
            ],
            outputs=[
                status_msg, task_id_state, progress_bar, status_text,
                heatmap_gallery, overlay_gallery, h5ad_file, summary_text, stats_table
            ]
        )
        
        # Manual refresh
        refresh_btn.click(
            fn=check_task_status,
            inputs=[task_id_state],
            outputs=[
                progress_bar, status_text, heatmap_gallery,
                overlay_gallery, h5ad_file, summary_text, stats_table
            ]
        )
        
        # Setup timer for auto-refresh when enabled
        timer = gr.Timer(3)
        
        def check_auto_refresh(auto_enabled, task_id):
            """Check if auto-refresh is enabled and return status."""
            if auto_enabled and task_id:
                return check_task_status(task_id)
            return 0, "", None, None, None, None, None
        
        timer.tick(
            fn=check_auto_refresh,
            inputs=[auto_refresh, task_id_state],
            outputs=[
                progress_bar, status_text, heatmap_gallery,
                overlay_gallery, h5ad_file, summary_text, stats_table
            ]
        )
        
        # Footer
        gr.Markdown("""
        ---
        ### 📖 Usage Instructions
        1. Upload WSI file (supports .svs, .tif, .tiff, .mrxs format)
        2. Select model and configuration parameters
        3. Click "Start Prediction" to submit task
        4. Click "Refresh Status" to view progress (or wait for auto refresh)
        5. View and download results in the result tab after completion
        
        ⚠️ **Note**: The prediction process may take several minutes depending on the image size and configuration parameters.
        """)
    
    return demo


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PASTA Web Interface")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    parser.add_argument("--share", action="store_true", help="Create public share link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Clean up old tasks on startup
    cleaned = cleanup_old_tasks(max_age_hours=24)
    print(f"Cleaned up {cleaned} old task directories")
    
    # Build and launch interface
    demo = build_interface()
    
    print(f"""
    ╔══════════════════════════════════════════════════╗
    ║     🔬 PASTA Web Interface Starting...          ║
    ╚══════════════════════════════════════════════════╝
    
    📍 Local URL:     http://{args.host}:{args.port}
    🌐 Network URL:   http://<your-ip>:{args.port}
    
    Press Ctrl+C to stop the server
    """)
    
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        show_error=True,
        css=custom_css,
        theme=gr.themes.Soft()
    )

