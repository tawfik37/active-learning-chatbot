#!/usr/bin/env python3
"""
Upload Local Model to Modal Volume
Upload a trained model from your local machine to Modal's persistent storage
"""

import modal
import sys

app = modal.App("upload-model")
volume = modal.Volume.from_name("chatbot-models")

# Use the same image as main app for consistency
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("transformers", "sentencepiece", "torch")
)

@app.function(
    image=image,
    volumes={"/models": volume},
    timeout=3600,  # 1 hour for large model uploads
)
def upload_model_to_volume(model_files: dict[str, bytes], model_name: str):
    """
    Upload model files to Modal volume.
    
    Args:
        model_files: Dictionary mapping file paths to file contents
        model_name: Name of the model (e.g., "qwen-finetuned-v1")
    """
    import os
    import json
    
    print("\n" + "="*80)
    print(f"UPLOADING MODEL: {model_name}")
    print("="*80 + "\n")
    
    target_dir = f"/models/{model_name}"
    os.makedirs(target_dir, exist_ok=True)
    
    # Write all files
    total_size = 0
    for rel_path, content in model_files.items():
        file_path = os.path.join(target_dir, rel_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'wb') as f:
            f.write(content)
        
        file_size = len(content)
        total_size += file_size
        print(f"✓ Wrote {rel_path} ({file_size/1024**2:.1f} MB)")
    
    print(f"\nTotal uploaded: {total_size/1024**3:.2f} GB")
    
    # Extract version number from model name
    try:
        version = int(model_name.split('-v')[-1])
    except:
        version = 1
    
    # Update config file
    config_path = "/models/_latest_model_config.json"
    config = {
        "latest_model_path": target_dir,
        "latest_version": version
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ Updated config: {config_path}")
    print(f"  Model path: {target_dir}")
    print(f"  Version: v{version}")
    
    # Commit changes to volume
    print("\nCommitting changes to volume...")
    volume.commit()
    
    print("\nUpload complete!")
    return {
        "model_path": target_dir,
        "version": version,
        "total_size_gb": total_size/1024**3
    }


@app.local_entrypoint()
def main(local_path: str):
    """
    Main entry point for uploading a local model.
    
    Usage:
        modal run upload_model.py --local-path ./qwen-finetuned-v1
    """
    import os
    
    if not os.path.exists(local_path):
        print(f"Error: Model path not found: {local_path}")
        sys.exit(1)
    
    if not os.path.isdir(local_path):
        print(f"Error: Path is not a directory: {local_path}")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("PREPARING MODEL FOR UPLOAD")
    print("="*80)
    print(f"Source: {local_path}")
    
    # Read all files from local path
    model_files = {}
    total_local_size = 0
    
    for root, dirs, files in os.walk(local_path):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, local_path)
            
            with open(file_path, 'rb') as f:
                content = f.read()
            
            model_files[rel_path] = content
            total_local_size += len(content)
            print(f"{rel_path} ({len(content)/1024**2:.1f} MB)")
    
    print(f"\nTotal to upload: {total_local_size/1024**3:.2f} GB")
    print(f"Number of files: {len(model_files)}")
    
    # Confirm upload
    print("\nThis will upload the model to Modal's cloud storage.")
    print("   Press Ctrl+C to cancel, or Enter to continue...")
    try:
        input()
    except KeyboardInterrupt:
        print("\nUpload cancelled.")
        sys.exit(0)
    
    # Get model name
    model_name = os.path.basename(local_path.rstrip('/'))
    
    # Upload to Modal
    print(f"\n🚀 Uploading to Modal volume as '{model_name}'...")
    result = upload_model_to_volume.remote(model_files, model_name)
    
    print("\n" + "="*80)
    print("SUCCESS!")
    print("="*80)
    print(f"Model '{model_name}' is now available in Modal.")
    print(f"\nTo use it, your API will automatically load: {result['model_path']}")
    print(f"\nYou can verify with: modal run inspect_volume.py")