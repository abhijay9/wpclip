import torch
from torch import nn
import clip
from tqdm import tqdm
import csv
from pathlib import Path
from PIL import Image
import pandas as pd
import argparse
import sys

from configs import wpclip_v1 as cfg


def get_autocast():
    """Get autocast function compatible with different PyTorch versions."""
    try:
        return torch.cuda.amp.autocast
    except AttributeError:
        # Dummy autocast for PyTorch < 1.6
        class DummyAutocast:
            def __init__(self, enabled=True, dtype=None):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return DummyAutocast


def convert_models_to_fp32(model):
    """Convert model parameters and gradients to fp32."""
    for param in model.parameters():
        param.data = param.data.float()
        if param.grad is not None:
            param.grad.data = param.grad.data.float()


def calculate_probabilities(logits_per_image, antonym_keys):
    """
    Calculate probabilities for each antonym pair.
    
    Args:
        logits_per_image: Image logits from CLIP model
        antonym_keys: List of antonym pair keys
        
    Returns:
        dict: Probabilities for each antonym pair
    """
    probs = {}
    for i, key in enumerate(antonym_keys):
        # Extract logits for current antonym pair and apply softmax
        logits = logits_per_image[:, 2*i:2*i+2].softmax(dim=-1)
        # Calculate probability ratio with numerical stability
        probs[key] = logits[:, 0] / (logits[:, 0] + logits[:, 1] + 1e-7)
    
    return probs


def get_antonym_pairs():
    """
    Get antonym pairs from the dataset CSV to maintain consistency.
    
    Returns:
        tuple: (antonyms, antonym_keys, txt_prompts)
    """
    # Load CSV to get principle names
    try:
        art_df = pd.read_csv("./data/art_classes_real.csv")
        principles = list(art_df.columns[1:-1])  # Exclude 'img_id' and 'set' columns
    except FileNotFoundError:
        # Fallback to common art principles if CSV not found
        print("Warning: CSV file not found, using default art principles")
        principles = [
            "linear-vs-painterly",
            "closed-vs-open", 
            "multiplicity-vs-unity",
            "clearness-vs-unclearness"
        ]
    
    # Extract principle antonym pairs
    antonyms = [
        [p.split("_")[0] for p in principle.split("-vs-")] 
        for principle in principles
    ]
    
    # Create keys for antonym pairs
    antonym_keys = [f"{p[0]}-{p[1]}" for p in antonyms]
    
    # Create text prompts for CLIP
    txt_prompts = [term for antonym_pair in antonyms for term in antonym_pair]
    
    return antonyms, antonym_keys, txt_prompts


def evaluate_single_image(model_path, image_path, output_dir="samples"):
    """
    Evaluate a single image and return/export results.
    
    Args:
        model_path (str): Path to trained model weights
        image_path (str): Path to the image to evaluate
        output_dir (str): Directory to save output CSV file
        
    Returns:
        dict: Predicted probabilities for each antonym pair
    """
    # Setup device and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
    
    model, preprocess = clip.load(cfg.model["vision_backbone"], device=device, jit=False)
    
    # Load trained weights with proper device mapping
    print(f"Loading model from: {model_path}")
    if device == "cuda":
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.cuda()
    else:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    
    # Get antonym pairs and text prompts
    antonyms, antonym_keys, txt_prompts = get_antonym_pairs()
    
    # Setup for evaluation
    if device == "cpu":
        model.float()
    else:
        model.half()  # Use fp16 for faster GPU inference
        torch.backends.cudnn.benchmark = True
    
    autocast = get_autocast()
    text_tokens = clip.tokenize(txt_prompts).to(device)
    
    # Pre-encode text tokens for efficiency
    print("Pre-encoding text tokens...")
    with torch.no_grad():
        if device == "cuda":
            with autocast(enabled=True, dtype=torch.float16):
                text_features = model.encode_text(text_tokens)
        else:
            text_features = model.encode_text(text_tokens)
    
    # Load and preprocess the image
    print(f"Loading image: {image_path}")
    try:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess image and add batch dimension
        image_tensor = preprocess(image).unsqueeze(0).to(device, non_blocking=True)
        
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
    # Evaluation
    model.eval()
    print("Running inference...")
    
    with torch.no_grad():
        # Forward pass with mixed precision for GPU optimization
        if device == "cuda":
            with autocast(enabled=True, dtype=torch.float16):
                # Use pre-encoded text features for efficiency
                image_features = model.encode_image(image_tensor)
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity (equivalent to logits_per_image)
                logits_per_image = (image_features @ text_features_norm.T) * model.logit_scale.exp()
        else:
            # CPU path
            logits_per_image, _ = model(image_tensor, text_tokens)
        
        # Calculate probabilities for each antonym pair
        predicted_probs = calculate_probabilities(logits_per_image, antonym_keys)
    
    # Convert tensors to float values for easier handling
    results = {}
    for key in antonym_keys:
        results[key] = predicted_probs[key].item()
    
    # Get image name without extension for output filename
    image_path_obj = Path(image_path)
    image_name = image_path_obj.stem  # Gets filename without extension
    
    # Print results
    print(f"\nResults for image: {image_path_obj.name}")
    print("=" * 60)
    for key, prob in results.items():
        antonym_pair = key.split('-')
        print(f"{key:20s}: {prob:.4f} (closer to '{antonym_pair[0]}')")
    print("=" * 60)
    
    # Create output filename using image name
    output_file = Path(output_dir) / f"{image_name}_results.csv"
    
    print(f"Exporting results to: {output_file}")
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file, delimiter=",")
        
        # Write header
        headers = ["image_name"] + [f"{key}_probability" for key in antonym_keys]
        writer.writerow(headers)
        
        # Write results
        row = [image_path_obj.name] + [results[key] for key in antonym_keys]
        writer.writerow(row)
    
    print(f"Results exported to: {output_file}")
    
    return results


def main():
    """Main execution function for single image evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a single image using trained WPCLIP model")
    parser.add_argument("-i", "--image_path", type=str, help="Path to the image file to evaluate")
    parser.add_argument("--model_path", type=str, default="ckpts/wpclip.pt", 
                       help="Path to the trained model weights (default: ckpts/wpclip.pt)")
    parser.add_argument("--output_dir", type=str, default="samples", 
                       help="Directory to save output CSV file (default: samples)")
    
    args = parser.parse_args()
    
    # Check if image path exists
    if not Path(args.image_path).exists():
        print(f"Error: Image file not found at {args.image_path}")
        print("Please provide a valid path to an image file.")
        sys.exit(1)
    
    # Check if model exists
    if not Path(args.model_path).exists():
        print(f"Error: Model file not found at {args.model_path}")
        print("Please update --model_path to point to your trained model.")
        sys.exit(1)
    
    # Run evaluation
    results = evaluate_single_image(args.model_path, args.image_path, args.output_dir)
    
    if results:
        print("\nEvaluation completed successfully!")
    else:
        print("Evaluation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()