import torch
from torch import nn
import clip
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import csv
from pathlib import Path

from configs import wpclip_v1 as cfg
from data import wp_data


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


def setup_csv_file(output_file, antonym_keys):
    """
    Setup CSV file with appropriate headers.
    
    Args:
        output_file (str): Path to output CSV file
        antonym_keys (list): List of antonym pair keys
        
    Returns:
        tuple: (file_handle, csv_writer)
    """
    # Create column headers
    columns = ["image_name"] + antonym_keys + [f"{key.split('-')[0]}_pred" for key in antonym_keys]
    
    # Open file and write headers
    file_handle = open(output_file, mode="w", newline="")
    writer = csv.writer(file_handle, delimiter=",")
    writer.writerow(columns)
    
    return file_handle, writer


def write_batch_results(csv_writer, data_blob, probabilities, antonym_keys):
    """
    Write batch results to CSV file.
    
    Args:
        csv_writer: CSV writer object
        data_blob: Batch data containing image paths and ground truth
        probabilities: Predicted probabilities
        antonym_keys: List of antonym pair keys
    """
    batch_size = len(data_blob["img_path"])
    
    for i in range(batch_size):
        # Extract just the image filename from full path
        full_path = data_blob["img_path"][i]
        image_name = Path(full_path).name
        
        # Start row with image filename only
        row = [image_name]
        
        # Add ground truth scores
        for key in antonym_keys:
            row.append(data_blob[key][i].item())
        
        # Add predicted scores
        for key in antonym_keys:
            row.append(probabilities[key][i].item())
        
        csv_writer.writerow(row)


def evaluate_and_export(model_path, output_file):
    """
    Main function to evaluate model and export results to CSV.
    
    Args:
        model_path (str): Path to trained model weights
        output_file (str): Path for output CSV file
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
        # Ensure model is on GPU
        model = model.cuda()
    else:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    
    # Setup dataset and dataloader
    dataset_path = "/scratch/abhijay/datasets/WAGA/analyses_notebooks/datasets/raw/imgs/"
    test_dataset = wp_data.WolfflinPrinciplesDataset(
        dataset_path, 
        preprocess,
        data_type="test"
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.test["test_batch_size"], 
        shuffle=False  # Changed to False for consistent ordering
    )
    
    # Get antonym keys and create text prompts
    antonym_keys = test_dataset.antonym_keys
    txt_prompts = [term for antonym_pair in test_dataset.antonyms for term in antonym_pair]
    
    # Setup for evaluation
    if device == "cpu":
        model.float()
    else:
        # Optimize for GPU usage
        model.half()  # Use fp16 for faster GPU inference
        torch.backends.cudnn.benchmark = True  # Optimize cudnn for consistent input sizes
    
    autocast = get_autocast()
    loss_fn = nn.MSELoss()
    text_tokens = clip.tokenize(txt_prompts).to(device)
    
    # Pre-encode text tokens for efficiency
    print("Pre-encoding text tokens...")
    with torch.no_grad():
        if device == "cuda":
            with autocast(enabled=True, dtype=torch.float16):
                text_features = model.encode_text(text_tokens)
        else:
            text_features = model.encode_text(text_tokens)
    
    # Setup CSV output
    csv_file, csv_writer = setup_csv_file(output_file, antonym_keys)
    
    # Evaluation loop
    model.eval()
    test_losses = {key: [] for key in antonym_keys}
    
    print(f"Starting evaluation and export to: {output_file}")
    
    try:
        with torch.no_grad():
            for batch_idx, data_blob in enumerate(tqdm(test_loader, desc="Evaluating")):
                # Prepare batch data
                images = data_blob["img"].to(device, non_blocking=True)
                ground_truth_scores = {
                    key: data_blob[key].float().to(device, non_blocking=True) 
                    for key in antonym_keys
                }
                
                # Forward pass with mixed precision for GPU optimization
                if device == "cuda":
                    with autocast(enabled=True, dtype=torch.float16):
                        # Use pre-encoded text features for efficiency
                        image_features = model.encode_image(images)
                        # Normalize features
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
                        
                        # Calculate similarity (equivalent to logits_per_image)
                        logits_per_image = (image_features @ text_features_norm.T) * model.logit_scale.exp()
                else:
                    # CPU path
                    logits_per_image, _ = model(images, text_tokens)
                
                # Calculate probabilities for each antonym pair
                predicted_probs = calculate_probabilities(logits_per_image, antonym_keys)
                
                # Calculate losses
                for key in antonym_keys:
                    loss = loss_fn(predicted_probs[key], ground_truth_scores[key])
                    test_losses[key].append(loss)
                
                # Write results to CSV
                write_batch_results(csv_writer, data_blob, predicted_probs, antonym_keys)
    
    finally:
        # Ensure CSV file is properly closed
        csv_file.close()
    
    # Calculate and print results
    avg_losses = {
        key: sum(losses).item() / len(losses) 
        for key, losses in test_losses.items()
    }
    
    total_avg_loss = sum(avg_losses.values())
    
    print("\nEvaluation Results:")
    print("=" * 60)
    for key, avg_loss in avg_losses.items():
        print(f"{key:20s}: {avg_loss:.6f}")
    print("=" * 60)
    print(f"Total Average Loss: {total_avg_loss:.6f}")
    print(f"Results exported to: {output_file}")
    
    return avg_losses, total_avg_loss

def main():
    """Main execution function."""
    # Configuration
    MODEL_PATH = "ckpts/wpclip.pt"
    OUTPUT_FILE = "clip_FT.csv"
    
    # Ensure output directory exists
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    
    # Run evaluation
    evaluate_and_export(MODEL_PATH, OUTPUT_FILE)


if __name__ == "__main__":
    main()