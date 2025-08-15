from PIL import Image
from torch.utils.data import Dataset
import pandas as pd


class WolfflinPrinciplesDataset(Dataset):
    """Dataset for Wölflin's principles of art classification."""
    
    def __init__(self, data_root, preprocess, data_type="training"):
        """
        Initialize the dataset.
        
        Args:
            data_root (str): Root directory containing images
            preprocess: Image preprocessing function
            data_type (str): Type of data - "training", "validation", or "test"
        """
        self.preprocess = preprocess
        self.img_paths = []
        
        # Load appropriate CSV based on data type
        if data_type == "test":
            art_df = pd.read_csv("./data/art_classes_gan.csv")
            principles = list(art_df.columns[1:])
        else:
            # Use real art images for training
            art_df = pd.read_csv("./data/art_classes_real.csv")
            art_df = art_df[art_df["set"] == data_type]
            principles = list(art_df.columns[1:-1])
        
        # Extract principle antonym pairs
        self.antonyms = [
            [p.split("_")[0] for p in principle.split("-vs-")] 
            for principle in principles
        ]
        
        # Create keys for antonym pairs
        self.antonym_keys = [f"{p[0]}-{p[1]}" for p in self.antonyms]
        
        # Initialize scores dictionary
        self.scores = {k: [] for k in self.antonym_keys}
        
        # Populate image paths and scores
        for _, row in art_df.iterrows():
            self.img_paths.append(data_root + row["img_id"])
            
            for j, key in enumerate(self.antonym_keys):
                # Normalize score from 0-5 scale to 0-1 scale
                self.scores[key].append(row[principles[j]] / 5.0)
    
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: Dictionary containing image path, preprocessed image, and scores
        """
        img_path = self.img_paths[idx]
        img = self.preprocess(Image.open(img_path))
        
        # Build return dictionary
        return_dict = {
            "img_path": img_path, 
            "img": img
        }
        
        # Add scores for each antonym pair
        for key in self.antonym_keys:
            return_dict[key] = self.scores[key][idx]
        
        return return_dict