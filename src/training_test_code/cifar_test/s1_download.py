import os
import torchvision
from tqdm import tqdm

def download_and_save_cifar(data_root: str = "./data"):
    """
    Downloads the official CIFAR-10 training set and saves the images as PNGs.
    Writes a flat folder of 50,000 RGB PNGs for training and metric references.
    Class labels are not saved.
    """
    print("Downloading official CIFAR-10 dataset...")
    # Download the dataset (PIL Images)
    dataset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True)
    
    # Define target directories
    official_dir = os.path.join(data_root, "cifar_official")
    flat_dir = os.path.join(official_dir, "fid_reference_flat")
    
    # Create the flat directory
    os.makedirs(flat_dir, exist_ok=True)
    
    print(f"Saving 50,000 images to {official_dir}...")
    
    # Iterate through the dataset and save images
    for i in tqdm(range(len(dataset)), desc="Processing CIFAR-10"):
        img, _ = dataset[i]
        
        # Format filename with leading zeros (e.g., 00001.png)
        filename = f"{i:05d}.png"
        
        # Save the reference image without modifying its pixels.
        flat_path = os.path.join(flat_dir, filename)
        img.save(flat_path, format="PNG")
        

    print("Done! CIFAR-10 images successfully extracted and saved.")

if __name__ == "__main__":
    download_and_save_cifar(data_root="./data")
