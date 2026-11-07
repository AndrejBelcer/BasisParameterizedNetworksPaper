import os
import torch
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
import concurrent.futures
from typing import Tuple

# Project imports
from src.utilities.tensor_utils import normalize_tensor

def generate_initial_noise(
    batch_size: int,
    channels: int,
    img_size: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
    noise_factor: float = 1.0,
):
    """Generates deterministic Gaussian noise using the supplied generator."""
    return (torch.randn((batch_size, channels, img_size, img_size), device=device, dtype=dtype, generator=generator) * noise_factor)

def _load_all_images(folder_path: str, image_size: int = 32, apply_horizontal_flip: bool = True) -> torch.Tensor:
    """Loads all images into a single CPU tensor."""
    print(f"Scanning {folder_path} for images...")
    images_list = []
    
    for root, _, files in os.walk(folder_path):
        for fname in files:
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(root, fname)
                try:
                    with Image.open(path) as img:
                        img = img.convert('RGB')
                        if img.size != (image_size, image_size):
                            img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
                        
                        arr = np.array(img).astype(np.float32) / 255.0
                        tensor = normalize_tensor(torch.from_numpy(arr).permute(2, 0, 1))
                        images_list.append(tensor)
                except Exception as e:
                    print(f"  [Warning] Failed to load {fname}: {e}")

    train_tensor = torch.stack(images_list)
    
    if apply_horizontal_flip:
        print(f"  [Augmentation] Applying horizontal flip...")
        flipped_tensor = torch.flip(train_tensor, dims=[3])
        train_tensor = torch.cat([train_tensor, flipped_tensor], dim=0)
        
    print(f"Loaded {len(train_tensor)} images. Shape: {train_tensor.shape}")
    return train_tensor

# --- Multiprocessing Worker Function ---
def _solve_assignment_chunk(args) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Computes the cost matrix and solves the Hungarian matching for a single chunk.
    Must be at the top level for multiprocessing pickling.
    """
    G_chunk, N_chunk, metric = args
    B = G_chunk.shape[0]
    
    # Flatten spatial dimensions for distance calculation
    G_flat = G_chunk.view(B, -1)
    N_flat = N_chunk.view(B, -1)
    
    if metric == 'L2':
        # Squared L2 distance
        cost_matrix = torch.cdist(G_flat, N_flat, p=2).pow(2).numpy()
    elif metric == 'L3':
        # L3 distance (Sum of absolute differences cubed)
        diff = G_flat.unsqueeze(1) - N_flat.unsqueeze(0) # [B, B, Features]
        cost_matrix = torch.sum(torch.abs(diff) ** 3, dim=2).numpy()
    else:
        raise ValueError(f"Unknown metric: {metric}")

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Calculate the distances of the chosen pairs
    chosen_distances = torch.tensor(cost_matrix[row_ind, col_ind])
    
    return col_ind, chosen_distances

# --- Main Assignment Logic ---

def CreateInitialPairs(
    dataset_path: str, 
    cache_save_path: str,
    image_size: int = 32,
    channels: int = 3,
    seed: int = 9999,
    metric: str = 'L2'
):
    """Loads images, generates noise, calculates initial random distance, and saves to cache."""
    print(f"\n--- Creating Initial Random Pairs (Seed: {seed}) ---")
    
    G = _load_all_images(dataset_path, image_size=image_size, apply_horizontal_flip=True)
    total_samples = len(G)
    
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)
    N = generate_initial_noise(total_samples, channels, image_size, torch.device('cpu'), torch.float32, generator)
    
    # Calculate initial distances (element-wise, no assignment)
    B = G.shape[0]
    G_flat = G.view(B, -1)
    N_flat = N.view(B, -1)
    
    if metric == 'L2':
        distances = torch.sum((G_flat - N_flat) ** 2, dim=1)
    elif metric == 'L3':
        distances = torch.sum(torch.abs(G_flat - N_flat) ** 3, dim=1)
    else:
        raise ValueError(f"Unknown metric: {metric}")
        
    total_distance = distances.sum().item()
    print(f"Initial Random Assignment Complete. Total {metric} Distance: {total_distance:,.2f}")
    
    # Sort globally by distance (shortest to longest)
    sorted_indices = torch.argsort(distances)
    G_sorted = G[sorted_indices]
    N_sorted = N[sorted_indices]
    distances_sorted = distances[sorted_indices]
    
    # Save Cache
    os.makedirs(os.path.dirname(cache_save_path), exist_ok=True)
    torch.save({"G": G_sorted, "N": N_sorted, "distances": distances_sorted}, cache_save_path)
    print(f"Saved initial dataset to {cache_save_path}")


def SequentialAssignment(
    cache_load_path: str, 
    hungarian_size: int = 1024, 
    num_workers: int = 5,
    metric: str = 'L2'
):
    """Loads existing pairs, chunks them sequentially, and runs Hungarian matching."""
    print(f"\n--- Starting Sequential Assignment ({metric}) ---")
    
    if not os.path.exists(cache_load_path):
        raise FileNotFoundError(f"Cache not found: {cache_load_path}. Run CreateInitialPairs first.")
        
    data = torch.load(cache_load_path)
    G = data["G"]
    N = data["N"]
    distances = data["distances"]
    total_samples = len(G)
    
    print(f"Loaded dataset. Current Total Distance: {distances.sum().item():,.2f}")
    
    # Prepare chunks
    chunks = []
    for i in range(0, total_samples, hungarian_size):
        G_chunk = G[i : i + hungarian_size]
        N_chunk = N[i : i + hungarian_size]
        chunks.append((G_chunk, N_chunk, metric))
        
    print(f"Split dataset into {len(chunks)} chunks. Running Hungarian Matching...")
    
    # Process in parallel
    new_N = torch.empty_like(N)
    all_distances = []
    total_distance = 0.0
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(_solve_assignment_chunk, chunks))
        
        for i, (col_ind, chunk_distances) in enumerate(results):
            start_idx = i * hungarian_size
            end_idx = start_idx + len(col_ind)
            
            # Reorder N based on assignment
            new_N[start_idx:end_idx] = chunks[i][1][col_ind]
            all_distances.append(chunk_distances)
            total_distance += chunk_distances.sum().item()
            
    all_distances = torch.cat(all_distances)
    print(f"Sequential Assignment Complete. New Total Distance: {total_distance:,.2f}")
    
    # Sort globally by distance (shortest to longest)
    sorted_indices = torch.argsort(all_distances)
    G_sorted = G[sorted_indices]
    N_sorted = new_N[sorted_indices]
    distances_sorted = all_distances[sorted_indices]
    
    # Overwrite Cache
    torch.save({"G": G_sorted, "N": N_sorted, "distances": distances_sorted}, cache_load_path)
    print(f"Overwrote sorted dataset to {cache_load_path}")


def TailHeavyAssignment(
    cache_load_path: str, 
    hungarian_size: int = 1024, 
    num_workers: int = 5, 
    num_chunks: int = 25,
    metric: str = 'L2'
):
    """Refines matching by heavily sampling pairs with the worst (longest) distances."""
    print(f"\n--- Starting Tail-Heavy Assignment ({metric}) ---")
    
    if not os.path.exists(cache_load_path):
        raise FileNotFoundError(f"Cache not found: {cache_load_path}. Run CreateInitialPairs first.")
        
    data = torch.load(cache_load_path)
    G = data["G"]
    N = data["N"]
    distances = data["distances"]
    total_samples = len(G)
    
    print(f"Loaded dataset. Current Total Distance: {distances.sum().item():,.2f}")
    
    # 1. Generate tail-heavy probabilities (linear increase towards the end)
    weights = np.linspace(0.01, 1.0, total_samples)
    weights /= weights.sum()
    
    total_to_sample = min(num_chunks * hungarian_size, total_samples)
    
    # 2. Sample indices without replacement
    chosen_indices = np.random.choice(total_samples, size=total_to_sample, replace=False, p=weights)
    np.random.shuffle(chosen_indices) # Shuffle so chunks are randomized
    
    # 3. Prepare chunks
    chunks = []
    chunk_indices_list = []
    
    for i in range(0, total_to_sample, hungarian_size):
        idx_chunk = chosen_indices[i : i + hungarian_size]
        chunk_indices_list.append(idx_chunk)
        chunks.append((G[idx_chunk], N[idx_chunk], metric))
        
    print(f"Sampled {total_to_sample} pairs into {len(chunks)} chunks. Running matching...")
    
    # 4. Process in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(_solve_assignment_chunk, chunks))
        
        for i, (col_ind, chunk_distances) in enumerate(results):
            idx_chunk = chunk_indices_list[i]
            
            # Update N and distances in their original global positions
            N[idx_chunk] = chunks[i][1][col_ind]
            distances[idx_chunk] = chunk_distances

    new_total_distance = distances.sum().item()
    print(f"Tail-Heavy Assignment Complete. New Total Distance: {new_total_distance:,.2f}")
    
    # 5. Re-sort globally
    sorted_indices = torch.argsort(distances)
    G_sorted = G[sorted_indices]
    N_sorted = N[sorted_indices]
    distances_sorted = distances[sorted_indices]
    
    # 6. Overwrite Cache
    torch.save({"G": G_sorted, "N": N_sorted, "distances": distances_sorted}, cache_load_path)
    print(f"Overwrote sorted dataset to {cache_load_path}")



def RandomAssignment(
    cache_load_path: str, 
    hungarian_size: int = 2048, 
    num_workers: int = 60,
    metric: str = 'L2'
):
    """Randomly chunks the dataset to allow global mixing of good and bad pairs."""
    print(f"\n--- Starting Pure Random Assignment ({metric}) ---")
    
    data = torch.load(cache_load_path)
    G, N, distances = data["G"], data["N"], data["distances"]
    total_samples = len(G)
    
    # 1. Generate a completely random permutation of all indices
    shuffled_indices = np.random.permutation(total_samples)
    
    chunks = []
    chunk_indices_list = []
    
    # 2. Group into mutually exclusive chunks
    for i in range(0, total_samples, hungarian_size):
        idx_chunk = shuffled_indices[i : i + hungarian_size]
        chunk_indices_list.append(idx_chunk)
        chunks.append((G[idx_chunk], N[idx_chunk], metric))
        
    print(f"Created {len(chunks)} random chunks. Running matching on {num_workers} cores...")
    
    # 3. Process in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(_solve_assignment_chunk, chunks))
        
        for i, (col_ind, chunk_distances) in enumerate(results):
            idx_chunk = chunk_indices_list[i]
            N[idx_chunk] = chunks[i][1][col_ind]
            distances[idx_chunk] = chunk_distances

    print(f"Random Assignment Complete. New Total Distance: {distances.sum().item():,.2f}")
    
    # 4. Re-sort globally and save
    sorted_indices = torch.argsort(distances)
    torch.save({"G": G[sorted_indices], "N": N[sorted_indices], "distances": distances[sorted_indices]}, cache_load_path)


def SurgicalTailAssignment(
    cache_load_path: str, 
    hungarian_size: int = 2048, 
    num_workers: int = 60,
    tail_size: int = 25000,
    metric: str = 'L2'
):
    """Pairs the worst offenders with completely random pairs to provide 'good' spare noises."""
    print(f"\n--- Starting Surgical Tail-Heavy Assignment ({metric}) ---")
    
    data = torch.load(cache_load_path)
    G, N, distances = data["G"], data["N"], data["distances"]
    total_samples = len(G)
    
    # 1. Identify the worst pairs (tail) and the rest (pool)
    # Assuming data is already sorted shortest to longest
    worst_indices = np.arange(total_samples - tail_size, total_samples)
    pool_indices = np.arange(0, total_samples - tail_size)
    
    # 2. Randomly sample from the pool to match the tail size
    random_helpers = np.random.choice(pool_indices, size=tail_size, replace=False)
    
    # 3. Combine and shuffle so chunks get a 50/50 mix of worst and random
    combined_indices = np.concatenate([worst_indices, random_helpers])
    np.random.shuffle(combined_indices)
    
    chunks = []
    chunk_indices_list = []
    
    for i in range(0, len(combined_indices), hungarian_size):
        idx_chunk = combined_indices[i : i + hungarian_size]
        chunk_indices_list.append(idx_chunk)
        chunks.append((G[idx_chunk], N[idx_chunk], metric))
        
    print(f"Surgical chunks created ({len(chunks)}). Running matching...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(_solve_assignment_chunk, chunks))
        
        for i, (col_ind, chunk_distances) in enumerate(results):
            idx_chunk = chunk_indices_list[i]
            N[idx_chunk] = chunks[i][1][col_ind]
            distances[idx_chunk] = chunk_distances

    print(f"Surgical Assignment Complete. New Total Distance: {distances.sum().item():,.2f}")
    
    sorted_indices = torch.argsort(distances)
    torch.save({"G": G[sorted_indices], "N": N[sorted_indices], "distances": distances[sorted_indices]}, cache_load_path)


def StochasticPairwiseSwap(cache_load_path: str, iterations: int = 1000, metric: str = 'L2'):
    """
    Highly optimized, vectorized 2x2 swapping. 
    Runs millions of swap checks per second on the CPU without multiprocessing.
    """
    print(f"\n--- Starting Vectorized Stochastic Swapping ({iterations} iterations) ---")
    
    data = torch.load(cache_load_path)
    G, N, distances = data["G"], data["N"], data["distances"]
    total_samples = len(G)
    
    # Create views ONCE. 
    # Because they are views, any changes to N will automatically reflect in N_flat!
    G_flat = G.view(total_samples, -1)
    N_flat = N.view(total_samples, -1)
    
    successful_swaps = 0
    
    for _ in range(iterations):
        # 1. Create mutually exclusive pairs by shuffling all indices and splitting in half
        perm = torch.randperm(total_samples)
        idx_A = perm[:total_samples // 2]
        idx_B = perm[total_samples // 2 : (total_samples // 2) * 2]
        
        # 2. Get current distances for these pairs
        dist_A_current = distances[idx_A]
        dist_B_current = distances[idx_B]
        current_total = dist_A_current + dist_B_current
        
        # 3. Calculate swapped distances
        if metric == 'L2':
            # torch.square is slightly faster than ** 2
            dist_A_swap = torch.sum(torch.square(G_flat[idx_A] - N_flat[idx_B]), dim=1)
            dist_B_swap = torch.sum(torch.square(G_flat[idx_B] - N_flat[idx_A]), dim=1)
        elif metric == 'L3':
            dist_A_swap = torch.sum(torch.abs(G_flat[idx_A] - N_flat[idx_B]) ** 3, dim=1)
            dist_B_swap = torch.sum(torch.abs(G_flat[idx_B] - N_flat[idx_A]) ** 3, dim=1)
            
        swap_total = dist_A_swap + dist_B_swap
        
        # 4. Find where swapping improves the distance
        improvement_mask = swap_total < current_total
        
        if improvement_mask.any():
            # Extract the actual indices that need swapping
            swap_idx_A = idx_A[improvement_mask]
            swap_idx_B = idx_B[improvement_mask]
            
            # Perform the swap on N (N_flat updates automatically!)
            temp_N = N[swap_idx_A].clone()
            N[swap_idx_A] = N[swap_idx_B]
            N[swap_idx_B] = temp_N
            
            # Update distances
            distances[swap_idx_A] = dist_A_swap[improvement_mask]
            distances[swap_idx_B] = dist_B_swap[improvement_mask]
            
            successful_swaps += improvement_mask.sum().item()

    print(f"Stochastic Swapping Complete. Performed {successful_swaps:,} beneficial swaps.")
    print(f"New Total Distance: {distances.sum().item():,.2f}")
    
    sorted_indices = torch.argsort(distances)
    torch.save({"G": G[sorted_indices], "N": N[sorted_indices], "distances": distances[sorted_indices]}, cache_load_path)





def DeterministicTailSwap(
    cache_load_path: str, 
    k_worst: int = 1000, 
    num_workers: int = 7, 
    metric: str = 'L2'
):
    """
    Sequentially takes the worst K pairs and finds the absolute best single swap 
    across the entire dataset of 100,000 pairs.
    """
    print(f"\n--- Starting Deterministic Tail Swap (Targeting worst {k_worst} pairs) ---")
    
    # Tell PyTorch to use your 7 laptop cores for the underlying C++ tensor math
    torch.set_num_threads(num_workers)
    
    data = torch.load(cache_load_path)
    G, N, distances = data["G"], data["N"], data["distances"]
    total_samples = len(G)
    
    # Create flat views (modifying N will automatically update N_flat)
    G_flat = G.view(total_samples, -1)
    N_flat = N.view(total_samples, -1)
    
    successful_swaps = 0
    total_improvement = 0.0
    
    # We iterate backwards from the end of the array (the worst distances)
    # Note: We use a while loop or dynamic index because we want to re-evaluate 
    # the "current" worst if we want to be perfectly optimal, but iterating 
    # through the original worst K indices is faster and matches your prompt.
    
    for step in range(1, k_worst + 1):
        # Index of the current worst item we are targeting
        w = total_samples - step
        
        # Extract the single worst image and noise
        G_w = G_flat[w].unsqueeze(0) # Shape: [1, 3072]
        N_w = N_flat[w].unsqueeze(0) # Shape: [1, 3072]
        D_w = distances[w]
        
        # 1. Calculate distance if G_w is paired with EVERY noise in the dataset
        if metric == 'L2':
            dist_Gw_Nall = torch.sum(torch.square(G_w - N_flat), dim=1)
            # 2. Calculate distance if EVERY image is paired with N_w
            dist_Gall_Nw = torch.sum(torch.square(G_flat - N_w), dim=1)
        elif metric == 'L3':
            dist_Gw_Nall = torch.sum(torch.abs(G_w - N_flat) ** 3, dim=1)
            dist_Gall_Nw = torch.sum(torch.abs(G_flat - N_w) ** 3, dim=1)
            
        # 3. Calculate the total distance of the two pairs IF we swapped them
        swap_totals = dist_Gw_Nall + dist_Gall_Nw
        
        # 4. Calculate the current total distance of the two pairs
        current_totals = D_w + distances
        
        # 5. Find the improvement for all 100,000 possible swaps
        improvements = current_totals - swap_totals
        
        # 6. Find the absolute best swap
        best_idx = torch.argmax(improvements).item()
        best_improvement = improvements[best_idx].item()
        
        # 7. If the best swap actually improves the total distance, do it!
        if best_improvement > 0:
            # Swap the noises
            temp_N = N[w].clone()
            N[w] = N[best_idx]
            N[best_idx] = temp_N
            
            # Update the distances
            distances[w] = dist_Gw_Nall[best_idx]
            distances[best_idx] = dist_Gall_Nw[best_idx]
            
            successful_swaps += 1
            total_improvement += best_improvement
            
            if step % 100 == 0 or step == 1:
                print(f"  [Target {step}/{k_worst}] Swapped worst pair with index {best_idx}. Improvement: {best_improvement:,.2f}")
        else:
            if step % 100 == 0 or step == 1:
                print(f"  [Target {step}/{k_worst}] No beneficial swap found for this pair.")

    print(f"\nDeterministic Swap Complete.")
    print(f"Performed {successful_swaps} optimal swaps.")
    print(f"Total distance reduced by: {total_improvement:,.2f}")
    print(f"New Total Distance: {distances.sum().item():,.2f}")
    
    # Re-sort globally and save
    sorted_indices = torch.argsort(distances)
    torch.save({"G": G[sorted_indices], "N": N[sorted_indices], "distances": distances[sorted_indices]}, cache_load_path)
    print(f"Saved optimized dataset to {cache_load_path}")




def PrepareDataset(dataset_path: str = r"data/cifar_official/fid_reference_flat", cache_path: str = r"cache/training_pairs/cifar_fixed_pairs.pt", metric: str = 'L2'):
    DATASET_PATH = dataset_path
    CACHE_PATH = cache_path
    METRIC = metric

    num_workers = 7
    batch_size = 2048 # 100k / 2048 = ~48 chunks
    epochs = 10
    num_chunks = 25
    k_worst = 0
    tail_size = 20480

    # 1. Initial Creation
    if not os.path.exists(CACHE_PATH):
        CreateInitialPairs(DATASET_PATH, CACHE_PATH, seed=9999, metric=METRIC)

    # 2. The Optimization Loop
    for epoch in range(epochs):
        print(f"\n================ EPOCH {epoch+1}/{epochs} ================")
        
        # Step 1: Global structure (Sequential & Random)
        SequentialAssignment(CACHE_PATH, metric=METRIC, hungarian_size=batch_size, num_workers=num_workers)
        for _ in range(2):
            RandomAssignment(CACHE_PATH, metric=METRIC, hungarian_size=batch_size, num_workers=num_workers)
        
        # Step 2: Micro-smoothing (Fast vectorized swaps to untangle minor criss-crosses)
        # 1000 iterations takes just a few seconds but tests 50 million potential swaps!
        #StochasticPairwiseSwap(CACHE_PATH, iterations=1000, metric=METRIC)
        
        # Step 3: Attack the Tails (Original Tail-Heavy & Surgical Tail-Heavy)
        SequentialAssignment(CACHE_PATH, metric=METRIC, hungarian_size=batch_size, num_workers=num_workers)
        for _ in range(2):
            TailHeavyAssignment(CACHE_PATH, num_chunks=num_chunks, hungarian_size=batch_size, metric=METRIC, num_workers=num_workers)
        SurgicalTailAssignment(CACHE_PATH, tail_size=tail_size, hungarian_size=batch_size, metric=METRIC, num_workers=num_workers)
        
        # Step 4: Final Micro-smoothing for the epoch
        #StochasticPairwiseSwap(CACHE_PATH, iterations=1000, metric=METRIC)


    DeterministicTailSwap(CACHE_PATH, k_worst=k_worst, num_workers=num_workers, metric=METRIC)






def QuickTail(cache_path: str = r"cache/training_pairs/cifar_fixed_pairs.pt", metric: str = 'L2'):
    CACHE_PATH = cache_path
    METRIC = metric

    num_workers = 7
    DeterministicTailSwap(CACHE_PATH, k_worst=100, num_workers=num_workers, metric=METRIC)






if __name__ == "__main__":
    PrepareDataset()
