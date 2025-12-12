import os
import sys
import torch
import torch.nn as nn
import numpy as np
import random

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import entropy

import torch.distributed as dist

from torch.optim.lr_scheduler import _LRScheduler

class WarmupDecayScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, final_lr=1e-6, warmup_start_lr=1e-8):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.final_lr = final_lr
        self.warmup_start_lr = warmup_start_lr
        super(WarmupDecayScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [self.warmup_start_lr + alpha * (self.base_lr - self.warmup_start_lr) for _ in self.base_lrs]
        else:
            # Linear decay
            alpha = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return [self.base_lr + alpha * (self.final_lr - self.base_lr) for _ in self.base_lrs]

# Set fixed seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# Initialize distributed environment
def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print(f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)
    
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()

# Custom distributed utils
def is_main_process():
    """Check if this is the main process in distributed training."""
    if not torch.distributed.is_available():
        return True
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0

def get_rank():
    """Get the rank of the current process."""
    if not torch.distributed.is_available():
        return 0
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()

def get_world_size():
    """Get the world size (total number of processes)."""
    if not torch.distributed.is_available():
        return 1
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()




def chunked_all_gather(tensor, chunk_size=1000):
    """
    Splits a tensor into smaller chunks and performs all_gather on each chunk.
    This prevents timeout issues with large tensors.
    
    Args:
        tensor: Input tensor to gather across processes
        chunk_size: Maximum number of elements per dimension 0 chunk
        
    Returns:
        List of gathered tensors from all processes
    """
    world_size = get_world_size()
    
    if world_size == 1:
        return [tensor]
    
    # Get current device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        # If no CUDA, we can't do distributed operations
        return [tensor]
    
    # Ensure tensor is on the correct device
    if tensor.device != device:
        tensor = tensor.to(device)
    
    # Split tensor into chunks along dimension 0
    num_chunks = (tensor.size(0) + chunk_size - 1) // chunk_size  # Ceiling division
    
    if is_main_process():
        print(f"Splitting tensor of shape {tensor.shape} into {num_chunks} chunks for all_gather")
    
    # Process chunks one by one to avoid OOM
    gathered_chunks = []
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, tensor.size(0))
        chunk = tensor[start_idx:end_idx]
        
        if is_main_process():
            print(f"Processing chunk {i+1}/{num_chunks}, shape: {chunk.shape}")
        
        # Determine chunk sizes for all processes
        local_size = torch.tensor([chunk.size(0)], device=device)
        size_list = [torch.zeros_like(local_size) for _ in range(world_size)]
        
        try:
            # Gather chunk sizes
            dist.all_gather(size_list, local_size)
            size_list = [int(size.item()) for size in size_list]
            
            # Find max size for padding
            max_size = max(size_list)
            
            # Pad chunk if necessary
            size_diff = max_size - chunk.size(0)
            if size_diff > 0:
                padding = torch.zeros((size_diff, *chunk.shape[1:]), 
                                      dtype=chunk.dtype, device=device)
                chunk = torch.cat([chunk, padding], dim=0)
            
            # Gather padded chunks
            output_tensors = [torch.zeros((max_size, *chunk.shape[1:]), 
                                          dtype=chunk.dtype, device=device) 
                               for _ in range(world_size)]
            
            dist.all_gather(output_tensors, chunk)
            
            # Trim padding based on original sizes
            for j, size in enumerate(size_list):
                output_tensors[j] = output_tensors[j][:size]
            
            gathered_chunks.extend(output_tensors)
            
            # Clean up GPU memory after each chunk
            del chunk, output_tensors
            torch.cuda.empty_cache()
            
        except Exception as e:
            if is_main_process():
                print(f"Error in all_gather for chunk {i+1}/{num_chunks}: {e}")
                # On error, just return what we have so far
                return gathered_chunks if gathered_chunks else [tensor]
    
    return gathered_chunks
    
def all_gather(tensor):
    """Wrapper around chunked_all_gather to maintain compatibility"""
    return chunked_all_gather(tensor)

def barrier():
    """Synchronization point for all processes."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()



# Model loader - extracts backbone from checkpoint
def load_dino_backbone(checkpoint_path, device):
    """
    Load the DINO backbone from a checkpoint file.
    Returns the extracted backbone model.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Check if student key exists
    if 'student' not in checkpoint:
        raise ValueError(f"Checkpoint doesn't contain 'student' key. Available keys: {list(checkpoint.keys())}")
    
    # Extract architecture parameters from args
    args = checkpoint['args']
    
    # Print available args for debugging
    if is_main_process():
        print(f"Using architecture parameters from checkpoint args")
    
    # Extract ViT parameters
    img_size = 224  # Usually fixed
    patch_size = args.patch_size
    embed_dim = args.embeddingdim
    depth = args.vitdepth
    num_heads = args.vitheads
    num_register_tokens = 4  # This might not be explicitly in args
    
    if is_main_process():
        print(f"Model architecture: patch_size={patch_size}, "
              f"embed_dim={embed_dim}, depth={depth}, num_heads={num_heads}")
    
    # Extract the backbone from the student model
    backbone_state_dict = {}
    for k, v in checkpoint['student'].items():
        # Handle different checkpoint formats
        if k.startswith('module.backbone.'):
            new_key = k.replace('module.backbone.', '')
            backbone_state_dict[new_key] = v
        elif k.startswith('backbone.'):
            new_key = k.replace('backbone.', '')
            backbone_state_dict[new_key] = v
        elif k.startswith('module.'):
            # This is the case if the backbone is directly wrapped in DDP
            new_key = k.replace('module.', '')
            backbone_state_dict[new_key] = v
    
    # Import the correct model architecture
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from models import VisionTransformer
    
    # Create a model with the extracted architecture parameters
    backbone = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        num_register_tokens=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_norm=False,
        dual_norm=False,  # Equivalent to dp_norm in previous model
        drop_path_rate=0.4,  # Matching original implementation
        pre_norm=False,
    )
    
    # Load the state dict
    missing_keys, unexpected_keys = backbone.load_state_dict(backbone_state_dict, strict=False)
    
    if is_main_process():
        if missing_keys:
            print(f"Warning: Missing keys in state_dict: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in state_dict: {unexpected_keys[:5]}...")
    
    backbone.to(device)
    backbone.eval()
    
    # Remove output head if present
    if hasattr(backbone, 'fc'):
        backbone.fc = nn.Identity()
    if hasattr(backbone, 'head'):
        backbone.head = nn.Identity()
        
    return backbone




########################################################
# Representation Metrics
########################################################



def original_twonn_dimension(data, return_xy=False):
    """
    Calculates intrinsic dimension of the provided data points with the TWO-NN algorithm.
    
    -----------
    Parameters:
    
    data : 2d array-like
        2d data matrix. Samples on rows and features on columns.

    return_xy : bool (default=False)
        Whether to return also the coordinate vectors used for the linear fit.
        
    -----------
    Returns:
    
    d : int
        Intrinsic dimension of the dataset according to TWO-NN.

    x : 1d array (optional)
        Array with the -log(mu) values.

    y : 1d array (optional)
        Array with the -log(F(mu_{sigma(i)})) values.
        
    -----------
    References:
    
    [1] E. Facco, M. d’Errico, A. Rodriguez & A. Laio
        Estimating the intrinsic dimension of datasets by a minimal neighborhood information (https://doi.org/10.1038/s41598-017-11873-y)
    
    
    """
    
    
    data = np.array(data)
    
    N = len(data)
    
    #mu = r2/r1 for each data point
    mu = []
    for i,x in enumerate(data):
        
        dist = np.sort(np.sqrt(np.sum((x-data)**2, axis=1)))
        r1, r2 = dist[dist>0][:2]

        mu.append((i+1,r2/r1))
        

    #permutation function
    sigma_i = dict(zip(range(1,len(mu)+1), np.array(sorted(mu, key=lambda x: x[1]))[:,0].astype(int)))

    mu = dict(mu)

    #cdf F(mu_{sigma(i)})
    F_i = {}
    for i in mu:
        F_i[sigma_i[i]] = i/N

    #fitting coordinates
    x = np.log([mu[i] for i in sorted(mu.keys())])
    y = np.array([1-F_i[i] for i in sorted(mu.keys())])

    #avoid having log(0)
    x = x[y>0]
    y = y[y>0]

    y = -1*np.log(y)

    #fit line through origin to get the dimension
    d = np.linalg.lstsq(np.vstack([x, np.zeros(len(x))]).T, y, rcond=None)[0][0]
        
    if return_xy:
        return d, x, y
    else: 
        return d



def twonn_dimension(data, return_xy=False, min_threshold=1e-10):
    """
    Calculates intrinsic dimension with TWO-NN algorithm, skipping problematic points.
    
    Args:
        data: Input data matrix
        return_xy: Whether to return fitting coordinates
        min_threshold: Minimum distance to consider valid (skip if r1 < this)
        
    Returns:
        d: Intrinsic dimension or feature dimension as fallback
    """
    data = np.array(data)
    N = len(data)
    
    # mu = r2/r1 for each data point, now with safety checks
    mu = []
    for i, x in enumerate(data):
        try:
            dist = np.sort(np.sqrt(np.sum((x-data)**2, axis=1)))
            # Extract non-zero distances
            non_zero_dist = dist[dist > min_threshold]
            
            # Skip if we don't have at least 2 valid neighbors
            if len(non_zero_dist) < 2:
                continue
                
            r1, r2 = non_zero_dist[:2]
            mu.append((i+1, r2/r1))
        except Exception:
            # Skip any point causing errors
            continue
    
    # Skip dimension calculation if we don't have enough valid points
    if len(mu) < 3:
        return data.shape[1]  # Return feature dimension as fallback
    
    # Continue with original algorithm using only valid points
    sigma_i = dict(zip(range(1, len(mu)+1), np.array(sorted(mu, key=lambda x: x[1]))[:,0].astype(int)))
    mu = dict(mu)

    # cdf F(mu_{sigma(i)})
    F_i = {}
    for i in mu:
        F_i[sigma_i[i]] = i/N

    # fitting coordinates
    x = np.log([mu[i] for i in sorted(mu.keys())])
    y = np.array([1-F_i[i] for i in sorted(mu.keys())])

    # avoid having log(0)
    x = x[y>0]
    y = y[y>0]
    y = -1*np.log(y)

    # Handle case where fitting might fail
    try:
        # Original fitting code
        d = np.linalg.lstsq(np.vstack([x, np.zeros(len(x))]).T, y, rcond=None)[0][0]
        
        # Add validation of result
        if not np.isfinite(d) or d <= 0:
            return data.shape[1]  # Return feature dimension as fallback
    except np.linalg.LinAlgError:
        # Return feature dimension if SVD fails
        return data.shape[1]
        
    if return_xy:
        return d, x, y
    else: 
        return d

def calculate_clid_metric(test_features):
    """
    Calculate CLID metric with robust handling of edge cases.
    
    Args:
        test_features: Feature matrix (tensor)
        
    Returns:
        cl: Contrastive learning intrinsic dimensionality
        global_d: Global intrinsic dimensionality
    """
    # Convert PyTorch tensor to NumPy array if necessary
    X = test_features.numpy() if hasattr(test_features, 'numpy') else np.array(test_features)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    train_samples = int(0.8 * X_std.shape[0])  # 80% of the data for training
    X_train = X_std[:train_samples]
    X_val = X_std[train_samples:]

    kmeans = MiniBatchKMeans(n_clusters=int(np.sqrt(X_std.shape[0])), n_init='auto')
    kmeans.fit(X_train)
    tr_clusters = kmeans.predict(X_train)
    val_clusters = kmeans.predict(X_val)

    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X_train, tr_clusters)
    val_knn_labels = clf.predict(X_val)

    cl = accuracy_score(val_clusters, val_knn_labels)
    
    # Calculate dimensionality with robust batch handling
    global_d_values = []
    for i in range(0, X_std.shape[0], 1000):
        batch = X_std[i:i+1000]
        if len(batch) >= 3:  # Only process batch if it has enough points
            dim = twonn_dimension(batch, min_threshold=1e-10)
            global_d_values.append(dim)
    
    # Calculate mean from valid batches
    global_d = np.mean(global_d_values) if global_d_values else X_std.shape[1]
    
    return cl/global_d


def calculate_lidar_metric(test_features, delta=1e-5, eps=1e-8):
    """
    Calculate the LiDAR metric for feature representations.
    
    Args:
        test_features: Features with shape [n_samples, n_augmentations, feature_dim]
        delta: Small value for numerical stability
        eps: Small value to avoid division by zero
        
    Returns:
        lidar_metric: The computed LiDAR metric
    """
    # Convert PyTorch tensor to NumPy array if necessary
    if isinstance(test_features, torch.Tensor):
        test_features = test_features.cpu().numpy()
    
    # Don't assume a fixed number of augmentations
    n_samples, n_augmentations, feature_dim = test_features.shape
    
    # Reshape for LDA
    X = test_features.reshape(-1, feature_dim)
    y = np.repeat(np.arange(n_samples), n_augmentations)
    
    # Fit LDA
    lda = LinearDiscriminantAnalysis(solver='svd')
    lda.fit(X, y)
    
    # Calculate metric
    lda_matrix = lda.scalings_
    eigenvalues = np.linalg.eigvalsh(lda_matrix.T @ lda_matrix)
    normalized_eigenvalues = eigenvalues / (np.sum(eigenvalues) + eps)
    
    lidar_metric = np.exp(entropy(normalized_eigenvalues, base=np.e))
    
    return lidar_metric



def calculate_rankme_metric(embeddings, eps=1e-8, max_subset_size=10000):
    """Calculate RankMe metric with a maximum subset size to limit memory usage"""
    # Convert PyTorch tensor to NumPy array if necessary
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings)
    
    # Limit number of embeddings to prevent memory issues
    n_embed = embeddings.shape[0]
    if n_embed > max_subset_size:
        # Use random subset
        subset_ind = torch.randperm(n_embed)[:max_subset_size]
        embeddings = embeddings[subset_ind]
    
    # Compute singular values efficiently
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    cov = torch.mm(centered.T, centered) / (embeddings.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(cov)
    # Convert eigenvalues to singular values (sqrt of eigenvalues)
    singular_values = torch.sqrt(eigenvalues.clamp(min=0))
    
    # Normalize singular values
    normalized_singular_values = singular_values / (singular_values.sum() + eps)
    
    # Compute RankMe metric
    rankme_metric = torch.exp(-torch.sum(normalized_singular_values * torch.log(normalized_singular_values + eps)))
    
    return rankme_metric



def calculate_alphareq_metric(test_features):

    # The fast method
    # Step 1: Center the features by subtracting the mean
    centered_f_thetas = test_features - test_features.mean(dim=0)
    
    # Step 2: Calculate the covariance matrix
    sigma_f_theta = torch.cov(centered_f_thetas.T) # exactly like equation in section 2.1 of paper because the features have been centered

    # Step 3: Calculate the eigenvalues of the empirical covariance matrix
    eigenvalues = torch.linalg.eigvals(sigma_f_theta)

    # Temporary fix for complex eigenvalues
    eigenvalues = torch.abs(eigenvalues)
    
    # Sort the eigenvalues in descending order
    eigvals_sorted,_ = torch.sort(eigenvalues, descending = True) # this is \lambda_j \in [\lambda_min, \lambda_max], and we've sorted them in descending order to fit

    # Step 4: Calculate the decay coefficient
    x = np.log(np.arange(1, len(eigvals_sorted)+1))
    y = np.log(eigvals_sorted.numpy())
    # remove untenable bits
    mask = np.isfinite(y)
    x = x[mask]
    y = y[mask]

    # linear regression
    slope, intercept = np.polyfit(x,y,1)
    # Calculate the R-squared value
    y_pred = slope * x + intercept
    ss_tot = np.sum((y - np.mean(y))**2)
    ss_res = np.sum((y - y_pred)**2)
    r_squared = 1 - (ss_res / ss_tot)
    # Calculate the standard error of the estimate
    n = len(x)
    std_error = np.sqrt(ss_res / (n - 2))

    # The negative of the slope is alpha
    alpha = -slope

    return {
        'alpha': alpha,
        'r_squared': r_squared,
        'std_error': std_error,
        'intercept': intercept
    }





