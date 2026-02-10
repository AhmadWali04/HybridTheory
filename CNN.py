"""
CNN.py - Neural Network-based Color Transfer System

This module implements a deep neural network that learns intelligent color mappings
from source image colors to a target color palette. It's designed to integrate with
imageTriangulation.py and colour.py for end-to-end color stylization.

Key Components:
- ColorTransferNet: Main neural network for color mapping
- PaletteUsageLoss: Ensures all palette colors are used (prevents mode collapse)
- Training pipeline with visualization
- Application functions for triangulated image coloring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans', 'bitstream vera sans', 'sans-serif']
plt.rcParams['font.family'] = 'sans-serif'
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import time
from datetime import timedelta

try:
    import plotly.graph_objects as go
    import plotly.subplots as sp
except ImportError:
    go = None
    sp = None

# Import our own modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import colour
import tensorboard_feedback
from PIL import Image
import glob
import re


# ============================================================================
# FEEDBACK UTILITIES
# ============================================================================

def load_previous_feedback(feedback_dir='feedback_data', model_name=None, template_name=None):
    """
    Load and process all previous feedback sessions for a specific model.

    Args:
        feedback_dir: Directory containing feedback JSON files
        model_name: Model name to load feedback for (e.g., 'spiderman_hybridTheory')
                   If None, loads from all subdirectories (backward compatibility)
        template_name: Template image basename for nested directory lookup

    Returns:
        List of tuples: [(scores, weight), ...] where:
            - scores: [freq0..freq9, place0..place9]
            - weight: recency weight (more recent = higher weight)

    Example:
        feedback_list = load_previous_feedback('feedback_data', 'spiderman_hybridTheory', 'spiderman')
        # Returns: [([5,7,3,9,8,2,6,4,0,0,5,7,3,9,8,2,6,4,0,0], 1.0),
        #           ([8,3,7,2,6,5,1,9,4,7,8,3,7,2,6,5,1,9,4,7], 0.75), ...]
    """
    all_feedback = tensorboard_feedback.load_all_feedback(feedback_dir, model_name=model_name,
                                                          template_name=template_name)

    if not all_feedback:
        return []

    feedback_list = []

    for i, session in enumerate(all_feedback):
        freq_scores = session['frequency_scores']
        place_scores = session['placement_scores']
        scores = freq_scores + place_scores

        # Calculate recency weight: more recent = higher weight
        # Linear weight from 0.3 (oldest) to 1.0 (newest)
        recency = 0.3 + (i / max(1, len(all_feedback) - 1)) * 0.7

        feedback_list.append((scores, recency))

    return feedback_list


def compute_feedback_weighted_loss(model, source_pixels, target_palette,
                                   feedback_list, device='cpu', n_samples=1000):
    """
    Compute loss incorporating user feedback about color usage.

    FEEDBACK SCORING SYSTEM (NEW):
    - 0 = Never Used (color not appearing, must be added)
    - 1 = Too Little (appearing but rarely)
    - 5 = Perfect (ideal usage)
    - 9 = Too Much (overused, reduce)

    WEIGHT CALCULATION LOGIC:
    The penalty weight follows a symmetric V-shaped curve centered at 5:
    - Scores below 5: Linear from 2.0 (score 0) to 0.0 (score 5)
      Formula: weight = 2.0 - (score / 5.0) * 2.0
    - Scores above 5: Linear from 0.0 (score 5) to 2.0 (score 9)
      Formula: weight = ((score - 5) / 4.0) * 2.0

    Both under-use (0→2.0) and over-use (9→2.0) are penalized equally,
    so the model is statistically driven toward a score of 5.

    FREQUENCY LOSS (weight: 0.7):
    Penalizes not using colors based on feedback.

    PLACEMENT LOSS (weight: 0.3):
    Penalizes using colors in perceptually wrong locations.

    Args:
        model: ColorTransferNet instance
        source_pixels: Source pixels (N, 3), normalized [0,1]
        target_palette: Target palette (10, 3), normalized [0,1]
        feedback_list: List of (scores, recency_weight) tuples
        device: 'cpu' or 'cuda'
        n_samples: Number of samples for loss calculation

    Returns:
        Scalar tensor with feedback-weighted loss
    """
    if not feedback_list:
        # No feedback yet, return 0
        return torch.tensor(0.0, device=device)
    
    # Average feedback scores, weighted by recency
    total_weight = sum(weight for _, weight in feedback_list)
    avg_freq_scores = np.zeros(10)
    avg_place_scores = np.zeros(10)
    
    for scores, weight in feedback_list:
        freq_scores = np.array(scores[:10], dtype=np.float32)
        place_scores = np.array(scores[10:20], dtype=np.float32)
        avg_freq_scores += freq_scores * weight
        avg_place_scores += place_scores * weight
    
    avg_freq_scores /= total_weight
    avg_place_scores /= total_weight
    
    # Sample pixels and get network output
    if len(source_pixels) > n_samples:
        indices = np.random.choice(len(source_pixels), n_samples, replace=False)
        sampled_pixels = source_pixels[indices]
    else:
        sampled_pixels = source_pixels
    
    model.eval()
    with torch.no_grad():
        network_output = model(sampled_pixels)  # (N, 3)
    
    # Denormalize to [0, 255] for comparison
    output_colors_255 = network_output * 255.0
    palette_255 = target_palette * 255.0
    
    # ========== FREQUENCY LOSS ==========
    # For each color in palette, check if network output includes it
    frequency_loss = 0.0
    
    for color_idx in range(10):
        target_color = palette_255[color_idx:color_idx+1]  # (1, 3)
        
        # Find closest network output to this palette color
        distances = torch.norm(output_colors_255 - target_color, dim=1)  # (N,)
        closest_dist = torch.min(distances)
        
        # Frequency feedback: 0 = never used, 1 = too little, 5 = perfect, 9 = too much
        freq_score = np.clip(avg_freq_scores[color_idx], 0, 9)  # Ensure valid range

        # Symmetric V-shaped curve centered at 5 (perfect)
        if freq_score < 5:
            # Under-used colors: linear penalty (0→2.0, 5→0.0)
            freq_penalty_weight = 2.0 - (freq_score / 5.0) * 2.0
        else:
            # Over-used colors: equal penalty (5→0.0, 9→2.0)
            freq_penalty_weight = ((freq_score - 5) / 4.0) * 2.0
        
        # Penalize if network isn't using this color
        frequency_loss += freq_penalty_weight * closest_dist
    
    frequency_loss /= 10.0
    
    # ========== PLACEMENT LOSS ==========
    # Check if colors are used in appropriate places based on source
    placement_loss = 0.0
    
    for color_idx in range(10):
        # Placement feedback: same V-shaped curve as frequency
        placement_score = np.clip(avg_place_scores[color_idx], 0, 9)

        if placement_score == 0:
            # User said never used, skip placement evaluation
            continue

        # Symmetric V-shaped curve: 1 = wrong places, 5 = good, 9 = excellent
        if placement_score < 5:
            place_penalty_weight = 2.0 - (placement_score / 5.0) * 2.0
        else:
            place_penalty_weight = ((placement_score - 5) / 4.0) * 2.0
        
        # For each output pixel, check if it's close to palette
        target_color = palette_255[color_idx:color_idx+1]
        distances = torch.norm(output_colors_255 - target_color, dim=1)
        
        # Smooth distance: if already using this color, penalize inconsistency
        # This encourages coherent placement
        placement_loss += place_penalty_weight * torch.std(distances)
    
    placement_loss /= 10.0
    
    # ========== COMBINED FEEDBACK LOSS ==========
    # Frequency has more weight (user's main concern)
    feedback_weighted_loss = 0.7 * frequency_loss + 0.3 * placement_loss
    
    return feedback_weighted_loss


def fine_tune_with_feedback(model, source_pixels, target_palette,
                            source_pixels_lab, target_palette_lab,
                            epochs=150, batch_size=512, lr=0.0005,
                            device='cpu', feedback_dir='feedback_data',
                            log_dir='runs/fine_tuning', model_name=None,
                            template_name=None):
    """
    Fine-tune model with feedback from previous training sessions.

    Args:
        model: Previously trained ColorTransferNet
        source_pixels: Source pixels (N, 3), normalized [0,1]
        target_palette: Target palette (10, 3), normalized [0,1]
        source_pixels_lab: Source in LAB space
        target_palette_lab: Target in LAB space
        epochs: Number of fine-tuning epochs (default: 150)
        batch_size: Batch size (default: 512)
        lr: Learning rate (default: 0.0005, lower than initial training)
        device: 'cpu' or 'cuda'
        feedback_dir: Directory containing feedback JSON files
        log_dir: TensorBoard log directory for fine-tuning visualization
        model_name: Model name for loading model-specific feedback (e.g., 'spiderman_hybridTheory')
        template_name: Template image basename for nested directory lookup

    Returns:
        Tuple: (fine_tuned_model, loss_history)
    """
    from torch.utils.tensorboard import SummaryWriter

    # Load previous feedback from JSON (model-specific if model_name provided)
    feedback_list = load_previous_feedback(feedback_dir, model_name=model_name,
                                           template_name=template_name)

    if not feedback_list:
        print("\nNo previous feedback found. Skipping feedback fine-tuning.")
        return model, {'feedback': [], 'total': []}

    # Create TensorBoard writer for fine-tuning
    writer = SummaryWriter(log_dir)

    print("\n" + "="*70)
    print("FINE-TUNING WITH FEEDBACK (TensorBoard Logging)")
    print("="*70)
    print(f"Found {len(feedback_list)} previous sessions")
    print(f"Recent sessions weighted more heavily")
    print(f"Fine-tuning for {epochs} epochs with learning rate {lr}")
    print(f"TensorBoard: tensorboard --logdir={log_dir}")
    print(f"URL: http://127.0.0.1:6006")
    print("="*70)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_history = {'feedback': [], 'total': [], 'usage': [], 'nearest': [], 'entropy': []}

    # Initialize loss functions for auxiliary losses
    usage_loss_fn = PaletteUsageLoss().to(device)
    nearest_color_loss_fn = NearestColorDistanceLoss().to(device)
    entropy_loss_fn = PaletteEntropyLoss().to(device)

    print_progress_bar.start_time = time.time()

    for epoch in range(epochs):
        # Sample batch
        if len(source_pixels) > batch_size:
            indices = torch.randperm(len(source_pixels))[:batch_size]
            batch = source_pixels[indices].to(device)
        else:
            batch = source_pixels.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(batch)

        # Compute feedback loss
        feedback_loss = compute_feedback_weighted_loss(
            model, source_pixels, target_palette,
            feedback_list, device=device, n_samples=1000
        )

        # Also compute standard losses for reference
        usage_loss = usage_loss_fn(model, batch)
        nearest_loss = nearest_color_loss_fn(output, target_palette)
        entropy_loss = entropy_loss_fn(model, batch)

        # Total loss (weighted combination)
        total_loss = feedback_loss + 0.3 * usage_loss + 0.2 * nearest_loss + 0.2 * entropy_loss

        # Backward pass
        if total_loss.item() > 0:
            total_loss.backward()
            optimizer.step()

        # Record losses
        loss_history['feedback'].append(feedback_loss.item())
        loss_history['total'].append(total_loss.item())
        loss_history['usage'].append(usage_loss.item())
        loss_history['nearest'].append(nearest_loss.item())
        loss_history['entropy'].append(entropy_loss.item())

        # Log to TensorBoard
        writer.add_scalar('Fine_Tuning/Feedback_Loss', feedback_loss.item(), epoch)
        writer.add_scalar('Fine_Tuning/Usage_Loss', usage_loss.item(), epoch)
        writer.add_scalar('Fine_Tuning/Nearest_Color_Loss', nearest_loss.item(), epoch)
        writer.add_scalar('Fine_Tuning/Entropy_Loss', entropy_loss.item(), epoch)
        writer.add_scalar('Fine_Tuning/Total_Loss', total_loss.item(), epoch)

        # Log sample outputs every 50 epochs
        if epoch % 50 == 0 and epoch > 0:
            with torch.no_grad():
                sample_indices = torch.randperm(len(source_pixels))[:100]
                sample_input = source_pixels[sample_indices].to(device)
                sample_output = model(sample_input)

                # Log color distribution
                for i in range(3):
                    writer.add_histogram(f'Fine_Tuning/Output_Channel_{i}',
                                       sample_output[:, i].cpu(), epoch)

        # Progress bar
        print_progress_bar(epoch + 1, epochs,
                          prefix=f'Fine-tuning (Total Loss: {total_loss.item():.6f})',
                          length=40)

    writer.close()
    print("\n\n✓ Fine-tuning complete!")

    return model, loss_history


def generate_hex_prefix():
    """
    Generate a hex prefix for saved images.
    
    Returns:
        str: 3-character hex prefix (000-FFF, representing 0-4095)
    """
    # Count existing images in trainingData folder
    if os.path.exists('trainingData'):
        existing = len(glob.glob('trainingData/*.png'))
    else:
        existing = 0
    
    # Convert to hex (0-FFF range)
    hex_value = hex(existing % 4096)[2:].upper().zfill(3)
    return hex_value


def save_feedback_image(image, palette_rgb, scores, training_data_dir='trainingData'):
    """
    DEPRECATED: Save colored image with feedback scores encoded in filename.

    This function is deprecated. Feedback is now saved as JSON files by
    tensorboard_feedback.py. This function remains for backward compatibility
    but should not be used in new code.

    Args:
        image: PIL Image object (the colored triangulation)
        palette_rgb: numpy array (10, 3) - palette colors
        scores: list of 20 integers [freq0..freq9, place0..place9]
        training_data_dir: Directory to save to

    Returns:
        str: Full path to saved image
    """
    print("\n⚠️  WARNING: save_feedback_image() is deprecated!")
    print("   Feedback is now saved as JSON by tensorboard_feedback.py")
    print("   This function will be removed in a future version.")

    # Create directory if needed
    os.makedirs(training_data_dir, exist_ok=True)

    # Generate hex prefix and filename
    hex_prefix = generate_hex_prefix()
    score_suffix = tensorboard_feedback.scores_to_filename_suffix(scores)

    filename = f"{hex_prefix}_{score_suffix}.png"
    filepath = os.path.join(training_data_dir, filename)

    # Save image
    image.save(filepath)

    print(f"\n✓ Image saved (legacy format): {filename}")
    print(f"   Consider using TensorBoard feedback system instead")

    return filepath


def print_progress_bar(iteration, total, prefix='', length=50, decimals=1):
    """
    Create terminal progress bar with time estimation.
    
    Args:
        iteration: Current iteration (0 to total)
        total: Total iterations
        prefix: Prefix string
        length: Length of progress bar
        decimals: Decimals for percentage
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '░' * (length - filled_length)
    
    # Calculate time remaining
    if iteration > 0:
        elapsed = time.time() - print_progress_bar.start_time
        rate = elapsed / iteration
        remaining = rate * (total - iteration)
        eta_str = str(timedelta(seconds=int(remaining)))
    else:
        eta_str = "calculating..."
    
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% ETA: {eta_str}')
    sys.stdout.flush()


class ColorTransferNet(nn.Module):
    """
    Neural network that learns to map source image colors to target palette colors.

    Architecture:
    - Input: RGB color [3 values, 0-1 normalized]
    - Hidden layers: 5 layers with 256 neurons each
    - Activation: ReLU after each hidden layer
    - Normalization: BatchNorm1d after each ReLU
    - Output: Logits over palette colors, softmax-blended with temperature

    The network outputs logits over the target palette and uses temperature-scaled
    softmax to select/blend palette colors. This constrains outputs to actual palette
    colors and prevents grey-averaging.

    Temperature controls color selection sharpness:
    - Low (0.1-0.5): Sharp picks, more colorful, uses more palette colors
    - High (1.0+): Soft blending, smoother but less saturated
    """

    def __init__(self, num_palette_colors=10, n_layers=5, hidden_dim=256, temperature=0.5):
        """
        Initialize the ColorTransferNet.

        Args:
            num_palette_colors: Number of target palette colors (default: 10)
            n_layers: Number of hidden layers (default: 5)
            hidden_dim: Number of neurons in each hidden layer (default: 256)
            temperature: Softmax temperature for palette selection (default: 0.5)
        """
        super(ColorTransferNet, self).__init__()

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.num_palette_colors = num_palette_colors
        self.temperature = temperature

        # Palette stored as buffer (saved with model, not a trainable parameter)
        self.register_buffer('palette', None)

        # Build network layers dynamically
        layers = []

        # First layer: 3 (RGB) -> hidden_dim
        layers.append(nn.Linear(3, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers: hidden_dim -> hidden_dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))

        # Output layer: logits over palette colors (no Sigmoid)
        layers.append(nn.Linear(hidden_dim, num_palette_colors))

        self.network = nn.Sequential(*layers)

    def set_palette(self, palette_tensor):
        """
        Set the target palette colors. Must be called before forward().

        Args:
            palette_tensor: Tensor of shape (num_palette_colors, 3) with RGB values in [0, 1]
        """
        self.register_buffer('palette', palette_tensor)

    def forward(self, colors):
        """
        Forward pass through the network.

        Args:
            colors: Input tensor of shape (batch_size, 3) with values in [0, 1]

        Returns:
            output: Tensor of shape (batch_size, 3) with values in [0, 1]
                    Each output is a temperature-weighted blend of palette colors.
        """
        logits = self.network(colors)                              # (batch, num_palette_colors)
        weights = F.softmax(logits / self.temperature, dim=-1)     # (batch, num_palette_colors)
        output = torch.matmul(weights, self.palette)               # (batch, 3)
        return output


class PaletteUsageLoss(nn.Module):
    """
    Ensures all palette colors are used roughly equally (prevents mode collapse).

    Operates on the softmax assignment weights directly, which makes it
    fully differentiable unlike the old histogram-based approach.

    Computes the average palette weight across the batch and penalizes
    deviation from uniform usage. Uses a fixed temperature of 1.0 internally
    to ensure gradients flow even when the model's temperature is very low.
    """

    def __init__(self):
        super(PaletteUsageLoss, self).__init__()

    def forward(self, model, batch):
        """
        Compute palette usage loss.

        Args:
            model: ColorTransferNet with network, temperature, and palette attributes
            batch: Input tensor of shape (N, 3) with values in [0, 1]

        Returns:
            loss: Scalar tensor penalizing unbalanced palette usage
        """
        logits = model.network(batch)  # (N, num_palette_colors)

        # Use a fixed temperature of 1.0 for the usage loss to ensure
        # gradients flow well even when the model's actual temperature is very low.
        # The model's temperature controls output sharpness; this temperature
        # controls how strongly the diversity signal propagates.
        weights = F.softmax(logits / 1.0, dim=-1)  # (N, num_palette_colors)

        # Average usage of each palette color across the batch
        avg_usage = weights.mean(dim=0)  # (num_palette_colors,)

        # Target: uniform usage
        target_usage = torch.ones_like(avg_usage) / len(avg_usage)

        # MSE from uniform encourages all palette colors to be used
        loss = F.mse_loss(avg_usage, target_usage)

        return loss


class NearestColorDistanceLoss(nn.Module):
    """
    Ensures each output color is close to at least one target palette color.
    
    This encourages the network to map colors to the actual palette,
    rather than outputting arbitrary intermediate colors.
    """
    
    def __init__(self):
        super(NearestColorDistanceLoss, self).__init__()
    
    def forward(self, output_colors, target_palette):
        """
        Compute nearest color distance loss.
        
        Args:
            output_colors: Tensor of shape (N, 3) - network output
            target_palette: Tensor of shape (M, 3) - target colors
        
        Returns:
            loss: Scalar tensor
        """
        # Expand dimensions for broadcasting
        # output_colors: (N, 1, 3)
        # target_palette: (1, M, 3)
        output_expanded = output_colors.unsqueeze(1)  # (N, 1, 3)
        target_expanded = target_palette.unsqueeze(0)  # (1, M, 3)
        
        # Compute pairwise distances
        distances = torch.norm(output_expanded - target_expanded, dim=2)  # (N, M)
        
        # Minimum distance for each output color to any target color
        min_distances = distances.min(dim=1)[0]  # (N,)
        
        # Mean of minimum distances
        loss = min_distances.mean()
        
        return loss


class PaletteEntropyLoss(nn.Module):
    """
    Penalizes high entropy in palette assignment weights.

    Encourages the network to make decisive color picks rather than
    blending all palette colors equally (which produces muddy averages).
    Lower entropy = sharper palette assignment = more colorful output.
    """

    def __init__(self):
        super(PaletteEntropyLoss, self).__init__()

    def forward(self, model, batch):
        """
        Compute entropy of palette assignment weights.

        Args:
            model: ColorTransferNet with temperature and network attributes
            batch: Input tensor of shape (N, 3) with values in [0, 1]

        Returns:
            loss: Scalar tensor (mean entropy across batch)
        """
        logits = model.network(batch)
        weights = F.softmax(logits / model.temperature, dim=-1)
        entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1)
        return entropy.mean()


def compute_smoothness_loss(model, source_pixels, n_samples=1000, noise_scale=0.01, device='cpu'):
    """
    Regularization to ensure smooth color transitions.
    
    Similar input colors should map to similar output colors.
    This prevents jarring discontinuities in the color mapping and makes
    the network learn smoother transformations.
    
    Args:
        model: ColorTransferNet
        source_pixels: Input pixel tensor of shape (N, 3)
        n_samples: Number of samples to test (default: 1000)
        noise_scale: Amount of Gaussian noise to add (default: 0.01)
        device: 'cpu' or 'cuda'
    
    Returns:
        smoothness_loss: Scalar tensor
    """
    # Sample random colors from source
    if len(source_pixels) > n_samples:
        indices = torch.randperm(len(source_pixels))[:n_samples]
        sampled_colors = source_pixels[indices]
    else:
        sampled_colors = source_pixels
    
    # Add small noise to create perturbed versions
    noise = torch.randn_like(sampled_colors) * noise_scale
    perturbed_colors = torch.clamp(sampled_colors + noise, 0, 1)
    
    # Get outputs for both
    with torch.no_grad():
        output1 = model(sampled_colors)
        output2 = model(perturbed_colors)
    
    # Compute L2 distance between outputs
    output_diff = torch.norm(output1 - output2, dim=1)
    
    # Small input changes should produce small output changes
    # This is a soft constraint, not a hard requirement
    smoothness_loss = output_diff.mean()
    
    return smoothness_loss


def prepare_training_data(source_image_path, target_image_path, 
                         num_clusters=25, num_distinct=10, 
                         use_lab=True, device='cpu'):
    """
    Prepare data for training the CNN.
    
    This function:
    1. Loads both source and target images
    2. Extracts color palettes using K-Means clustering
    3. Selects distinct colors using greedy max-min distance
    4. Converts all data to PyTorch tensors with proper normalization
    
    Args:
        source_image_path: Path to image to be triangulated (e.g., spiderman.jpg)
        target_image_path: Path to style image (e.g., vegetables.jpg)
        num_clusters: Number of K-Means clusters for palette extraction (default: 25)
        num_distinct: Number of distinct colors to extract (default: 10)
        use_lab: If True, use LAB color space for clustering (recommended)
        device: 'cpu' or 'cuda'
    
    Returns:
        Dictionary containing:
            - 'source_pixels': All pixels from source image, tensor (N, 3), normalized [0, 1]
            - 'target_palette': Distinct colors from target image, tensor (num_distinct, 3), [0, 1]
            - 'source_pixels_lab': Source pixels in LAB space, tensor (N, 3)
            - 'target_palette_lab': Target palette in LAB space, tensor (num_distinct, 3)
            - 'target_percentages': Color distribution percentages
    
    Example:
        data = prepare_training_data('source.jpg', 'target.jpg', 
                                    num_clusters=25, num_distinct=10)
        source_pixels = data['source_pixels'].to(device)
        target_palette = data['target_palette'].to(device)
    """
    print("\n=== Preparing Training Data ===")
    
    # Load source image pixels
    print(f"Loading source image: {source_image_path}")
    _, source_pixels_rgb = colour.load_image_pixels(source_image_path)
    source_pixels_lab = colour.convert_rgb_pixels_to_lab(source_pixels_rgb)
    
    # Load target image and extract palette
    print(f"Loading target image: {target_image_path}")
    _, target_pixels_rgb = colour.load_image_pixels(target_image_path)
    target_pixels_lab = colour.convert_rgb_pixels_to_lab(target_pixels_rgb)
    
    # Extract palette from target image
    if use_lab:
        print(f"Running K-Means clustering in LAB space ({num_clusters} clusters)...")
        _, centers_lab, _, percentages = colour.run_kmeans_lab(target_pixels_lab, num_clusters)
        
        # Convert LAB centers back to RGB for display
        centers_rgb = []
        for lab in centers_lab:
            lab_img = np.array([[lab]], dtype=np.uint8)
            rgb_img = torch.tensor(lab)
            # Use cv2 to convert LAB to RGB
            import cv2
            rgb_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)
            centers_rgb.append(rgb_img[0, 0])
        centers_rgb = np.array(centers_rgb)
        
        print(f"Selecting {num_distinct} distinct colors using greedy max-min distance...")
        palette_lab, palette_rgb, _ = colour.select_distinct_colors_lab(
            centers_lab, centers_rgb, num_to_select=num_distinct
        )
        percentages_selected = percentages[_]
    else:
        print(f"Running K-Means clustering in RGB space ({num_clusters} clusters)...")
        _, centers_rgb, _, percentages = colour.run_kmeans(target_pixels_rgb, num_clusters)
        
        print(f"Selecting {num_distinct} distinct colors using greedy max-min distance...")
        palette_rgb, _ = colour.select_distinct_colors(centers_rgb, num_to_select=num_distinct)
        percentages_selected = percentages[_]
        
        # Convert palette to LAB for reference
        palette_rgb_uint8 = palette_rgb.astype(np.uint8).reshape(1, -1, 3)
        import cv2
        palette_lab = cv2.cvtColor(palette_rgb_uint8, cv2.COLOR_RGB2LAB).reshape(-1, 3)
    
    print(f"\nTarget palette extracted:")
    for i, (rgb, pct) in enumerate(zip(palette_rgb, percentages_selected)):
        print(f"  Color {i+1}: RGB{tuple(rgb)} - {pct:.1f}%")
    
    # Convert to PyTorch tensors and normalize to [0, 1]
    source_pixels_tensor = torch.from_numpy(source_pixels_rgb.astype(np.float32) / 255.0)
    target_palette_tensor = torch.from_numpy(palette_rgb.astype(np.float32) / 255.0)
    source_pixels_lab_tensor = torch.from_numpy(source_pixels_lab.astype(np.float32))
    target_palette_lab_tensor = torch.from_numpy(palette_lab.astype(np.float32))
    
    # Move to device
    source_pixels_tensor = source_pixels_tensor.to(device)
    target_palette_tensor = target_palette_tensor.to(device)
    source_pixels_lab_tensor = source_pixels_lab_tensor.to(device)
    target_palette_lab_tensor = target_palette_lab_tensor.to(device)
    
    return {
        'source_pixels': source_pixels_tensor,
        'target_palette': target_palette_tensor,
        'source_pixels_lab': source_pixels_lab_tensor,
        'target_palette_lab': target_palette_lab_tensor,
        'target_percentages': percentages_selected,
        'palette_rgb': palette_rgb,  # Keep original numpy for reference
    }


def train_color_transfer_network(source_pixels, target_palette,
                                source_pixels_lab, target_palette_lab,
                                epochs=1000, batch_size=512, lr=0.001,
                                temperature=0.5,
                                device='cpu', save_progress=True):
    """
    Train the ColorTransferNet to map source colors to target palette.

    Training Process:
    1. Initialize ColorTransferNet with palette-assignment architecture
    2. For each epoch:
       a. Sample random batch of source pixels
       b. Forward pass through network (logits -> softmax/temp -> palette blend)
       c. Compute composite loss:
          - PaletteUsageLoss: all palette colors used equally (weight: 1.0)
          - NearestColorDistanceLoss: each output to nearest target color (weight: 0.2)
          - SmoothnessLoss: smooth transformations (weight: 0.05)
          - PaletteEntropyLoss: decisive palette picks (weight: 0.3)
       d. Backward pass and optimize
       e. Log progress every 100 epochs
    3. Return trained model and loss history

    Args:
        source_pixels: Tensor (N, 3) - source image pixels, normalized [0, 1]
        target_palette: Tensor (M, 3) - target palette colors, normalized [0, 1]
        source_pixels_lab: Tensor (N, 3) - source in LAB space
        target_palette_lab: Tensor (M, 3) - target in LAB space
        epochs: Number of training epochs (default: 1000)
        batch_size: Batch size for training (default: 512)
        lr: Learning rate (default: 0.001)
        temperature: Softmax temperature for palette selection (default: 0.5)
        device: 'cpu' or 'cuda'
        save_progress: Whether to save progress plots

    Returns:
        Tuple containing:
            - trained_model: Trained ColorTransferNet
            - loss_history: Dict with loss values per epoch
    """
    num_palette_colors = target_palette.shape[0]

    print("\n=== Training Color Transfer Network ===")
    print(f"Device: {device}")
    print(f"Network architecture: 5 layers, 256 hidden units")
    print(f"Palette colors: {num_palette_colors}, Temperature: {temperature}")
    print(f"Training epochs: {epochs}, batch size: {batch_size}, learning rate: {lr}")

    # Initialize model with palette-assignment architecture
    model = ColorTransferNet(
        num_palette_colors=num_palette_colors,
        n_layers=5, hidden_dim=256,
        temperature=temperature
    ).to(device)

    # Set the target palette (stored as buffer, saved with model)
    model.set_palette(target_palette)

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Initialize loss functions
    usage_loss_fn = PaletteUsageLoss().to(device)
    nearest_color_loss_fn = NearestColorDistanceLoss().to(device)
    entropy_loss_fn = PaletteEntropyLoss().to(device)

    # Loss weights
    w_usage = 1.0
    w_nearest = 0.2
    w_smoothness = 0.05
    w_entropy = 0.3

    # Training loop with progress bar
    loss_history = {
        'total': [],
        'usage': [],
        'nearest_color': [],
        'smoothness': [],
        'entropy': []
    }

    # Initialize progress bar timer
    print_progress_bar.start_time = time.time()

    for epoch in range(epochs):
        # Sample random batch
        if len(source_pixels) > batch_size:
            indices = torch.randperm(len(source_pixels))[:batch_size]
            batch = source_pixels[indices]
        else:
            batch = source_pixels

        # Forward pass
        output = model(batch)

        # Compute losses
        usage_loss = usage_loss_fn(model, batch)
        nearest_color_loss = nearest_color_loss_fn(output, target_palette)
        smoothness_loss = compute_smoothness_loss(
            model, source_pixels, n_samples=1000,
            noise_scale=0.01, device=device
        )
        entropy_loss = entropy_loss_fn(model, batch)

        # Composite loss
        total_loss = (w_usage * usage_loss +
                     w_nearest * nearest_color_loss +
                     w_smoothness * smoothness_loss +
                     w_entropy * entropy_loss)

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Record losses
        loss_history['total'].append(total_loss.item())
        loss_history['usage'].append(usage_loss.item())
        loss_history['nearest_color'].append(nearest_color_loss.item())
        loss_history['smoothness'].append(smoothness_loss.item())
        loss_history['entropy'].append(entropy_loss.item())

        # Update progress bar
        print_progress_bar(epoch + 1, epochs,
                          prefix=f'Training (Loss: {total_loss.item():.6f})',
                          length=40)

        # Log detailed progress every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"\n  └─ Epoch {epoch+1}/{epochs} | "
                  f"Total: {total_loss.item():.6f} | "
                  f"Usage: {usage_loss.item():.6f} | "
                  f"Nearest: {nearest_color_loss.item():.6f} | "
                  f"Entropy: {entropy_loss.item():.6f}")
            print_progress_bar.start_time = time.time() - (time.time() - print_progress_bar.start_time)

    print("\n\n✓ Training complete!")

    return model, loss_history


def visualize_training_progress(model, source_pixels, target_palette, 
                                loss_history, device='cpu', save_path=None):
    """
    Visualize how the network's color mapping improves during training.

    Creates 4 separate interactive Plotly figures:
    1. Network output color distribution vs target (3D scatter with targets)
    2. Total loss over training (line plot)
    3. Component losses (line plot with usage, nearest-color, smoothness)
    4. Sample color mappings in 3D RGB space (scatter with arrows)

    Uses Plotly for interactive visualization matching colour.py style.

    Args:
        model: Trained ColorTransferNet
        source_pixels: Source pixel tensor
        target_palette: Target palette tensor
        loss_history: Dict with loss values from training
        device: 'cpu' or 'cuda'
        save_path: Optional path to save visualization (not used with Plotly)
    """
    if go is None:
        raise RuntimeError("Plotly is not installed. Install it or use: pip install plotly")
    
    print("\n=== Creating Training Visualizations with Plotly (4 separate windows) ===")
    
    # Get model output
    model.eval()
    with torch.no_grad():
        # Downsample source for visualization
        if len(source_pixels) > 5000:
            indices = torch.randperm(len(source_pixels))[:5000]
            source_sample = source_pixels[indices]
            output_sample = model(source_sample)
        else:
            source_sample = source_pixels
            output_sample = model(source_sample)
    
    # Convert to numpy and denormalize
    source_np = source_sample.cpu().numpy() * 255
    output_np = output_sample.cpu().numpy() * 255
    target_np = target_palette.cpu().numpy() * 255
    
    # Normalize for color mapping
    output_colors = output_np / 255.0
    target_colors = target_np / 255.0
    
    # Figure 1: Network output vs target
    fig3 = go.Figure(
        data=[
            go.Scatter3d(
                x=output_np[:, 0],
                y=output_np[:, 1],
                z=output_np[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=output_colors,
                    opacity=0.5
                ),
                name='Network Output'
            ),
            go.Scatter3d(
                x=target_np[:, 0],
                y=target_np[:, 1],
                z=target_np[:, 2],
                mode='markers',
                marker=dict(
                    size=15,
                    color=target_colors,
                    line=dict(color='black', width=2),
                    opacity=0.9
                ),
                text=[f"RGB({int(t[0])}, {int(t[1])}, {int(t[2])})" for t in target_np],
                hoverinfo='text',
                name='Target Palette'
            )
        ]
    )
    fig3.update_layout(
        title='Network Output vs Target Palette',
        scene=dict(
            xaxis=dict(title='R', range=[0, 255]),
            yaxis=dict(title='G', range=[0, 255]),
            zaxis=dict(title='B', range=[0, 255])
        ),
        width=900,
        height=800,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig3.show()
    
    # Figure 4: Total loss
    fig4 = go.Figure()
    fig4.add_trace(
        go.Scatter(
            x=list(range(len(loss_history['total']))),
            y=loss_history['total'],
            mode='lines',
            name='Total Loss',
            line=dict(color='#1f77b4', width=3),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.2)'
        )
    )
    fig4.update_layout(
        title='Total Loss Over Training',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        hovermode='x unified',
        width=900,
        height=600,
        margin=dict(l=60, r=40, b=60, t=40)
    )
    fig4.show()
    
    # Figure 5: Component losses
    fig5 = go.Figure()
    fig5.add_trace(
        go.Scatter(
            x=list(range(len(loss_history['usage']))),
            y=loss_history['usage'],
            mode='lines',
            name='Usage Loss',
            line=dict(width=2.5),
            opacity=0.8
        )
    )
    fig5.add_trace(
        go.Scatter(
            x=list(range(len(loss_history['nearest_color']))),
            y=loss_history['nearest_color'],
            mode='lines',
            name='Nearest Color Loss',
            line=dict(width=2.5),
            opacity=0.8
        )
    )
    fig5.add_trace(
        go.Scatter(
            x=list(range(len(loss_history['smoothness']))),
            y=loss_history['smoothness'],
            mode='lines',
            name='Smoothness Loss',
            line=dict(width=2.5),
            opacity=0.8
        )
    )
    fig5.update_layout(
        title='Component Losses During Training',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        hovermode='x unified',
        width=900,
        height=600,
        margin=dict(l=60, r=40, b=60, t=40)
    )
    fig5.show()
    
    # Figure 4: Sample color mappings in 3D RGB space
    fig6 = go.Figure()

    # Sample a few mappings to show
    sample_indices = np.random.choice(len(source_np), min(10, len(source_np)), replace=False)

    # Plot arrows as 3D lines (source -> output)
    for idx in sample_indices:
        src_color = source_np[idx]
        out_color = output_np[idx]

        src_color_str = f'rgb({int(src_color[0])}, {int(src_color[1])}, {int(src_color[2])})'
        out_color_str = f'rgb({int(out_color[0])}, {int(out_color[1])}, {int(out_color[2])})'

        # Line connecting source to output
        fig6.add_trace(
            go.Scatter3d(
                x=[src_color[0], out_color[0]],
                y=[src_color[1], out_color[1]],
                z=[src_color[2], out_color[2]],
                mode='lines+markers',
                line=dict(color='rgba(100, 100, 100, 0.5)', width=3),
                marker=dict(size=[6, 8], color=[src_color_str, out_color_str],
                           line=dict(color='black', width=1)),
                hovertext=[
                    f"Source: RGB({int(src_color[0])}, {int(src_color[1])}, {int(src_color[2])})",
                    f"Output: RGB({int(out_color[0])}, {int(out_color[1])}, {int(out_color[2])})"
                ],
                hoverinfo='text',
                showlegend=False
            )
        )

    # Plot target palette as larger 3D points
    target_color_strs = [f'rgb({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})'
                         for c in target_colors]

    fig6.add_trace(
        go.Scatter3d(
            x=target_np[:, 0],
            y=target_np[:, 1],
            z=target_np[:, 2],
            mode='markers',
            marker=dict(
                size=12,
                color=target_color_strs,
                line=dict(color='black', width=2),
                opacity=0.9
            ),
            text=[f"Target: RGB({int(t[0])}, {int(t[1])}, {int(t[2])})" for t in target_np],
            hoverinfo='text',
            name='Target Palette'
        )
    )

    fig6.update_layout(
        title='Sample Color Mappings (3D RGB Space)',
        scene=dict(
            xaxis=dict(title='R', range=[0, 255]),
            yaxis=dict(title='G', range=[0, 255]),
            zaxis=dict(title='B', range=[0, 255])
        ),
        hovermode='closest',
        width=900,
        height=800,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig6.show()

    print("\nOpened 4 interactive Plotly visualization windows!")
    print("Tip: Use the Plotly toolbar to zoom, pan, hover for details, and save figures as PNG.")


def apply_cnn_to_triangulation(model, S, triangles, image_orig, device='cpu'):
    """
    Apply trained CNN to color each triangle in the triangulation.
    
    This is the key integration point with imageTriangulation.py
    
    Process:
    1. For each triangle in triangles.simplices:
       a. Calculate centroid coordinates
       b. Get centroid color from image_orig using PIL's getpixel()
       c. Convert RGB [0-255] to normalized [0, 1] tensor
       d. Pass through model to get transformed color
       e. Convert back to [0-255] for matplotlib
       f. Store color for this triangle
    2. Create matplotlib figure with colored triangles
    3. Return figure and color list
    
    Args:
        model: Trained ColorTransferNet
        S: Vertices array from triangulation, shape (N, 2)
        triangles: Delaunay triangulation object with .simplices attribute
        image_orig: Original PIL Image
        device: 'cpu' or 'cuda'
    
    Returns:
        Dictionary containing:
            - 'figure': matplotlib figure object
            - 'triangle_colors': List of (num_triangles, 3) RGB colors [0-255]
            - 'ax': matplotlib axes object
    
    Example:
        result = apply_cnn_to_triangulation(model, S, triangles, image_orig, device='cpu')
        result['figure'].show()
    """
    print("\n=== Applying CNN to Triangulation ===")
    
    model.eval()
    triangle_colors = []
    
    # Process all triangles
    num_triangles = len(triangles.simplices)
    print(f"Processing {num_triangles} triangles...")
    
    with torch.no_grad():
        for i, triangle_idx in enumerate(triangles.simplices):
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i+1}/{num_triangles} triangles...")
            
            # Get triangle vertices
            vertices = S[triangle_idx]
            
            # Calculate centroid
            centroid = vertices.mean(axis=0)
            
            # Get color from original image at centroid
            try:
                centroid_color = image_orig.getpixel((int(centroid[0]), int(centroid[1])))
            except:
                # Handle edge cases where coordinates are out of bounds
                x = np.clip(int(centroid[0]), 0, image_orig.width - 1)
                y = np.clip(int(centroid[1]), 0, image_orig.height - 1)
                centroid_color = image_orig.getpixel((x, y))
            
            # Extract RGB (in case there's an alpha channel)
            if isinstance(centroid_color, tuple):
                if len(centroid_color) > 3:
                    centroid_rgb = centroid_color[:3]
                else:
                    centroid_rgb = centroid_color
            else:
                # Grayscale image
                centroid_rgb = (centroid_color, centroid_color, centroid_color)
            
            # Convert to normalized tensor
            color_tensor = torch.tensor([centroid_rgb[0]/255.0, centroid_rgb[1]/255.0, centroid_rgb[2]/255.0],
                                      dtype=torch.float32).unsqueeze(0).to(device)
            
            # Pass through model
            transformed_color = model(color_tensor).squeeze(0).cpu().numpy()
            
            # Convert back to [0-255] range
            transformed_rgb = (transformed_color * 255).astype(np.uint8)
            triangle_colors.append(transformed_rgb)
    
    triangle_colors = np.array(triangle_colors)
    print(f"Completed! Generated colors for all {num_triangles} triangles.")
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Use image coordinates
    
    # Plot each triangle with its CNN-generated color
    for i, triangle_idx in enumerate(triangles.simplices):
        vertices = S[triangle_idx]
        xs = vertices[:, 0]
        ys = vertices[:, 1]
        
        # Normalize color for matplotlib
        rgb_normalized = triangle_colors[i] / 255.0
        ax.fill(xs, ys, color=rgb_normalized, edgecolor='none')
    
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    return {
        'figure': fig,
        'triangle_colors': triangle_colors,
        'ax': ax
    }


def save_trained_model(model, filepath, metadata=None):
    """
    Save trained model with metadata.
    
    Args:
        model: ColorTransferNet to save
        filepath: Where to save (e.g., 'models/color_transfer_spiderman_vegetables.pth')
        metadata: Dict with info like:
            - source_image: path to source
            - target_image: path to target
            - num_clusters: K-Means clusters used
            - num_distinct: Distinct colors used
            - training_epochs: epochs trained
            - final_loss: final loss value
    
    Example:
        save_trained_model(
            model, 'models/my_model.pth',
            metadata={
                'source_image': 'spiderman.jpg',
                'target_image': 'vegetables.jpg',
                'num_clusters': 25,
                'num_distinct': 10,
                'training_epochs': 1000,
                'final_loss': 0.0234
            }
        )
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Prepare save data (palette buffer is included in state_dict automatically)
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_architecture': {
            'n_layers': model.n_layers,
            'hidden_dim': model.hidden_dim,
            'num_palette_colors': model.num_palette_colors,
            'temperature': model.temperature,
        },
        'metadata': metadata or {}
    }
    
    # Save
    torch.save(save_dict, filepath)
    print(f"Model saved to: {filepath}")


def load_trained_model(filepath, device='cpu'):
    """
    Load a previously trained model.
    
    Args:
        filepath: Path to saved model
        device: 'cpu' or 'cuda'
    
    Returns:
        Tuple containing:
            - model: Loaded ColorTransferNet
            - metadata: Dictionary with training info
    
    Example:
        model, metadata = load_trained_model('models/my_model.pth', device='cpu')
        print(f"Model trained on: {metadata['source_image']}")
    """
    # Load checkpoint
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    # Recreate model with palette-assignment architecture
    arch = checkpoint['model_architecture']
    num_colors = arch.get('num_palette_colors', 10)
    model = ColorTransferNet(
        num_palette_colors=num_colors,
        n_layers=arch['n_layers'],
        hidden_dim=arch['hidden_dim'],
        temperature=arch.get('temperature', 0.5),
    ).to(device)

    # Initialize palette buffer so load_state_dict can copy into it
    # (register_buffer with None creates a non-persistent buffer that
    #  load_state_dict cannot copy_ into)
    model.set_palette(torch.zeros(num_colors, 3, device=device))

    # Load weights (overwrites dummy palette with the real saved one)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get metadata
    metadata = checkpoint.get('metadata', {})
    
    print(f"Model loaded from: {filepath}")
    if metadata:
        print("Model metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    
    return model, metadata


def compare_methods(S, triangles, image_orig, source_path, target_path,
                   model=None, num_clusters=25, num_distinct=10, device='cpu'):
    """
    Create side-by-side comparison of different coloring methods.
    
    Creates a figure with subplots showing:
    1. Original triangulation (centroid-based coloring)
    2. Nearest color matching (simple palette mapping)
    3. CNN color transfer (neural network method)
    4. Color palettes used
    
    This helps visualize the improvement from using CNN vs simple nearest-color.
    
    Args:
        S: Vertices array
        triangles: Delaunay triangulation
        image_orig: Original PIL Image
        source_path: Path to source image
        target_path: Path to target palette image
        model: Trained ColorTransferNet (if None, will train one)
        num_clusters: K-Means clusters
        num_distinct: Distinct colors to use
        device: 'cpu' or 'cuda'
    
    Returns:
        fig: matplotlib figure with all methods
    """
    print("\n=== Creating Method Comparison ===")
    
    # Method 1: Original coloring (centroid-based)
    print("Method 1: Original centroid-based coloring...")
    fig_orig, ax_orig = plt.subplots(figsize=(10, 8))
    ax_orig.set_aspect('equal')
    ax_orig.invert_yaxis()
    
    for triangle_idx in triangles.simplices:
        vertices = S[triangle_idx]
        xs = vertices[:, 0]
        ys = vertices[:, 1]
        
        centroid = vertices.mean(axis=0)
        try:
            color = image_orig.getpixel((int(centroid[0]), int(centroid[1])))
        except:
            x = np.clip(int(centroid[0]), 0, image_orig.width - 1)
            y = np.clip(int(centroid[1]), 0, image_orig.height - 1)
            color = image_orig.getpixel((x, y))
        
        if len(color) > 3:
            color = color[:3]
        
        rgb_norm = (color[0]/255.0, color[1]/255.0, color[2]/255.0)
        ax_orig.fill(xs, ys, color=rgb_norm, edgecolor='none')
    
    ax_orig.set_axis_off()
    ax_orig.set_title('Method 1: Original Centroid Coloring', fontsize=12, fontweight='bold')
    plt.close(fig_orig)
    
    # Method 2: Nearest color (from existing code)
    print("Method 2: Nearest color matching...")
    _, pixels = colour.load_image_pixels(source_path)
    _, centers, _, _ = colour.run_kmeans(pixels, num_clusters)
    palette, _ = colour.select_distinct_colors(centers, num_to_select=num_distinct)
    
    fig_nearest, ax_nearest = plt.subplots(figsize=(10, 8))
    ax_nearest.set_aspect('equal')
    ax_nearest.invert_yaxis()
    
    for triangle_idx in triangles.simplices:
        vertices = S[triangle_idx]
        xs = vertices[:, 0]
        ys = vertices[:, 1]
        
        centroid = vertices.mean(axis=0)
        try:
            centroid_color = image_orig.getpixel((int(centroid[0]), int(centroid[1])))
        except:
            x = np.clip(int(centroid[0]), 0, image_orig.width - 1)
            y = np.clip(int(centroid[1]), 0, image_orig.height - 1)
            centroid_color = image_orig.getpixel((x, y))
        
        if len(centroid_color) > 3:
            centroid_rgb = centroid_color[:3]
        else:
            centroid_rgb = centroid_color
        
        # Find closest color in palette
        distances = np.linalg.norm(palette - np.array(centroid_rgb), axis=1)
        closest_color = palette[np.argmin(distances)]
        
        rgb_norm = (closest_color[0]/255.0, closest_color[1]/255.0, closest_color[2]/255.0)
        ax_nearest.fill(xs, ys, color=rgb_norm, edgecolor='none')
    
    ax_nearest.set_axis_off()
    ax_nearest.set_title('Method 2: Nearest Color Matching', fontsize=12, fontweight='bold')
    plt.close(fig_nearest)
    
    # Method 3: CNN color transfer
    print("Method 3: CNN color transfer...")
    if model is None:
        print("Training model...")
        data = prepare_training_data(source_path, target_path, 
                                    num_clusters, num_distinct, device=device)
        model, _ = train_color_transfer_network(
            data['source_pixels'], data['target_palette'],
            data['source_pixels_lab'], data['target_palette_lab'],
            epochs=500, batch_size=512, lr=0.001, device=device
        )
    
    result_cnn = apply_cnn_to_triangulation(model, S, triangles, image_orig, device=device)
    fig_cnn = result_cnn['figure']
    
    # Create comparison figure
    print("Creating comparison visualization...")
    fig = plt.figure(figsize=(18, 12))
    
    # Subplot 1: Original
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(np.array(fig_orig.canvas.buffer_rgba())[:, :, :3])
    ax1.set_axis_off()
    
    # Subplot 2: Nearest color
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(np.array(fig_nearest.canvas.buffer_rgba())[:, :, :3])
    ax2.set_axis_off()
    
    # Subplot 3: CNN
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(np.array(fig_cnn.canvas.buffer_rgba())[:, :, :3])
    ax3.set_axis_off()
    
    # Subplot 4: Palettes
    ax4 = fig.add_subplot(2, 2, 4)
    colour.visualize_color_palette(palette, np.ones(len(palette)) / len(palette), 'Target')
    ax4.set_axis_off()
    
    plt.tight_layout()
    plt.close(fig_orig)
    plt.close(fig_nearest)
    plt.close(fig_cnn)
    
    return fig


if __name__ == '__main__':
    # Basic testing
    print("CNN.py loaded successfully!")
    print("Run example_cnn_usage.py for complete examples.")
