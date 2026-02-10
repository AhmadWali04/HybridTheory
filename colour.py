"""
colour.py - Color Extraction and Processing Functions

This module contains functions for extracting and processing colors from images:
- Image loading and pixel extraction
- RGB to LAB color space conversion
- K-Means clustering for color extraction
- Distinct color selection using LAB-based algorithms

Visualization functions have been moved to plotting.py.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans


def load_image_pixels(image_path):
    """
    Load an image and convert it to a 2D array of RGB values.

    Args:
        image_path: Path to the image file

    Returns:
        image_rgb: Image in RGB format
        pixels: 2D array of shape (num_pixels, 3) with RGB values
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at path: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Handle alpha channel if present
    if image_rgb.shape[-1] == 4:  # RGBA
        image_rgb = image_rgb[:, :, :3]  # Keep only RGB channels

    # Validate and clean data
    pixels = image_rgb.reshape(-1, 3)
    pixels = np.clip(pixels, 0, 255).astype(np.uint8)

    # Diagnostics
    print(f"Image shape: {image_rgb.shape}")
    print(f"Pixels data shape: {pixels.shape}")
    return image_rgb, pixels


def convert_rgb_pixels_to_lab(pixels_rgb):
    """
    Convert RGB pixels to LAB color space.

    Args:
        pixels_rgb: Array of RGB pixel values (N, 3)

    Returns:
        pixels_lab: Array of LAB pixel values (N, 3)
    """
    # Reshape to image format for cv2
    h = int(np.sqrt(len(pixels_rgb)))
    w = len(pixels_rgb) // h
    remainder = len(pixels_rgb) % h
    if remainder != 0:
        # Add padding if needed
        padding = h - remainder
        pixels_rgb_padded = np.vstack([pixels_rgb, np.zeros((padding, 3), dtype=np.uint8)])
        h = int(np.sqrt(len(pixels_rgb_padded)))
        w = len(pixels_rgb_padded) // h
        img_rgb = pixels_rgb_padded[:h*w].reshape(h, w, 3)
    else:
        img_rgb = pixels_rgb.reshape(h, w, 3)

    # Convert to LAB
    img_lab = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB)
    pixels_lab = img_lab.reshape(-1, 3)

    # Remove padding if we added any
    if remainder != 0:
        pixels_lab = pixels_lab[:len(pixels_rgb)]

    return pixels_lab


def run_kmeans(pixels, num_clusters):
    """
    Run K-Means clustering to find dominant colors.

    Args:
        pixels: Array of RGB pixel data
        num_clusters: Number of clusters to find

    Returns:
        kmeans: KMeans object after fitting
        centers: Cluster centers in RGB format (coordinates of each cluster center)
        labels: Labels assigned to each pixel (which cluster each pixel belongs to)
        percentages: Numpy array containing the percentage of each cluster
    """
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    kmeans.fit(pixels)

    centers = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    counts = np.bincount(labels)
    percentages = counts / len(labels) * 100
    return kmeans, centers, labels, percentages


def run_kmeans_lab(pixels_lab, num_clusters):
    """
    Run K-Means clustering in LAB color space.

    Args:
        pixels_lab: Array of LAB pixel values
        num_clusters: Number of clusters

    Returns:
        kmeans: KMeans object
        centers_lab: Cluster centers in LAB format
        labels: Pixel cluster assignments
        percentages: Percentage of pixels in each cluster
    """
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    kmeans.fit(pixels_lab)

    centers_lab = kmeans.cluster_centers_
    labels = kmeans.labels_
    counts = np.bincount(labels)
    percentages = counts / len(labels) * 100
    return kmeans, centers_lab, labels, percentages


def convert_rgb_centers_to_lab(centers):
    """
    Convert RGB cluster centers to LAB color space.

    Args:
        centers: The RGB clusters

    Returns:
        centers_lab: The RGB clusters converted to LAB
    """
    centers_rgb = centers.astype(np.uint8).reshape(1, -1, 3)
    centers_lab = cv2.cvtColor(centers_rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3)
    return centers_lab


def convert_lab_centers_to_rgb(centers_lab):
    """
    Convert LAB cluster centers to RGB color space.

    Args:
        centers_lab: Array of LAB cluster centers

    Returns:
        centers_rgb: Array of RGB cluster centers
    """
    centers_rgb = []
    for lab in centers_lab:
        lab_img = np.array([[lab]], dtype=np.uint8)
        rgb_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)
        centers_rgb.append(rgb_img[0, 0])
    return np.array(centers_rgb)


def select_distinct_colors(centers, num_to_select):
    """
    Select distinct colors using Greedy Max-Min Distance algorithm in LAB space.

    Args:
        centers: The RGB color centers from K-Means
        num_to_select: Number of distinct colors to select

    Returns:
        selected_rgb: Selected colors in RGB format
        selected_indices: Indices of selected colors in original centers array
    """
    # Convert to LAB space for perceptual distance
    centers_lab = convert_rgb_centers_to_lab(centers)

    num_colors = len(centers)
    if num_to_select >= num_colors:
        return centers, np.arange(num_colors)

    selected_indices = []
    remaining_indices = list(range(num_colors))

    # Step 1: Select color with highest L (lightness) value
    lightness_values = centers_lab[:, 0]
    first_idx = np.argmax(lightness_values)
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)

    # Step 2: Iteratively select colors with maximum distance from nearest selected color
    while len(selected_indices) < num_to_select:
        max_min_distance = -1
        next_color_idx = None

        for candidate_idx in remaining_indices:
            # Calculate distance to all already-selected colors
            candidate_lab = centers_lab[candidate_idx]
            min_distance = float('inf')

            for selected_idx in selected_indices:
                selected_lab = centers_lab[selected_idx]
                # Euclidean distance in LAB space
                distance = np.linalg.norm(candidate_lab - selected_lab)
                min_distance = min(min_distance, distance)

            # Track candidate with maximum minimum distance
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                next_color_idx = candidate_idx

        selected_indices.append(next_color_idx)
        remaining_indices.remove(next_color_idx)

    selected_rgb = centers[selected_indices]
    return selected_rgb, np.array(selected_indices)


def select_distinct_colors_lab(centers_lab, centers_rgb, num_to_select=5):
    """
    Select distinct colors using Greedy Max-Min Distance algorithm in LAB space.
    Works directly with LAB centers.

    Args:
        centers_lab: LAB color centers from K-Means (shape: [n_clusters, 3])
        centers_rgb: Corresponding RGB values (shape: [n_clusters, 3])
        num_to_select: Number of distinct colors to select

    Returns:
        selected_lab: Selected colors in LAB format
        selected_rgb: Selected colors in RGB format
        selected_indices: Indices of selected colors
    """
    num_colors = len(centers_lab)
    if num_to_select >= num_colors:
        return centers_lab, centers_rgb, np.arange(num_colors)

    selected_indices = []
    remaining_indices = list(range(num_colors))

    # Step 1: Select color with highest L (lightness) value
    lightness_values = centers_lab[:, 0]
    first_idx = np.argmax(lightness_values)
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)

    # Step 2: Iteratively select colors with maximum distance from nearest selected color
    while len(selected_indices) < num_to_select:
        max_min_distance = -1
        next_color_idx = None

        for candidate_idx in remaining_indices:
            candidate_lab = centers_lab[candidate_idx]
            min_distance = float('inf')

            for selected_idx in selected_indices:
                selected_lab = centers_lab[selected_idx]
                distance = np.linalg.norm(candidate_lab - selected_lab)
                min_distance = min(min_distance, distance)

            if min_distance > max_min_distance:
                max_min_distance = min_distance
                next_color_idx = candidate_idx

        selected_indices.append(next_color_idx)
        remaining_indices.remove(next_color_idx)

    selected_lab = centers_lab[selected_indices]
    selected_rgb = centers_rgb[selected_indices]
    return selected_lab, selected_rgb, np.array(selected_indices)


def get_palette_for_cnn(image_path, num_clusters=25, num_distinct=10, use_lab=True):
    """
    Extract color palette optimized for CNN training.

    This is a convenience wrapper that combines existing functions
    to produce exactly what CNN.py needs.

    Args:
        image_path: Path to palette/style image
        num_clusters: Number of K-Means clusters (default: 25)
        num_distinct: Number of distinct colors to select (default: 10)
        use_lab: If True, use LAB color space (recommended - default: True)

    Returns:
        Tuple containing:
            - palette_rgb: numpy array (num_distinct, 3) - RGB colors [0-255]
            - palette_lab: numpy array (num_distinct, 3) - LAB colors
            - percentages: numpy array (num_distinct,) - percentage distribution

    Example:
        palette_rgb, palette_lab, percentages = get_palette_for_cnn(
            'vegetables.jpg', num_clusters=25, num_distinct=10
        )
        print(f"Extracted {len(palette_rgb)} distinct colors")
    """
    print(f"\nExtracting palette from: {image_path}")

    # Load image
    _, pixels = load_image_pixels(image_path)
    pixels_lab = convert_rgb_pixels_to_lab(pixels)

    if use_lab:
        # Use LAB color space for clustering
        print(f"Running K-Means in LAB space ({num_clusters} clusters)...")
        _, centers_lab, _, percentages = run_kmeans_lab(pixels_lab, num_clusters)

        # Convert LAB centers to RGB for output
        centers_rgb = convert_lab_centers_to_rgb(centers_lab)

        # Select distinct colors
        print(f"Selecting {num_distinct} distinct colors...")
        palette_lab, palette_rgb, selected_indices = select_distinct_colors_lab(
            centers_lab, centers_rgb, num_to_select=num_distinct
        )
        percentages_selected = percentages[selected_indices]
    else:
        # Use RGB color space for clustering
        print(f"Running K-Means in RGB space ({num_clusters} clusters)...")
        _, centers_rgb, _, percentages = run_kmeans(pixels, num_clusters)

        # Select distinct colors
        print(f"Selecting {num_distinct} distinct colors...")
        palette_rgb, selected_indices = select_distinct_colors(centers_rgb, num_to_select=num_distinct)
        percentages_selected = percentages[selected_indices]

        # Convert palette to LAB for reference
        palette_rgb_uint8 = palette_rgb.astype(np.uint8).reshape(1, -1, 3)
        palette_lab = cv2.cvtColor(palette_rgb_uint8, cv2.COLOR_RGB2LAB).reshape(-1, 3)

    print(f"Palette extracted: {len(palette_rgb)} colors")
    return palette_rgb, palette_lab, percentages_selected


def main():
    """
    Demonstration of color extraction and visualization workflow.
    """
    import config
    from plotting import (
        plot_rgb_cloud_interactive,
        plot_lab_cloud_interactive,
        plot_cluster_centers_3d_interactive,
        plot_cluster_centers_lab_kmeans_interactive,
        visualize_color_palette
    )

    # Use config for paths and parameters
    path = config.TEMPLATE_IMAGE
    num_clusters = config.NUM_CLUSTERS
    num_distinct = config.NUM_DISTINCT
    use_interactive = True

    # Load image and convert to LAB
    _, pixels = load_image_pixels(path)
    pixels_lab = convert_rgb_pixels_to_lab(pixels)
    print(f"\nLAB pixels shape: {pixels_lab.shape}")
    print(f"LAB range - L: [{pixels_lab[:, 0].min()}, {pixels_lab[:, 0].max()}], "
          f"A: [{pixels_lab[:, 1].min()}, {pixels_lab[:, 1].max()}], "
          f"B: [{pixels_lab[:, 2].min()}, {pixels_lab[:, 2].max()}]")

    # Plot RGB cloud
    print("\n=== RGB Color Cloud ===")
    title_rgb = 'All Image Pixels in RGB Color Space'
    plot_rgb_cloud_interactive(pixels, title_rgb)

    # Plot LAB cloud
    print("\n=== LAB Color Cloud ===")
    title_lab = 'All Image Pixels in LAB Color Space'
    plot_lab_cloud_interactive(pixels_lab, pixels, title_lab)

    # RGB K-Means clustering
    print("\n=== RGB K-Means Clustering ===")
    _, centers_rgb, _, percentages_rgb = run_kmeans(pixels, num_clusters)
    print("\nAll RGB K-Means cluster centers:")
    for i, (center, percentage) in enumerate(zip(centers_rgb, percentages_rgb)):
        print(f"Color {i + 1}: RGB{tuple(center)} - {percentage:.1f}%")

    # LAB K-Means clustering
    print("\n=== LAB K-Means Clustering ===")
    _, centers_lab, _, percentages_lab = run_kmeans_lab(pixels_lab, num_clusters)

    # Convert LAB centers back to RGB for display
    centers_lab_as_rgb = convert_lab_centers_to_rgb(centers_lab)

    print("\nAll LAB K-Means cluster centers:")
    for i, (center_lab, center_rgb, percentage) in enumerate(zip(centers_lab, centers_lab_as_rgb, percentages_lab)):
        print(f"Color {i + 1}: LAB({int(center_lab[0])}, {int(center_lab[1])}, {int(center_lab[2])}) = "
              f"RGB{tuple(center_rgb)} - {percentage:.1f}%")

    # Plot RGB cluster centers
    if use_interactive:
        plot_cluster_centers_3d_interactive(centers_rgb, percentages_rgb)

    # Plot LAB cluster centers
    if use_interactive:
        plot_cluster_centers_lab_kmeans_interactive(centers_lab, percentages_lab, centers_lab_as_rgb)

    # Select distinct RGB colors
    print("\n=== Distinct RGB Colors ===")
    distinct_centers_rgb, selected_indices_rgb = select_distinct_colors(centers_rgb, num_to_select=num_distinct)
    distinct_percentages_rgb = percentages_rgb[selected_indices_rgb]

    print(f"\nSelected {num_distinct} distinct RGB colors:")
    for i, (center, percentage) in enumerate(zip(distinct_centers_rgb, distinct_percentages_rgb)):
        print(f"Color {i + 1}: RGB{tuple(center)} - {percentage:.1f}%")

    # Select distinct LAB colors
    print("\n=== Distinct LAB Colors ===")
    distinct_centers_lab, distinct_centers_lab_as_rgb, selected_indices_lab = select_distinct_colors_lab(
        centers_lab, centers_lab_as_rgb, num_to_select=num_distinct
    )
    distinct_percentages_lab = percentages_lab[selected_indices_lab]

    print(f"\nSelected {num_distinct} distinct LAB colors:")
    for i, (center_lab, center_rgb, percentage) in enumerate(
            zip(distinct_centers_lab, distinct_centers_lab_as_rgb, distinct_percentages_lab)):
        print(f"Color {i + 1}: LAB({int(center_lab[0])}, {int(center_lab[1])}, {int(center_lab[2])}) = "
              f"RGB{tuple(center_rgb)} - {percentage:.1f}%")

    # Visualize distinct RGB colors palette
    print("\n=== RGB Color Palette ===")
    visualize_color_palette(distinct_centers_rgb, distinct_percentages_rgb, 'RGB')

    # Visualize distinct LAB colors palette
    print("\n=== LAB Color Palette ===")
    visualize_color_palette(distinct_centers_lab_as_rgb, distinct_percentages_lab, 'LAB')


if __name__ == '__main__':
    main()
