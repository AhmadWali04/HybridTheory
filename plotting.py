import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

try:
    import plotly.graph_objects as go
except ImportError:
    go = None


def plot_rgb_cloud_interactive(pixels, title, max_points=50000):
    """
    Create a 3D scatter plot using Plotly to visualize RGB pixel distribution.

    Args:
        pixels: Array of RGB pixel values (N, 3)
        title: Plot title
        max_points: Maximum number of points to plot (default: 50000)

    Returns:
        Nothing, but shows the cloud of all the points plotted
    """
    if go is None:
        raise RuntimeError("Plotly is not installed. Install it or use matplotlib plots.")

    if len(pixels) > max_points:
        indices = np.random.choice(len(pixels), max_points, replace=False)
        pixels_sampled = pixels[indices]
        print(f"Downsampled from {len(pixels)} to {max_points} points for visualization")
    else:
        pixels_sampled = pixels

    normalized = pixels_sampled / 255.0  
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=pixels_sampled[:, 0],
                y=pixels_sampled[:, 1],
                z=pixels_sampled[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=normalized,
                    opacity=0.5
                )
            )
        ]
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title='R', range=[0, 255]),
            yaxis=dict(title='G', range=[0, 255]),
            zaxis=dict(title='B', range=[0, 255])
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show()


def plot_lab_cloud_interactive(pixels_lab, pixels_rgb, title, max_points=50000):
    """
    Plot LAB color space cloud with colors from original RGB values.

    Args:
        pixels_lab: Array of LAB pixel values (N, 3)
        pixels_rgb: Array of RGB pixel values (N, 3) for coloring
        title: Plot title
        max_points: Maximum number of points to plot

    Returns:
        Nothing, but shows the cloud of all the points plotted
    """
    if go is None:
        raise RuntimeError("Plotly is not installed. Install it or use matplotlib plots.")
    if len(pixels_lab) > max_points:
        indices = np.random.choice(len(pixels_lab), max_points, replace=False)
        pixels_lab_sampled = pixels_lab[indices]
        pixels_rgb_sampled = pixels_rgb[indices]
        print(f"Downsampled from {len(pixels_lab)} to {max_points} points for LAB visualization")
    else:
        pixels_lab_sampled = pixels_lab
        pixels_rgb_sampled = pixels_rgb

    colors = [f'rgb({int(r)},{int(g)},{int(b)})' for r, g, b in pixels_rgb_sampled]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=pixels_lab_sampled[:, 0],  
                y=pixels_lab_sampled[:, 1],  
                z=pixels_lab_sampled[:, 2],  
                mode='markers',
                marker=dict(
                    size=2,
                    color=colors,
                    opacity=0.5
                )
            )
        ]
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title='L (Lightness)', range=[0, 100]),
            yaxis=dict(title='A (Green-Red)', range=[-128, 127]),
            zaxis=dict(title='B (Blue-Yellow)', range=[-128, 127])
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show()


def plot_cluster_centers_3d_interactive(centers, percentages):
    """
    Plot RGB cluster centers in 3D space.

    Args:
        centers: The coordinates of each cluster object
        percentages: The percentage of the total image that each colour is present

    Returns:
        Nothing, but it displays the plot
    """
    if go is None:
        raise RuntimeError("Plotly is not installed. Install it or use matplotlib plots.")

    normalized = centers / 255.0
    labels = [
        f"RGB({c[0]}, {c[1]}, {c[2]})<br>{p:.1f}%"
        for c, p in zip(centers, percentages)
    ]
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=centers[:, 0],
                y=centers[:, 1],
                z=centers[:, 2],
                mode='markers+text',
                text=labels,
                textposition='top center',
                marker=dict(
                    size=8,
                    color=normalized,
                    line=dict(color='black', width=1)
                )
            )
        ]
    )
    fig.update_layout(
        title='K-Means Cluster Centers in RGB Space',
        scene=dict(
            xaxis_title='R',
            yaxis_title='G',
            zaxis_title='B'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show()


def plot_cluster_centers_lab_kmeans_interactive(centers_lab, percentages, centers_rgb):
    """
    Plot LAB K-Means cluster centers colored by their RGB equivalents.

    Args:
        centers_lab: Cluster centers in LAB space
        percentages: Percentage distribution
        centers_rgb: Corresponding RGB values for coloring

    Returns:
        Nothing, but displays the plot for LAB cluster centers
    """
    if go is None:
        raise RuntimeError("Plotly is not installed. Install it or use matplotlib plots.")

    colors = [f'rgb({int(r)},{int(g)},{int(b)})' for r, g, b in centers_rgb]
    labels = [
        f"LAB({int(c[0])}, {int(c[1])}, {int(c[2])})<br>RGB({int(rgb[0])}, {int(rgb[1])}, {int(rgb[2])})<br>{p:.1f}%"
        for c, rgb, p in zip(centers_lab, centers_rgb, percentages)
    ]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=centers_lab[:, 0],
                y=centers_lab[:, 1],
                z=centers_lab[:, 2],
                mode='markers+text',
                text=labels,
                textposition='top center',
                marker=dict(
                    size=10,
                    color=colors,
                    line=dict(color='black', width=1)
                )
            )
        ]
    )
    fig.update_layout(
        title='K-Means Cluster Centers in LAB Space',
        scene=dict(
            xaxis=dict(title='L (Lightness)'),
            yaxis=dict(title='A (Green-Red)'),
            zaxis=dict(title='B (Blue-Yellow)')
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show()


def plot_cluster_centers_lab_interactive(centers, percentages):
    """
    Plot K-Means cluster centers in LAB space.

    Args:
        centers: RGB cluster centers (will be converted to LAB)
        percentages: Percentage distribution

    Returns:
        Nothing, but displays the plot
    """
    if go is None:
        raise RuntimeError("Plotly is not installed. Install it or use matplotlib plots.")

    from colour import convert_rgb_centers_to_lab

    centers_lab = convert_rgb_centers_to_lab(centers)
    normalized = centers / 255.0
    labels = [
        f"LAB({c_lab[0]}, {c_lab[1]}, {c_lab[2]})<br>RGB({c_rgb[0]}, {c_rgb[1]}, {c_rgb[2]})<br>{p:.1f}%"
        for c_lab, c_rgb, p in zip(centers_lab, centers, percentages)
    ]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=centers_lab[:, 0],
                y=centers_lab[:, 1],
                z=centers_lab[:, 2],
                mode='markers+text',
                text=labels,
                textposition='top center',
                marker=dict(
                    size=8,
                    color=normalized,
                    line=dict(color='black', width=1)
                )
            )
        ]
    )
    fig.update_layout(
        title='K-Means Cluster Centers in LAB Space',
        scene=dict(
            xaxis_title='L',
            yaxis_title='A',
            zaxis_title='B'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show()


def visualize_color_palette(centers, percentages, color_space):
    """
    Visualize the color palette as a horizontal bar graph.

    Args:
        centers: The RGB clusters themselves
        percentages: The percent that those clusters compose the image of
        color_space: Either 'RGB' or 'LAB' to determine the title

    Returns:
        Nothing, but shows a graph
    """
    fig, ax = plt.subplots(figsize=(12, 3))
    x_position = 0
    for center, percentage in zip(centers, percentages):
        normalized_color = center / 255.0
        width = percentage
        rect = Rectangle((x_position, 0), width, 1, facecolor=normalized_color, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        label = f"RGB({center[0]}, {center[1]}, {center[2]})\n{percentage:.1f}%"
        ax.text(
            x_position + width / 2,
            0.5,
            label,
            ha='center',
            va='center',
            fontsize=9,
            fontweight='bold',
            color='white' if sum(center) < 384 else 'black'
        )
        x_position += width
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Percentage (%)', fontsize=12, fontweight='bold')

    ax.set_title(f'Dominant {color_space} Color Palette', fontsize=14, fontweight='bold')
    ax.set_yticks([])
    ax.set_aspect('auto')
    plt.tight_layout()
    plt.show()


def visualize_palette_comparison(palette1_rgb, palette2_rgb,
                                label1="Palette 1", label2="Palette 2"):
    """
    Visualize two color palettes side by side in a single figure.

    Useful for comparing source vs target palettes before CNN training.

    Args:
        palette1_rgb: numpy array (N, 3) - first palette, RGB values [0-255]
        palette2_rgb: numpy array (M, 3) - second palette, RGB values [0-255]
        label1: Label for first palette (default: "Palette 1")
        label2: Label for second palette (default: "Palette 2")

    Example:
        source_palette, _, _ = get_palette_for_cnn('spiderman.jpg')
        target_palette, _, _ = get_palette_for_cnn('vegetables.jpg')
        visualize_palette_comparison(source_palette, target_palette,
                                    "Spiderman Colors", "Vegetable Colors")
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    # Palette 1
    x_position = 0
    for center in palette1_rgb:
        normalized_color = center / 255.0
        width = 1.0 / len(palette1_rgb)
        rect = Rectangle((x_position, 0), width, 1,
                        facecolor=normalized_color, edgecolor='black', linewidth=2)
        ax1.add_patch(rect)
        x_position += width

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title(label1, fontsize=14, fontweight='bold')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_aspect('auto')

    # Palette 2
    x_position = 0
    for center in palette2_rgb:
        normalized_color = center / 255.0
        width = 1.0 / len(palette2_rgb)
        rect = Rectangle((x_position, 0), width, 1,
                        facecolor=normalized_color, edgecolor='black', linewidth=2)
        ax2.add_patch(rect)
        x_position += width

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title(label2, fontsize=14, fontweight='bold')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_aspect('auto')

    plt.tight_layout()
    plt.show()



def visualize_single_palette(palette_rgb, title="Color Palette", show_labels=True):
    """
    Display a single palette in its own popup window.

    Args:
        palette_rgb: numpy array (N, 3) - palette colors [0-255]
        title: Window title
        show_labels: Whether to show RGB labels on each color

    Returns:
        fig: matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(12, 3))

    x_position = 0
    for center in palette_rgb:
        normalized_color = center / 255.0
        width = 1.0 / len(palette_rgb)
        rect = Rectangle((x_position, 0), width, 1,
                        facecolor=normalized_color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)

        if show_labels:
            # Add RGB label
            label = f"RGB\n({int(center[0])},\n{int(center[1])},\n{int(center[2])})"
            ax.text(x_position + width/2, 0.5, label, ha='center', va='center',
                   fontsize=8,
                   color='white' if sum(center) < 384 else 'black')
        x_position += width

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('auto')

    plt.tight_layout()

    return fig


def visualize_dual_palettes(clustered_palette_rgb, distinct_palette_rgb,
                           title1="Clustered Colors (10)",
                           title2="LAB-Selected Distinct (5)"):
    """
    Display two palettes in separate popup windows simultaneously.

    This is the function for Use Case 4 that shows:
    - Popup 1: All colors from K-means clustering (e.g., 10 colors)
    - Popup 2: Most distinct colors selected by LAB algorithm (e.g., 5 colors)

    Args:
        clustered_palette_rgb: numpy array (N, 3) - colors from K-means clustering
        distinct_palette_rgb: numpy array (M, 3) - colors from LAB selection
        title1: Title for first popup (clustered colors)
        title2: Title for second popup (LAB-selected distinct colors)

    Returns:
        tuple: (fig1, fig2) - both matplotlib figure objects

    Example:
        # Extract 10 colors, then select top 5 most distinct
        visualize_dual_palettes(
            all_10_colors,
            top_5_distinct,
            title1="10 Clustered Colors from vegetables.jpg",
            title2="Top 5 Most Distinct (LAB Algorithm)"
        )
    """
    fig1 = visualize_single_palette(
        clustered_palette_rgb,
        title=title1,
        show_labels=True
    )
    plt.show(block=False)  
    fig2 = visualize_single_palette(
        distinct_palette_rgb,
        title=title2,
        show_labels=True
    )
    plt.show()  

    return fig1, fig2

def close_all_figures():
    """Close all open matplotlib figures."""
    plt.close('all')



def plot_palette_color_wheel(palette_lab, palette_rgb):
    """
    Plot palette colors on the LAB a*-b* plane (hue-saturation wheel).

    Shows where palette colors fall in terms of hue and saturation.
    Neutral colors cluster near the center, saturated colors spread outward.

    Args:
        palette_lab: numpy array (N, 3) LAB values (OpenCV scale: L=0-255, a=0-255, b=0-255)
        palette_rgb: numpy array (N, 3) RGB values [0-255]
    """
    if go is None:
        raise RuntimeError("Plotly is not installed.")

    a_vals = palette_lab[:, 1].astype(float)
    b_vals = palette_lab[:, 2].astype(float)

    colors = [f'rgb({int(r)},{int(g)},{int(b)})' for r, g, b in palette_rgb]
    labels = [f"Color {i+1} (L*={int(lab[0])})" for i, lab in enumerate(palette_lab)]

    fig = go.Figure()

    theta = np.linspace(0, 2 * np.pi, 100)
    circle_r = 80
    fig.add_trace(go.Scatter(
        x=128 + circle_r * np.cos(theta),
        y=128 + circle_r * np.sin(theta),
        mode='lines',
        line=dict(color='lightgray', width=1, dash='dot'),
        showlegend=False, hoverinfo='skip'
    ))

    fig.add_hline(y=128, line_color='lightgray', line_width=0.5)
    fig.add_vline(x=128, line_color='lightgray', line_width=0.5)

    fig.add_trace(go.Scatter(
        x=a_vals, y=b_vals,
        mode='markers+text',
        text=labels,
        textposition='top center',
        textfont=dict(size=10),
        marker=dict(
            size=20,
            color=colors,
            line=dict(color='black', width=2),
            opacity=0.9
        ),
        hovertext=[f"Color {i+1}<br>L*={int(lab[0])}, a*={int(lab[1])}, b*={int(lab[2])}<br>"
                    f"RGB({int(rgb[0])}, {int(rgb[1])}, {int(rgb[2])})"
                    for i, (lab, rgb) in enumerate(zip(palette_lab, palette_rgb))],
        hoverinfo='text',
        showlegend=False
    ))

    fig.update_layout(
        title='Palette Colors on LAB a*-b* Plane (Hue-Saturation)',
        xaxis_title='a* (Green ← → Red)',
        yaxis_title='b* (Blue ← → Yellow)',
        xaxis=dict(range=[0, 255], constrain='domain'),
        yaxis=dict(range=[0, 255], scaleanchor='x', scaleratio=1),
        width=800, height=800,
        margin=dict(l=60, r=40, b=60, t=40)
    )
    fig.show()


if __name__ == '__main__':
    print("Testing plotting.py...")

    test_palette = np.array([
        [255, 0, 0],      
        [0, 255, 0],      
        [0, 0, 255],      
        [255, 255, 0],    
        [255, 0, 255],    
        [0, 255, 255],    
        [128, 128, 128],  
        [255, 128, 0],    
        [128, 0, 255],    
        [0, 128, 128],    
    ])

    test_distinct = test_palette[:5] 

    print("Displaying dual palette visualization...")
    visualize_dual_palettes(
        test_palette,
        test_distinct,
        title1="Test: 10 Clustered Colors",
        title2="Test: 5 LAB-Selected Distinct"
    )

    print("Test complete!")
