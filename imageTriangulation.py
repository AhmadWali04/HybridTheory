import math
import matplotlib
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
import numpy as np
from optparse import OptionParser
import os
from PIL import Image
import random
from scipy.spatial import Delaunay
import sys

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import colour


def setup_matplotlib():
    """Configure matplotlib display settings."""
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    fontsize = 18
    params = {
        'axes.labelsize': fontsize,
        'font.size': fontsize,
        'legend.fontsize': 12,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize,
        'axes.titlesize': fontsize,
        'lines.linewidth': 1,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans', 'sans-serif'],
        'axes.grid': False,
        'figure.figsize': (6.75, 4),
        'figure.dpi': 250,
        'mathtext.fontset': 'cm'
    }
    for param in params.keys():
        matplotlib.rcParams[param] = params[param]


def load_image(filepath):
    """Load and display original image."""
    image_orig = Image.open(filepath)
    image = Image.open(filepath)
    image.show()
    return image_orig, image


def convert_to_greyscale(image):
    """Convert image to greyscale and display."""
    image_data = image.load()
    image = image.convert('RGBA')  
    for i in range(image.width):
        for j in range(image.height):
            try:
                r, g, b, a = image.getpixel((i, j))
            except:
                r, g, b = image.getpixel((i, j))
            grey = 0.299*r + 0.587*g + 0.114*b
            image_data[i, j] = (int(grey), int(grey), int(grey))
    image.show()
    return image


def sharpen_image(image):
    """Apply sharpening operator to image."""
    H = [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
    image_data = image.load()
    
    G = [[0]*image.height for i in range(image.width)]
    maxG = 0
    
    for i in range(1, image.width - 2):
        for j in range(1, image.height - 2):
            x1 = H[0][0]*image.getpixel((i - 1, j - 1))[0]
            x2 = H[0][1]*image.getpixel((i, j - 1))[0]
            x3 = H[0][2]*image.getpixel((i + 1, j - 1))[0]
            x4 = H[1][0]*image.getpixel((i - 1, j))[0]
            x5 = H[1][1]*image.getpixel((i, j))[0]
            x6 = H[1][2]*image.getpixel((i + 1, j))[0]
            x7 = H[2][0]*image.getpixel((i - 1, j + 1))[0]
            x8 = H[2][1]*image.getpixel((i, j + 1))[0]
            x9 = H[2][2]*image.getpixel((i + 1, j + 1))[0]
            G_val = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9
            G[i][j] = G_val
            
            if(G_val > maxG):
                maxG = G_val
    
    for i in range(1, image.width - 2):
        for j in range(1, image.height - 2):
            image_data[i, j] = (round(G[i][j]/maxG * 255), round(G[i][j]/maxG * 255), round(G[i][j]/maxG * 255))
    image.show()
    return image


def detect_edges(image):
    """Apply Sobel edge detection operator."""
    xKernel = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
    yKernel = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    
    image_data = image.load()
    Gx = [[0]*image.height for i in range(image.width)]
    Gy = [[0]*image.height for i in range(image.width)]
    G = [[0]*image.height for i in range(image.width)]
    theta = [[0]*image.height for i in range(image.width)]
    
    maxG = 0
    for i in range(1, image.width - 2):
        for j in range(1, image.height - 2):
            x1 = xKernel[0][0]*image.getpixel((i - 1, j - 1))[0]
            x2 = xKernel[0][1]*image.getpixel((i, j - 1))[0]
            x3 = xKernel[0][2]*image.getpixel((i + 1, j - 1))[0]
            x4 = xKernel[1][0]*image.getpixel((i - 1, j))[0]
            x5 = xKernel[1][1]*image.getpixel((i, j))[0]
            x6 = xKernel[1][2]*image.getpixel((i + 1, j))[0]
            x7 = xKernel[2][0]*image.getpixel((i - 1, j + 1))[0]
            x8 = xKernel[2][1]*image.getpixel((i, j + 1))[0]
            x9 = xKernel[2][2]*image.getpixel((i + 1, j + 1))[0]
            Gx_val = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9
            Gx[i][j] = Gx_val
            
            y1 = yKernel[0][0]*image.getpixel((i - 1, j - 1))[0]
            y2 = yKernel[0][1]*image.getpixel((i, j - 1))[0]
            y3 = yKernel[0][2]*image.getpixel((i + 1, j - 1))[0]
            y4 = yKernel[1][0]*image.getpixel((i - 1, j))[0]
            y5 = yKernel[1][1]*image.getpixel((i, j))[0]
            y6 = yKernel[1][2]*image.getpixel((i + 1, j))[0]
            y7 = yKernel[2][0]*image.getpixel((i - 1, j + 1))[0]
            y8 = yKernel[2][1]*image.getpixel((i, j + 1))[0]
            y9 = yKernel[2][2]*image.getpixel((i + 1, j + 1))[0]
            Gy_val = y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8 + y9
            Gy[i][j] = Gy_val
            
            G_val = math.sqrt(Gx_val**2 + Gy_val**2)
            G[i][j] = G_val
            
            try:
                theta[i][j] = math.atan(Gy_val/Gx_val)
            except:
                theta[i][j] = math.inf
            if(G_val > maxG):
                maxG = G_val
    
    for i in range(1, image.width - 2):
        for j in range(1, image.height - 2):
            image_data[i, j] = (round(G[i][j]/maxG * 255), round(G[i][j]/maxG * 255), round(G[i][j]/maxG * 255))
    image.show()
    return image


def determine_vertices(image, threshold, density_reduction):
    """Extract vertices from edge-detected image."""
    image_data = image.load()
    S = []
    for i in range(1, image.width - 2):
        for j in range(1, image.height - 2):
            if image_data[i, j][0] > threshold:
                S.append([i, j])
    S = random.sample(S, round(len(S)/density_reduction))  
    S.append([0, 0])
    S.append([0, image.height - 1])
    S.append([image.width - 1, 0])
    S.append([image.width, image.height])
    S = np.array(S)
    return S


def visualize_triangulation(S, triangles):
    """Display triangulated mesh."""
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.triplot(S[:,0], S[:,1], triangles.simplices, color='black')
    ax.set_axis_off()
    plt.show()


def colorize_triangulation(S, triangles, image_orig, save=False, image_name=None):
    """Color triangles based on original image and optionally save."""
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.invert_yaxis()  
    for triangle in range(len(triangles.simplices)):
        vertices = S[triangles.simplices[triangle]]
        a = vertices[0]
        b = vertices[1]
        c = vertices[2]

        xs = vertices[:,0]
        ys = vertices[:,1]

        centroid = [(a[0] + b[0] + c[0])/3, (a[1] + b[1] + c[1])/3]
        color = image_orig.getpixel((centroid[0], centroid[1]))
        R = color[0]/255
        G = color[1]/255
        B = color[2]/255
        ax.fill(xs, ys, color=(R, G, B))
    ax.set_axis_off()
    if save and image_name:
        if os.path.isabs(image_name) or '/' in image_name:
            save_path = image_name
        else:
            save_path = os.path.join(os.getcwd(), 'triangulatedImages', image_name)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

    plt.show()


def find_closest_color(target_rgb, distinct_colors):
    """
    Find the closest color from distinct_colors to target_rgb using Euclidean distance.

    Args:
        target_rgb: RGB tuple (r, g, b) - target color
        distinct_colors: numpy array of shape (n, 3) - distinct RGB colors

    Returns:
        closest_color: RGB tuple of the closest distinct color
    """
    target = np.array(target_rgb)
    distances = np.linalg.norm(distinct_colors - target, axis=1)
    closest_idx = np.argmin(distances)
    return distinct_colors[closest_idx]


def custom_color_triangulation(S, triangles, image_orig, filepath, num_clusters=10, num_distinct=10, save=False, image_name=None):
    """
    Color triangles using the closest distinct color from K-Means clustering.

    This function:
    1. Extracts 10 distinct colors from the image using colour.py
    2. For each triangle, gets the centroid-based color from the original image
    3. Maps that color to the closest of the 10 distinct colors
    4. Paints the triangle with only that distinct color

    Args:
        S: Vertices array
        triangles: Delaunay triangulation object
        image_orig: Original PIL Image
        filepath: Path to the image file
        num_clusters: Number of K-Means clusters to extract
        num_distinct: Number of distinct colors to select
        save: Whether to save the image
        image_name: Name to save the image with
    """
    _, pixels = colour.load_image_pixels(filepath)
    _, centers, _, percentages = colour.run_kmeans(pixels, num_clusters)
    distinct_centers, selected_indices = colour.select_distinct_colors(centers, num_to_select=num_distinct)

    print(f"\nUsing {num_distinct} distinct colors for custom triangulation:")
    for i, center in enumerate(distinct_centers):
        print(f"  Color {i + 1}: RGB{tuple(center)}")

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.invert_yaxis() 

    for triangle in range(len(triangles.simplices)):
        vertices = S[triangles.simplices[triangle]]
        a = vertices[0]
        b = vertices[1]
        c = vertices[2]

        xs = vertices[:,0]
        ys = vertices[:,1]

        centroid = [(a[0] + b[0] + c[0])/3, (a[1] + b[1] + c[1])/3]

        centroid_color = image_orig.getpixel((int(centroid[0]), int(centroid[1])))

        if len(centroid_color) > 3:
            centroid_rgb = centroid_color[:3]
        else:
            centroid_rgb = centroid_color
        closest_color = find_closest_color(centroid_rgb, distinct_centers)
        R = closest_color[0] / 255
        G = closest_color[1] / 255
        B = closest_color[2] / 255

        ax.fill(xs, ys, color=(R, G, B))

    ax.set_axis_off()
    ax.set_title(f'Custom Triangulation with {num_distinct} Distinct Colors', fontsize=12, fontweight='bold')

    if save and image_name:
        save_path = os.path.join(os.getcwd(), 'triangulatedImages', image_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        print(f"\nSaved custom triangulation to: {save_path}")

    plt.show()


def get_triangle_centroids_and_colors(S, triangles, image_orig):
    """
    Extract centroid coordinates and colors for all triangles.

    This prepares data that CNN.py needs for color transfer.

    Args:
        S: Vertices array, shape (N, 2)
        triangles: Delaunay triangulation object
        image_orig: Original PIL Image

    Returns:
        Dictionary containing:
            - 'centroids': numpy array (num_triangles, 2) - centroid coordinates
            - 'colors_rgb': numpy array (num_triangles, 3) - RGB colors [0-255]
            - 'triangle_vertices': list of num_triangles arrays, each shape (3, 2)

    This extracts exactly what the CNN needs to know about each triangle:
    - Where it is (centroid)
    - What color it currently is (from original image)
    - What its vertices are (for rendering later)
    """
    centroids = []
    colors_rgb = []
    triangle_vertices = []

    for triangle_idx in triangles.simplices:
        vertices = S[triangle_idx]
        centroid = vertices.mean(axis=0)
        try:
            color = image_orig.getpixel((int(centroid[0]), int(centroid[1])))
        except:
            x = np.clip(int(centroid[0]), 0, image_orig.width - 1)
            y = np.clip(int(centroid[1]), 0, image_orig.height - 1)
            color = image_orig.getpixel((x, y))
        if len(color) > 3:
            color_rgb = color[:3]
        else:
            color_rgb = color

        centroids.append(centroid)
        colors_rgb.append(color_rgb)
        triangle_vertices.append(vertices)

    return {
        'centroids': np.array(centroids),
        'colors_rgb': np.array(colors_rgb),
        'triangle_vertices': triangle_vertices
    }


def render_triangulation_with_colors(S, triangles, triangle_colors,
                                    title="Colored Triangulation",
                                    save_path=None):
    """
    Render a triangulation with specified colors for each triangle.

    This is a generalized version of colorize_triangulation that accepts
    pre-computed colors instead of sampling from the original image.
    CNN.py will use this to render its results.

    Args:
        S: Vertices array
        triangles: Delaunay triangulation
        triangle_colors: List/array of (num_triangles, 3) RGB colors [0-255]
        title: Plot title (default: "Colored Triangulation")
        save_path: Optional path to save figure

    Returns:
        fig: matplotlib figure object
        ax: matplotlib axes object

    Example:
        fig, ax = render_triangulation_with_colors(S, triangles, colors,
                                                   title="CNN Result")
        if save:
            plt.savefig('output.png')
        plt.show()
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect('equal')
    ax.invert_yaxis()  

    for i, triangle_idx in enumerate(triangles.simplices):
        vertices = S[triangle_idx]
        xs = vertices[:, 0]
        ys = vertices[:, 1]
        rgb_normalized = triangle_colors[i] / 255.0
        ax.fill(xs, ys, color=rgb_normalized, edgecolor='none')

    ax.set_axis_off()
    ax.set_title(title, fontsize=14, fontweight='bold')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0)
        print(f"Saved figure to: {save_path}")

    return fig, ax


def pipeline_with_cnn(source_image_path, target_image_path,
                     threshold=50, density_reduction=60,
                     num_clusters=25, num_distinct=10,
                     use_pretrained_model=None,
                     train_epochs=1000,
                     temperature=0.5,
                     save_model_path=None,
                     device='cpu',
                     save_output=False):
    """
    Complete pipeline integrating triangulation with CNN color transfer.

    This is the main entry point that ties everything together:
    1. Load and triangulate source image
    2. Extract palette from target image
    3. Train or load CNN model
    4. Apply CNN to triangulation
    5. Display and optionally save results

    Args:
        source_image_path: Image to triangulate
        target_image_path: Image to extract palette from
        threshold: Edge detection threshold (default: 50)
        density_reduction: Vertex density parameter (default: 60)
        num_clusters: K-Means clusters for palette (default: 25)
        num_distinct: Distinct colors to use (default: 10)
        use_pretrained_model: Path to pretrained model (skip training if provided)
        train_epochs: Epochs to train if not using pretrained (default: 1000)
        temperature: Softmax temperature for palette selection (default: 0.5)
        save_model_path: Where to save trained model (default: None)
        device: 'cpu' or 'cuda' (default: 'cpu')
        save_output: Whether to save output images (default: False)

    Returns:
        Dictionary containing:
            - 'S': Vertices array
            - 'triangles': Delaunay triangulation object
            - 'image_orig': Original PIL Image
            - 'model': Trained or loaded ColorTransferNet
            - 'cnn_result': Result dict from apply_cnn_to_triangulation
            - 'triangle_data': Result dict from get_triangle_centroids_and_colors

    Example:
        results = pipeline_with_cnn(
            source_image_path='spiderman.jpg',
            target_image_path='vegetables.jpg',
            threshold=50,
            density_reduction=60,
            train_epochs=1000,
            temperature=0.5,
            save_model_path='models/spiderman_veg.pth'
        )
        results['cnn_result']['figure'].show()
    """
    if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import CNN

    print("\n" + "="*60)
    print("CNN-BASED COLOR TRANSFER TRIANGULATION PIPELINE")
    print("="*60)

    print("\n[Step 1] Loading and triangulating source image...")
    setup_matplotlib()
    image_orig, image = load_image(source_image_path)
    image = convert_to_greyscale(image)
    image = sharpen_image(image)
    image = detect_edges(image)
    S = determine_vertices(image, threshold, density_reduction)
    triangles = Delaunay(S)
    print(f"Triangulation complete: {len(triangles.simplices)} triangles created")

    visualize_triangulation(S, triangles)

    print("\n[Step 2] Preparing model...")
    if use_pretrained_model:
        print(f"Loading pretrained model from: {use_pretrained_model}")
        model, metadata = CNN.load_trained_model(use_pretrained_model, device=device)
    else:
        print("Training new model...")
        data = CNN.prepare_training_data(
            source_image_path, target_image_path,
            num_clusters=num_clusters, num_distinct=num_distinct,
            use_lab=True, device=device
        )
        model, loss_history = CNN.train_color_transfer_network(
            data['source_pixels'], data['target_palette'],
            data['source_pixels_lab'], data['target_palette_lab'],
            epochs=train_epochs, batch_size=512, lr=0.001,
            temperature=temperature,
            device=device, save_progress=False
        )

        print("\nGenerating training visualization...")
        CNN.visualize_training_progress(
            model, data['source_pixels'], data['target_palette'],
            loss_history, device=device,
            save_path=None
        )

        if save_model_path:
            CNN.save_trained_model(
                model, save_model_path,
                metadata={
                    'source_image': source_image_path,
                    'target_image': target_image_path,
                    'num_clusters': num_clusters,
                    'num_distinct': num_distinct,
                    'training_epochs': train_epochs,
                    'final_loss': loss_history['total'][-1]
                }
            )

    print("\n[Step 3] Applying CNN to triangulation...")
    cnn_result = CNN.apply_cnn_to_triangulation(
        model, S, triangles, image_orig, device=device
    )

    print("\n[Step 4] Displaying results...")

    cnn_result['figure'].show()

    triangle_data = get_triangle_centroids_and_colors(S, triangles, image_orig)

    if save_output:
        import config
        from datetime import datetime
        base_output_dir = os.path.join(os.getcwd(), 'triangulatedImages')
        method_dir = 'customColored'  
        template_name = config.get_image_basename(source_image_path)
        model_name = config.get_model_name()
        output_dir = os.path.join(base_output_dir, method_dir, template_name, model_name)
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        cnn_output_path = os.path.join(output_dir, f'cnn_{timestamp}.png')
        cnn_result['figure'].savefig(cnn_output_path, dpi=150, bbox_inches='tight', pad_inches=0)
        print(f"Saved CNN result to: {cnn_output_path}")
    return {
        'S': S,
        'triangles': triangles,
        'image_orig': image_orig,
        'model': model,
        'cnn_result': cnn_result,
        'triangle_data': triangle_data
    }


def main():
    """Main execution function."""
    parser = OptionParser()
    parser.add_option('-d', '--d', dest='d',
                      action='store', type='int', default=60,
                      help='Density parameter')
    parser.add_option('-f', '--file', dest='filepath', default=os.getcwd() + '/HybridTheory/originalImages/waterLily.jpeg',
                      action='store', help='Image path for image to triangulate')
    parser.add_option('-g', '--g', dest='finalname', default='triangulated.jpeg',
                      action='store', help='Final image name for saving')
    parser.add_option('-s', '--s', dest='save',
                      action='store_true', help='Save final image')
    parser.add_option('-t', '--t', dest='t',
                      action='store', type='int', default=50,
                      help='Threshold value')
    
    (options, args) = parser.parse_args()
    
    setup_matplotlib()
    
    image_orig, image = load_image(options.filepath)
    image = convert_to_greyscale(image)
    image = sharpen_image(image)
    image = detect_edges(image)
    
    S = determine_vertices(image, options.t, options.d)
    triangles = Delaunay(S)
    visualize_triangulation(S, triangles)
    colorize_triangulation(S, triangles, image_orig, save=options.save, image_name=options.finalname)


if __name__ == '__main__':
    main()