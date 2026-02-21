"""
main.py

Example script showing how to use the CNN color transfer system.

This script demonstrates:
1. Basic usage: triangulate an image and apply a color palette using CNN
2. Reusing a previously trained model (much faster)
3. Plotting & visualization menu (11 plot types)
4. Standard coloring with original image colors (no CNN)
5. Cleanup orphaned data directories

"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  
import imageTriangulation
import colour
import CNN
import tensorboard_feedback
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans', 'bitstream vera sans', 'sans-serif']
plt.rcParams['font.family'] = 'sans-serif'
from datetime import datetime


def example_1_basic_usage():
    """
    Basic usage with FEEDBACK LOOP: triangulate, train, apply, rate, and fine-tune.

    Complete workflow:
    1. Loads template image and detects edges
    2. Performs Delaunay triangulation
    3. Extracts color palette from palette image
    4. Displays interactive LAB palette visualization
    5. Trains a CNN to map colors (1000 epochs)
    6. Applies CNN to paint each triangle
    7. Shows colored result
    8. Displays feedback form (rate frequency and placement of each color)
    9. Saves feedback to feedback_data/
    10. Fine-tunes model with all previous feedback (150 epochs)

    Expected output:
    - Triangulated image with colors from palette
    - Interactive feedback form for rating color usage
    - Fine-tuned model incorporating your feedback
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: CNN Color Transfer with Feedback Loop")
    print("="*70)

    config.print_config()

    errors = config.validate_config()
    if errors:
        print("\nConfiguration errors:")
        for err in errors:
            print(f"  - {err}")
        return None

    source_img = config.TEMPLATE_IMAGE
    target_img = config.PALETTE_IMAGE

    print("\nRunning CNN color transfer pipeline...")
    results = imageTriangulation.pipeline_with_cnn(
        source_image_path=source_img,
        target_image_path=target_img,
        threshold=50,
        density_reduction=config.DENSITY_REDUCTION,
        num_clusters=config.NUM_CLUSTERS,
        num_distinct=config.NUM_DISTINCT,
        train_epochs=config.EPOCHS,
        temperature=config.TEMPERATURE,
        save_model_path=config.get_model_path(),
        device='cpu',
        save_output=True
    )

    print("\nExtracting palette for feedback...")
    palette_rgb, palette_lab, percentages = colour.get_palette_for_cnn(
        target_img, num_clusters=config.NUM_CLUSTERS, num_distinct=config.NUM_DISTINCT
    )

    triangle_colors = results['cnn_result']['triangle_colors']  

    fig = results['cnn_result']['figure']
    fig.canvas.draw()

    image_data = np.asarray(fig.canvas.buffer_rgba())
    image_data = image_data[:, :, :3]  
    image_cnn = Image.fromarray(image_data)
    image_original = Image.open(source_img)
    feedback_choice = input("\nDo you want to help train me with feedback? (y/n): ").strip().lower()

    if feedback_choice == 'y':
        print("\n" + "="*70)
        print("TENSORBOARD FEEDBACK COLLECTION")
        print("="*70)
        print("\n Starting TensorBoard visualization...")
        print("   1. Open http://127.0.0.1:6006 in your browser")
        print("   2. Review the visualizations")
        print("   3. Return here to provide ratings")
        print("="*70)

        template_name = config.get_image_basename(source_img)
        session_name = config.get_session_name(datetime.now().strftime("%Y%m%d_%H%M%S"))
        scores = tensorboard_feedback.get_user_feedback_tensorboard(
            palette_rgb, palette_lab, triangle_colors,
            image_original, image_cnn,
            session_name=session_name,
            template_name=template_name
        )

        print("\nFine-tuning model with feedback from all previous sessions...")

        data = CNN.prepare_training_data(
            source_img, target_img,
            num_clusters=config.NUM_CLUSTERS, num_distinct=config.NUM_DISTINCT,
            use_lab=True, device='cpu'
        )

        model = results['model']
        model_ft, loss_history_ft = CNN.fine_tune_with_feedback(
            model,
            data['source_pixels'], data['target_palette'],
            data['source_pixels_lab'], data['target_palette_lab'],
            epochs=config.FINE_TUNE_EPOCHS, batch_size=512, lr=0.0005,
            device='cpu',
            feedback_dir='feedback_data',
            log_dir='runs/fine_tuning',
            model_name=config.get_model_name(),
            template_name=template_name
        )

        CNN.save_trained_model(
            model_ft,
            config.get_model_path(),
            metadata={
                'source_image': source_img,
                'target_image': target_img,
                'num_clusters': config.NUM_CLUSTERS,
                'num_distinct': config.NUM_DISTINCT,
                'training_epochs': config.EPOCHS,
                'fine_tuned_epochs': config.FINE_TUNE_EPOCHS,
                'temperature': config.TEMPERATURE,
                'final_loss': loss_history_ft['total'][-1] if loss_history_ft['total'] else 0
            }
        )

        print("\n" + "="*70)
        print("TRAINING & FEEDBACK CYCLE COMPLETE")
        print("="*70)
        print("\n WHAT JUST HAPPENED:")
        print(f"   Initial training: {config.EPOCHS} epochs (CNN learned color mapping)")
        print(f"   Temperature: {config.TEMPERATURE} (palette selection sharpness)")
        print("   Your feedback: Frequency and placement ratings for 10 colors")
        print(f"   Fine-tuning: {config.FINE_TUNE_EPOCHS} epochs (model adapted to your feedback)")
        print("\n SAVED TO:")
        print(f"   Model: {config.get_model_path()}")
        print(f"   Feedback: feedback_data/{session_name}.json")
        print(f"   TensorBoard: runs/feedback/{session_name}")
        print("\n HOW FEEDBACK WORKS:")
        print("   Frequency: Model learns to use colors you rated low")
        print("   Placement: Model learns better color placement")
        print("   Recent sessions weighted more heavily")
    else:
        print("\nSkipping feedback.")
        CNN.save_trained_model(
            results['model'],
            config.get_model_path(),
            metadata={
                'source_image': source_img,
                'target_image': target_img,
                'num_clusters': config.NUM_CLUSTERS,
                'num_distinct': config.NUM_DISTINCT,
                'training_epochs': config.EPOCHS,
                'temperature': config.TEMPERATURE,
            }
        )
        print(f"Model saved to: {config.get_model_path()}")

    print("\n NEXT STEPS:")
    print("   Run Example 1 again to incorporate more feedback")
    print("   Run Example 2 to apply model without retraining")
    print("\n TENSORBOARD:")
    print("   View training progress: tensorboard --logdir=runs")
    print("   Compare sessions in browser at http://127.0.0.1:6006")
    print("="*70)

    return results


def example_2_reuse_trained_model():
    """
    Reuse a trained model with FEEDBACK LOOP (faster than Example 1).

    This example demonstrates:
    1. Loading a pre-trained model from disk
    2. Applying it to triangulation without retraining
    3. Collecting feedback on color usage
    4. Fine-tuning model with feedback (150 epochs)

    This is much faster than Example 1 because we skip the initial 1000-epoch training.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Reusing Trained Model with Feedback")
    print("="*70)

    config.print_config()

    model_path = config.get_model_path()

    if not os.path.exists(model_path):
        print(f"\n Model not found at {model_path}")
        print("\n HOW TO GET A MODEL:")
        print("   1. Run Example 1 first to train a model")
        print("   2. Or download a pre-trained model")
        print(f"   3. Place it in: {model_path}")
        print("\n  Once you have a model:")
        print("   Loading takes ~5 seconds")
        print("   Applying takes ~2 minutes")
        print("   Feedback: ~3 minutes")
        print("   Total: ~5 minutes (3x faster than training!)")
        return None

    print("\n FAST MODE: Loading pre-trained model...")
    print(f"   Model file: {model_path}")

    source_img = config.TEMPLATE_IMAGE
    target_img = config.PALETTE_IMAGE

    results = imageTriangulation.pipeline_with_cnn(
        source_image_path=source_img,
        target_image_path=target_img,
        use_pretrained_model=model_path,
        threshold=50,
        density_reduction=config.DENSITY_REDUCTION,
        device='cpu',
        save_output=True
    )

    palette_rgb, palette_lab, percentages = colour.get_palette_for_cnn(
        target_img, num_clusters=config.NUM_CLUSTERS, num_distinct=config.NUM_DISTINCT
    )

    triangle_colors = results['cnn_result']['triangle_colors']

    fig = results['cnn_result']['figure']
    fig.canvas.draw()

    image_data = np.asarray(fig.canvas.buffer_rgba())
    image_data = image_data[:, :, :3]  
    image_cnn = Image.fromarray(image_data)

    image_original = Image.open(source_img)

   
    feedback_choice = input("\nDo you want to help train me with feedback? (y/n): ").strip().lower()

    if feedback_choice == 'y':
        print("\n" + "="*70)
        print("TENSORBOARD FEEDBACK COLLECTION")
        print("="*70)
        print("\n Review the TensorBoard dashboard, then provide ratings")
        print("="*70)

        template_name = config.get_image_basename(source_img)
        session_name = config.get_session_name(f"reuse_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        scores = tensorboard_feedback.get_user_feedback_tensorboard(
            palette_rgb, palette_lab, triangle_colors,
            image_original, image_cnn,
            session_name=session_name,
            template_name=template_name
        )

        print("\nFine-tuning model with feedback from all previous sessions...")

        data = CNN.prepare_training_data(
            source_img, target_img,
            num_clusters=config.NUM_CLUSTERS, num_distinct=config.NUM_DISTINCT,
            use_lab=True, device='cpu'
        )

        model = results['model']
        model_ft, loss_history_ft = CNN.fine_tune_with_feedback(
            model,
            data['source_pixels'], data['target_palette'],
            data['source_pixels_lab'], data['target_palette_lab'],
            epochs=config.FINE_TUNE_EPOCHS, batch_size=512, lr=0.0005,
            device='cpu',
            feedback_dir='feedback_data',
            log_dir='runs/fine_tuning',
            model_name=config.get_model_name(),
            template_name=template_name
        )

        CNN.save_trained_model(
            model_ft,
            model_path,
            metadata={
                'source_image': source_img,
                'target_image': target_img,
                'num_clusters': config.NUM_CLUSTERS,
                'num_distinct': config.NUM_DISTINCT,
                'fine_tuned_epochs': config.FINE_TUNE_EPOCHS,
                'temperature': config.TEMPERATURE,
                'final_loss': loss_history_ft['total'][-1] if loss_history_ft['total'] else 0
            }
        )

        print("\n" + "="*70)
        print("MODEL APPLIED & FINE-TUNED SUCCESSFULLY")
        print("="*70)
        print("\n WHAT JUST HAPPENED:")
        print("   Loaded pre-trained parameters instantly")
        print("   Applied model to create colored triangulation")
        print("   Collected your feedback on color usage")
        print(f"   Fine-tuned model for {config.FINE_TUNE_EPOCHS} epochs")
        print("\n HOW THIS WORKS:")
        print("   Model remembered how to map colors from Example 1")
        print(f"   Your feedback guided {config.FINE_TUNE_EPOCHS}-epoch fine-tuning")
        print("   Model now incorporates your preferences!")
    else:
        print("\nSkipping feedback. Model was not modified.")

    print("\n NEXT TIME:")
    print("   Run Example 1 or 2 again for more feedback cycles")
    print("   Model gets smarter with each feedback loop")
    print("\n TENSORBOARD:")
    print("   View all sessions: tensorboard --logdir=runs")
    print("   Browser: http://127.0.0.1:6006")
    print("="*70)

    return results


def example_3_plotting_menu():
    """
    Interactive plotting and visualization menu.

    Offers 8 visualization options:
    - Options 1-4: Image color analysis (RGB/LAB clouds and clusters)
    - Options 5-6: Distinct color palette display (LAB/RGB)
    - Option 7: Open TensorBoard
    - Option 8: Palette Color Wheel (LAB a*-b* plane)

    User first selects which image to analyze (template or palette),
    then can view multiple plots in a loop until they quit.
    """
    import plotting

    print("\n" + "="*70)
    print("PLOTTING & VISUALIZATION MENU")
    print("="*70)

    config.print_config()

    print("\nWhich image do you want to analyze?")
    print(f"  1. TEMPLATE_IMAGE: {config.TEMPLATE_IMAGE}")
    print(f"  2. PALETTE_IMAGE:  {config.PALETTE_IMAGE}")

    img_choice = input("\nEnter your choice (1 or 2): ").strip()
    if img_choice == '2':
        chosen_image = config.PALETTE_IMAGE
    else:
        chosen_image = config.TEMPLATE_IMAGE

    image_name = config.get_image_basename(chosen_image)
    print(f"\nAnalyzing: {chosen_image}")

    print("Loading image and computing clusters...")
    _, pixels = colour.load_image_pixels(chosen_image)
    pixels_lab = colour.convert_rgb_pixels_to_lab(pixels)

    _, centres_lab, _, percentages = colour.run_kmeans_lab(pixels_lab, config.NUM_CLUSTERS)
    centres_rgb = colour.convert_lab_centers_to_rgb(centres_lab)

    palette_lab, palette_rgb, selected_indices = colour.select_distinct_colors_lab(
        centres_lab, centres_rgb, num_to_select=config.NUM_DISTINCT
    )
    even_percentages = np.full(len(palette_rgb), 100.0 / len(palette_rgb))

    print("Ready!\n")

    while True:
        print("\n" + "-"*50)
        print(f"Plotting Menu (analyzing: {image_name})")
        print("-"*50)
        print("  1. RGB Cloud graph")
        print("  2. LAB Cloud graph")
        print("  3. RGB Clustered graph")
        print("  4. LAB Clustered graph")
        print("  5. Distinct LAB Colour Palette")
        print("  6. Distinct RGB Colour Palette")
        print("  7. Open TensorBoard")
        print("  8. Palette Color Wheel")
        print("  q. Quit")

        choice = input("\nEnter your choice: ").strip().lower()

        if choice == '1':
            print("Generating RGB Cloud...")
            plotting.plot_rgb_cloud_interactive(pixels, f'RGB Cloud - {image_name}')

        elif choice == '2':
            print("Generating LAB Cloud...")
            plotting.plot_lab_cloud_interactive(pixels_lab, pixels, f'LAB Cloud - {image_name}')

        elif choice == '3':
            print("Generating RGB Clustered graph...")
            plotting.plot_cluster_centers_3d_interactive(centres_rgb, percentages)

        elif choice == '4':
            print("Generating LAB Clustered graph...")
            plotting.plot_cluster_centers_lab_kmeans_interactive(centres_lab, percentages, centres_rgb)

        elif choice == '5':
            print("Generating Distinct LAB Colour Palette...")
            plotting.visualize_color_palette(palette_rgb, even_percentages, 'LAB')

        elif choice == '6':
            print("Generating Distinct RGB Colour Palette...")
            plotting.visualize_color_palette(palette_rgb, even_percentages, 'RGB')

        elif choice == '7':
            import subprocess
            import webbrowser
            print("Starting TensorBoard...")
            try:
                subprocess.Popen(
                    ['tensorboard', '--logdir=runs'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                webbrowser.open('http://127.0.0.1:6006')
                print("TensorBoard started at http://127.0.0.1:6006")
                print("Press Enter to continue...")
                input()
            except FileNotFoundError:
                print("  TensorBoard not found. Install with: pip install tensorboard")

        elif choice == '8':
            print("Generating Palette Color Wheel...")
            plotting.plot_palette_color_wheel(palette_lab, palette_rgb)

        elif choice == 'q':
            break

        else:
            print("Invalid choice. Please try again.")


def example_4_standard_coloring():
    """
    Standard coloring using only the template image's own colors.

    This example shows:
    1. Basic triangulation with original colors (no CNN, no palette image)
    2. Dominant color distribution from the template image
    3. Interactive RGB color cloud of the template image

    No CNN training and no palette image - this is a quick visualization tool.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Standard Coloring")
    print("="*70)

    source_img = config.TEMPLATE_IMAGE
    template_name = config.get_image_basename(source_img)

    print(f"\nTemplate image: {source_img}")
    print(f"Density reduction: {config.DENSITY_REDUCTION}")

    print("\nLoading and triangulating image...")
    imageTriangulation.setup_matplotlib()
    image_orig, image = imageTriangulation.load_image(source_img)
    image = imageTriangulation.convert_to_greyscale(image)
    image = imageTriangulation.sharpen_image(image)
    image = imageTriangulation.detect_edges(image)
    S = imageTriangulation.determine_vertices(
        image,
        threshold=50,
        density_reduction=config.DENSITY_REDUCTION
    )
    triangles = imageTriangulation.Delaunay(S)

    imageTriangulation.visualize_triangulation(S, triangles)

    print("\nColorizing triangulation...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(os.getcwd(), 'triangulatedImages', 'standardColored', template_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{template_name}_{config.DENSITY_REDUCTION}_{timestamp}.png')

    imageTriangulation.colorize_triangulation(
        S, triangles, image_orig,
        save=True,
        image_name=output_path
    )

    print("\nExtracting template image color distribution...")
    _, pixels = colour.load_image_pixels(source_img)
    pixels_lab = colour.convert_rgb_pixels_to_lab(pixels)
    _, centers_lab, _, _ = colour.run_kmeans_lab(pixels_lab, config.NUM_CLUSTERS)
    centers_rgb = colour.convert_lab_centers_to_rgb(centers_lab)
    _, palette_rgb, percentages = colour.select_distinct_colors_lab(
        centers_lab, centers_rgb, num_to_select=config.NUM_DISTINCT
    )

    print("\nDisplaying template color palette...")
    from plotting import visualize_color_palette
    visualize_color_palette(
        palette_rgb,
        percentages,
        f'Template Colors - {template_name}'
    )

    print("\nGenerating RGB color cloud...")
    from plotting import plot_rgb_cloud_interactive
    plot_rgb_cloud_interactive(
        pixels,
        f'RGB Color Space - {template_name}',
        max_points=50000
    )

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print("\nWhat you see:")
    print("  1. Triangulated image with original colors")
    print("  2. Dominant color palette from template image (bar chart)")
    print("  3. 3D RGB color cloud (interactive)")
    print(f"\nImage saved to: {output_path}")
    print("\nThis is a fast, non-ML visualization method.")
    print("Use Examples 1-2 for CNN-based color transfer.")
    print("="*70)

    return {
        'triangulation': (S, triangles),
        'palette_rgb': palette_rgb,
        'output_path': output_path
    }


def example_5_cleanup():
    """
    Clean up orphaned data directories.

    Scans templateImages/ for valid template stems, then identifies
    orphaned subdirectories across all data directories (feedback_data,
    models, runs, customColored, standardColored).

    For each data directory:
    - Shows which directories are orphaned
    - Explains what the directory contains
    - For image dirs, offers to back up to backups/ first
    - Asks for confirmation before deletion
    """
    print("\n" + "="*70)
    print("USE CASE 5: Cleanup Orphaned Data")
    print("="*70)
    print("\nThis will scan for data directories that don't correspond")
    print("to any current template image and offer to remove them.")
    print("\nNote: paletteImages/ and runs/fine_tuning/ are left alone.\n")

    import cleanup
    cleanup.cleanup_orphaned_data()


def interactive_menu():
    """
    Interactive menu for running examples.
    """
    while True:

        print("\nChoose an option:")
        print("1. Basic usage (train and apply CNN)")
        print("2. Reuse trained model (fast)")
        print("3. Plotting & visualization menu")
        print("4. Standard coloring (original colors, no CNN)")
        print("5. Cleanup orphaned data")
        print("c. Show current configuration")
        print("0. Exit")

        choice = input("\nEnter your choice (0-5, c): ").strip().lower()

        if choice == '0':
            print("Exiting.")
            break
        elif choice == '1':
            example_1_basic_usage()
        elif choice == '2':
            example_2_reuse_trained_model()
        elif choice == '3':
            example_3_plotting_menu()
        elif choice == '4':
            example_4_standard_coloring()
        elif choice == '5':
            example_5_cleanup()
        elif choice == 'c':
            config.print_config()
            errors = config.validate_config()
            if errors:
                print("\nConfiguration errors:")
                for err in errors:
                    print(f"  - {err}")
            else:
                print("\nConfiguration is valid!")
        else:
            print("Invalid choice. Please try again.")


if __name__ == '__main__':
    logo = """
 ___ _   ___   _____________ ___________   _____ _   _  _____ _____________   _____ 
|  _| | | \ \ / / ___ \ ___ \_   _|  _  \ |_   _| | | ||  ___|  _  | ___ \ \ / /_  |
| | | |_| |\ V /| |_/ / |_/ / | | | | | |   | | | |_| || |__ | | | | |_/ /\ V /  | |
| | |  _  | \ / | ___ \    /  | | | | | |   | | |  _  ||  __|| | | |    /  \ /   | |
| | | | | | | | | |_/ / |\ \ _| |_| |/ /    | | | | | || |___\ \_/ / |\ \  | |   | |
| |_\_| |_/ \_/ \____/\_| \_|\___/|___/     \_/ \_| |_/\____/ \___/\_| \_| \_/  _| |
|___|                                                                          |___|                                                                                    
"""

    print("\n" + "="*80)
    print(logo)
    print("="*80)
    print("If you're reading this, I appreciate you taking the time to see how I bridge math and art.")
    print("This is an open source AI Design Tool passion project I've been working on to bridge Music, Art, and Code.")
    print("Have fun exploring the capabilities of the system! Please submit a pull request or open an issue if you have any \nsuggestions, find any bugs, or want to contribute in any way.")
    print("\nBest, WALL-E")
    print("\n" + "-"*70)
    config.print_config()
    print("-"*70)
    interactive_menu()