"""
cleanup.py - Orphaned Data Cleanup for HybridTheory

Scans templateImages/ for valid template stems, then identifies and removes
orphaned subdirectories in data directories (feedback_data, models, runs,
customColored, standardColored).

Uses strict case-sensitive exact matching: "Rage" does NOT match "rager".

Usage:
    import cleanup
    cleanup.cleanup_orphaned_data()
"""

import os
import shutil
from pathlib import Path


def get_template_stems(template_dir='templateImages'):
    """
    Scan templateImages/ and return a set of file stems (no extensions).

    Only includes files, not subdirectories (e.g., skips 'gifs/').

    Args:
        template_dir: Path to template images directory

    Returns:
        set of str: e.g. {'spiderman1', 'kengan', 'rager', ...}
    """
    stems = set()
    if not os.path.exists(template_dir):
        return stems

    for entry in os.listdir(template_dir):
        full_path = os.path.join(template_dir, entry)
        if os.path.isfile(full_path):
            stems.add(Path(entry).stem)

    return stems


def find_orphaned_dirs(parent_dir, valid_stems):
    """
    Find subdirectories whose names don't match any valid stem.

    Uses exact case-sensitive matching.

    Args:
        parent_dir: Directory to scan
        valid_stems: Set of valid template stems

    Returns:
        list of str: Sorted list of orphaned directory names
    """
    if not os.path.exists(parent_dir):
        return []

    orphans = []
    for entry in os.listdir(parent_dir):
        full_path = os.path.join(parent_dir, entry)
        if os.path.isdir(full_path) and entry not in valid_stems:
            orphans.append(entry)

    return sorted(orphans)


def describe_directory_contents(dir_path):
    """
    Generate a human-readable summary of a directory's contents.

    Args:
        dir_path: Full path to directory

    Returns:
        str: Description like "2 subdirectories, 5 .png images (1.2 MB)"
    """
    file_counts = {}
    total_size = 0
    num_subdirs = 0

    for root, dirs, files in os.walk(dir_path):
        if root == dir_path:
            num_subdirs = len(dirs)
        for f in files:
            ext = Path(f).suffix.lower() or '(no ext)'
            file_counts[ext] = file_counts.get(ext, 0) + 1
            try:
                total_size += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass

    # Format size
    if total_size < 1024:
        size_str = f"{total_size} B"
    elif total_size < 1024 * 1024:
        size_str = f"{total_size / 1024:.1f} KB"
    else:
        size_str = f"{total_size / (1024 * 1024):.1f} MB"

    parts = []
    if num_subdirs > 0:
        parts.append(f"{num_subdirs} subdirectory" + ("ies" if num_subdirs > 1 else ""))
    for ext, count in sorted(file_counts.items()):
        parts.append(f"{count} {ext} file" + ("s" if count > 1 else ""))

    if not parts:
        return "empty directory"

    return f"{', '.join(parts)} ({size_str})"


def backup_directory(source_dir, backup_base='backups'):
    """
    Copy a directory tree to backups/ preserving relative structure.

    Args:
        source_dir: Path to directory to back up
        backup_base: Base backup directory (default: 'backups')
    """
    rel_path = os.path.relpath(source_dir, os.getcwd())
    dest_path = os.path.join(backup_base, rel_path)

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.copytree(source_dir, dest_path)
    print(f"    Backed up: {rel_path} -> {dest_path}")


def cleanup_orphaned_data(template_dir='templateImages'):
    """
    Main cleanup orchestrator.

    Scans templateImages/ for valid stems, then checks each data directory
    for orphaned subdirectories. For each, explains the contents, shows
    orphans, and asks user to confirm deletion.

    Directories scanned:
    - feedback_data/ (feedback JSON files)
    - models/ (trained CNN model weights)
    - runs/feedback/ (TensorBoard logs)
    - triangulatedImages/customColored/ (CNN-colored outputs)
    - triangulatedImages/standardColored/ (standard-colored outputs)

    Not scanned: paletteImages/, runs/fine_tuning/
    """
    valid_stems = get_template_stems(template_dir)

    if not valid_stems:
        print(f"No template images found in {template_dir}/")
        return

    print(f"\nFound {len(valid_stems)} template images:")
    for stem in sorted(valid_stems):
        print(f"  - {stem}")

    scan_targets = [
        {
            'path': 'feedback_data',
            'description': 'feedback JSON files from your training sessions (color ratings, placement preferences)',
            'has_images': False,
        },
        {
            'path': 'models',
            'description': 'trained CNN model weights (.pth files) used for color transfer',
            'has_images': False,
        },
        {
            'path': os.path.join('runs', 'feedback'),
            'description': 'TensorBoard logs from training/feedback sessions (loss curves, visualizations)',
            'has_images': False,
        },
        {
            'path': os.path.join('triangulatedImages', 'customColored'),
            'description': 'CNN-colored triangulation output images (your custom color transfers)',
            'has_images': True,
        },
        {
            'path': os.path.join('triangulatedImages', 'standardColored'),
            'description': 'standard-colored triangulation output images (original color triangulations)',
            'has_images': True,
        },
    ]

    total_deleted = 0

    for target in scan_targets:
        parent = target['path']
        print(f"\n{'='*70}")
        print(f"Scanning: {parent}/")
        print(f"Contains: {target['description']}")
        print(f"{'='*70}")

        if not os.path.exists(parent):
            print("  Directory does not exist. Skipping.")
            continue

        orphans = find_orphaned_dirs(parent, valid_stems)

        if not orphans:
            print("  No orphaned directories found.")
            continue

        print(f"\n  Found {len(orphans)} orphaned directory(ies):")
        for orphan in orphans:
            orphan_path = os.path.join(parent, orphan)
            desc = describe_directory_contents(orphan_path)
            print(f"    - {orphan}/ -- {desc}")

        # For image directories, offer backup first
        if target['has_images']:
            backup_choice = input("\n  Back up images before deletion? (y/n): ").strip().lower()
            if backup_choice == 'y':
                print("  Backing up...")
                for orphan in orphans:
                    orphan_path = os.path.join(parent, orphan)
                    try:
                        backup_directory(orphan_path)
                    except Exception as e:
                        print(f"    Failed to back up {orphan}: {e}")

        # Confirm deletion
        confirm = input(f"\n  Delete these {len(orphans)} orphaned directory(ies)? (y/n): ").strip().lower()
        if confirm == 'y':
            for orphan in orphans:
                orphan_path = os.path.join(parent, orphan)
                try:
                    shutil.rmtree(orphan_path)
                    print(f"    Deleted: {orphan_path}")
                    total_deleted += 1
                except Exception as e:
                    print(f"    Failed to delete {orphan_path}: {e}")
        else:
            print("  Skipped.")

    print(f"\n{'='*70}")
    print(f"CLEANUP COMPLETE: Removed {total_deleted} orphaned directory(ies)")
    print(f"{'='*70}")
