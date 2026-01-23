import os
import shutil
import zipfile

def create_zip():
    # Config
    output_filename = 'tiur_tricks_colab_v4.zip'
    staging_dir = 'temp_staging_area'
    target_root = 'tiur_tricks_colab'  # The folder name inside the zip
    
    # Files/Dirs to include
    include_dirs = ['tiur_tricks']
    include_files = ['README.md', 'README_COLAB.md', 'run_set1.py']
    
    # Add notebooks
    for f in os.listdir('.'):
        if f.endswith('.ipynb'):
            include_files.append(f)

    # Clean up any previous run
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    if os.path.exists(output_filename):
        os.remove(output_filename)

    # Create staging structure
    staging_path = os.path.join(staging_dir, target_root)
    os.makedirs(staging_path)

    print(f"Preparing files in {staging_path}...")

    # Copy directories
    for d in include_dirs:
        if os.path.exists(d):
            shutil.copytree(d, os.path.join(staging_path, d))
        else:
            print(f"Warning: Directory {d} not found.")

    # Copy files
    for f in include_files:
        if os.path.exists(f):
            shutil.copy2(f, staging_path)
        else:
            print(f"Warning: File {f} not found.")

    # Zip it up
    print(f"Creating {output_filename}...")
    shutil.make_archive(output_filename.replace('.zip', ''), 'zip', staging_dir)
    
    # Verify file name (make_archive adds .zip if not present, but we handle it)
    # If make_archive created .zip.zip, rename it (unlikely with this usage)
    
    # Cleanup
    shutil.rmtree(staging_dir)
    print(f"Done! Upload '{output_filename}' to your Google Drive.")

if __name__ == '__main__':
    create_zip()
