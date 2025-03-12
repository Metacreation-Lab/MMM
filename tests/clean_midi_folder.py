from pathlib import Path

# Paths
folder_path = Path("test_midis")
valid_files_path = Path("valid_files.txt")

# Read the valid files into a set
with valid_files_path.open("w") as f:
    valid_files = {line.strip() for line in f}

# List all files in the folder
for file_path in folder_path.iterdir():
    # Check if it's a file and if it's NOT in the valid files list
    if file_path.is_file() and file_path.name not in valid_files:
        file_path.unlink()  # Equivalent to os.remove()
        print(f"Deleted: {file_path.name}")

print("Cleanup complete!")
