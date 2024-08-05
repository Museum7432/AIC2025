#!/bin/bash

# Set source folder
SOURCE_FOLDER="dataset/Zip_keyframes"
DST_FOLDER="dataset/key_frames"
# Check if source folder exists
if [ ! -d "$SOURCE_FOLDER" ]; then
    echo "Error: Source folder $SOURCE_FOLDER does not exist."
    exit 1
fi

# Loop through all zip files in the source folder
for zip_file in "$SOURCE_FOLDER"/Keyframes_L*.zip; do
    # Check if there are any matching zip files
    if [ -e "$zip_file" ]; then
        # Extract the L number from the filename
        l_number=$(basename "$zip_file" .zip | sed 's/Keyframes_//')
        
        # Create main folder for this L number
        main_folder="$DST_FOLDER/$l_number"
        mkdir -p "$main_folder"
        
        # Extract the zip file to a temporary directory
        temp_dir=$(mktemp -d)
        unzip -q "$zip_file" -d "$temp_dir"
        
        # Move contents to the correct structure
        for subfolder in "$temp_dir"/keyframes/*; do
            if [ -d "$subfolder" ]; then
                subfolder_name=$(basename "$subfolder")
                mv "$subfolder" "$main_folder/"
                echo "Extracted: $subfolder_name to $main_folder/$subfolder_name"
            fi
        done
        
        # Clean up temporary directory
        rm -rf "$temp_dir"
    fi
done

echo "All zip files have been extracted to their respective folders in dataset/"