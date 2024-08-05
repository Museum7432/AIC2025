#!/bin/bash

# Function to download files from a folder
download_files() {
    folder_id="$1"
    output_folder="$2"

    # Create the output folder if it doesn't exist
    mkdir -p "$output_folder"

    # Change to the output folder
    cd "$output_folder" || exit

    # Download the folder using gdown
    gdown "https://drive.google.com/drive/folders/$folder_id" --folder -c --remaining-ok

    echo "Download completed. Files are in $output_folder"

    # Change back to the original directory
    cd - > /dev/null
}


# Main script
folder_id="1pK99oz18_SKuTAg6M7Ybcq0o-ovD4fpa"
output_folder="dataset/metadata"

download_files "$folder_id" "$output_folder"