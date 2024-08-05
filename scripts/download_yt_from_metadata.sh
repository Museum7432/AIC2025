#!/bin/bash

# Check if yt-dlp is installed
if ! command -v yt-dlp &> /dev/null
then
    echo "yt-dlp could not be found. Please install it first."
    exit 1
fi

# Check if jq is installed
if ! command -v jq &> /dev/null
then
    echo "jq could not be found. Please install it first."
    exit 1
fi

# Default folders
DEFAULT_SRC_FOLDER="dataset/metadata"
DEFAULT_DST_FOLDER="dataset/videos"

# Use default folders if not provided
src_folder="${1:-$DEFAULT_SRC_FOLDER}"
dst_folder="${2:-$DEFAULT_DST_FOLDER}"

echo "Source folder: $src_folder"
echo "Destination folder: $dst_folder"

# Create destination folder if it doesn't exist
mkdir -p "$dst_folder"

# Loop through all JSON files in the source folder
for json_file in "$src_folder"/*.json; do
    # Check if there are actually any JSON files
    if [ ! -e "$json_file" ]; then
        echo "No JSON files found in $src_folder"
        exit 1
    fi

    # Extract watch_url from JSON
    watch_url=$(jq -r '.watch_url' "$json_file")
    
    # Get the base name of the JSON file (without path and extension)
    base_name=$(basename "$json_file" .json)
    
    echo "Downloading: $base_name"
    
    # Download video using yt-dlp
    yt-dlp -o "$dst_folder/${base_name}.%(ext)s" "$watch_url"
done

echo "Download complete!"