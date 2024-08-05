#!/bin/bash

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null
then
    echo "ffmpeg could not be found. Please install it first."
    exit 1
fi

# Check if parallel is installed
if ! command -v parallel &> /dev/null
then
    echo "GNU Parallel could not be found. Please install it first."
    exit 1
fi

# Default folders
DEFAULT_SRC_FOLDER="dataset/videos"
DEFAULT_DST_FOLDER="dataset/audio"

# Use default folders if not provided
src_folder="${1:-$DEFAULT_SRC_FOLDER}"
dst_folder="${2:-$DEFAULT_DST_FOLDER}"

echo "Source folder: $src_folder"
echo "Destination folder: $dst_folder"

# Create destination folder if it doesn't exist
mkdir -p "$dst_folder"

# Function to extract audio from a single video
extract_audio() {
    video_file="$1"
    dst_folder="$2"
    base_name=$(basename "$video_file")
    file_name="${base_name%.*}"
    echo "Extracting audio from: $base_name"
    ffmpeg -i "$video_file" -q:a 0 -map a "$dst_folder/${file_name}.mp3" -y
}

export -f extract_audio

# Use GNU Parallel to process videos in parallel
find "$src_folder" -type f | parallel -j+0 extract_audio {} "$dst_folder"

echo "Audio extraction complete!"



