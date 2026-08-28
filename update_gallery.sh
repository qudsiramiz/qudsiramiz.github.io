#!/bin/bash
# Script to generate a JSON list of all images in the library_images folder.

cd "$(dirname "$0")"
IMAGE_DIR="images/library_images"
OUTPUT_FILE="gallery_images.json"

echo "[" > "$OUTPUT_FILE"
FIRST=true

for file in "$IMAGE_DIR"/*; do
    if [[ -f "$file" ]]; then
        # Check if it's an image file by extension
        if [[ "$file" =~ \.(jpg|jpeg|png|gif|webp|JPG|JPEG|PNG|GIF|heic|HEIC)$ ]]; then
            if [ "$FIRST" = true ]; then
                FIRST=false
            else
                echo "," >> "$OUTPUT_FILE"
            fi
            echo -n "  \"$file\"" >> "$OUTPUT_FILE"
        fi
    fi
done

echo "" >> "$OUTPUT_FILE"
echo "]" >> "$OUTPUT_FILE"

echo "Generated $OUTPUT_FILE with $(grep -c '"images/' "$OUTPUT_FILE") images."
