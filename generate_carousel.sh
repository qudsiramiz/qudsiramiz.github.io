#!/bin/bash
echo '<div class="carousel-container" style="position: relative; max-width: 100%; margin: 2em auto; overflow: hidden; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">'
echo '    <button id="prevBtn" class="button" style="position: absolute; left: 10px; top: 50%; transform: translateY(-50%); z-index: 10; padding: 0.5em 1em; min-width: 0; border-radius: 50%; opacity: 0.7;">&#10094;</button>'
echo '    <div id="carousel" style="display: flex; overflow-x: auto; scroll-behavior: smooth; scroll-snap-type: x mandatory; gap: 0;">'

for img in images/library_images/*; do
    # Only include valid image files
    if [[ $img =~ \.(jpg|jpeg|png|gif|JPG|JPEG|PNG|GIF|heic|HEIC)$ ]]; then
        echo "        <div style=\"scroll-snap-align: center; flex: 0 0 100%; width: 100%; display: flex; justify-content: center; align-items: center; background-color: #111;\">"
        echo "            <img src=\"$img\" loading=\"lazy\" style=\"max-width: 100%; max-height: 700px; object-fit: contain;\" alt=\"Bookshelf Image\">"
        echo "        </div>"
    fi
done

echo '    </div>'
echo '    <button id="nextBtn" class="button" style="position: absolute; right: 10px; top: 50%; transform: translateY(-50%); z-index: 10; padding: 0.5em 1em; min-width: 0; border-radius: 50%; opacity: 0.7;">&#10095;</button>'
echo '</div>'
echo '<style>'
echo '    #carousel::-webkit-scrollbar { display: none; }'
echo '    #carousel { -ms-overflow-style: none; scrollbar-width: none; }'
echo '    .carousel-container:hover button { opacity: 1 !important; transition: opacity 0.3s; }'
echo '</style>'
echo '<script>'
echo '    const carousel = document.getElementById("carousel");'
echo '    document.getElementById("nextBtn").addEventListener("click", () => {'
echo '        carousel.scrollBy({ left: carousel.clientWidth, behavior: "smooth" });'
echo '    });'
echo '    document.getElementById("prevBtn").addEventListener("click", () => {'
echo '        carousel.scrollBy({ left: -carousel.clientWidth, behavior: "smooth" });'
echo '    });'
echo '</script>'
