// Gallery logic for Curiosity Books Gallery

let galleryImages = [];
let currentImageIndex = 0;

// Fisher-Yates shuffle
function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}

function initGallery() {
    // Only initialize once
    if (galleryImages.length > 0) return;

    fetch('gallery_images.json')
        .then(response => response.json())
        .then(images => {
            galleryImages = images;
            // Removed shuffleArray to keep chronological (alphabetical) order
            renderThumbnails();
            
            // Move lightbox out of the <article> to avoid CSS transform relative positioning bugs
            const lightbox = document.getElementById('gallery-lightbox');
            if (lightbox && lightbox.parentNode !== document.body) {
                document.body.appendChild(lightbox);
            }
        })
        .catch(err => console.error('Error loading gallery images:', err));

    setupLightboxEvents();
}

function renderThumbnails() {
    const container = document.getElementById('gallery-thumbnails');
    container.innerHTML = '';
    
    galleryImages.forEach((src, index) => {
        const thumb = document.createElement('div');
        thumb.className = 'gallery-thumbnail';
        thumb.style.cursor = 'pointer';
        thumb.style.borderRadius = '4px';
        thumb.style.boxShadow = '0 2px 5px rgba(0,0,0,0.2)';
        thumb.style.aspectRatio = '1 / 1';
        thumb.style.transition = 'transform 0.2s';
        thumb.style.overflow = 'hidden';
        
        const img = document.createElement('img');
        img.src = src;
        img.loading = 'lazy'; // Huge performance boost by deferring offscreen images
        img.style.width = '100%';
        img.style.height = '100%';
        img.style.objectFit = 'cover';
        img.style.display = 'block';
        
        thumb.appendChild(img);
        
        thumb.onmouseover = () => thumb.style.transform = 'scale(1.05)';
        thumb.onmouseout = () => thumb.style.transform = 'scale(1)';
        
        thumb.onclick = () => openLightbox(index);
        
        container.appendChild(thumb);
    });
}

function openLightbox(index) {
    currentImageIndex = index;
    const lightbox = document.getElementById('gallery-lightbox');
    const lightboxImg = document.getElementById('lightbox-img');
    
    lightboxImg.src = galleryImages[currentImageIndex];
    lightbox.style.display = 'flex';
    document.body.style.overflow = 'hidden'; // prevent scrolling behind
}

function closeLightbox() {
    document.getElementById('gallery-lightbox').style.display = 'none';
    document.body.style.overflow = '';
}

function nextLightboxImage(e) {
    if (e) e.stopPropagation();
    currentImageIndex = (currentImageIndex + 1) % galleryImages.length;
    document.getElementById('lightbox-img').src = galleryImages[currentImageIndex];
}

function prevLightboxImage(e) {
    if (e) e.stopPropagation();
    currentImageIndex = (currentImageIndex - 1 + galleryImages.length) % galleryImages.length;
    document.getElementById('lightbox-img').src = galleryImages[currentImageIndex];
}

function setupLightboxEvents() {
    const lightbox = document.getElementById('gallery-lightbox');
    
    // Close when clicking outside the image
    lightbox.addEventListener('click', (e) => {
        if (e.target === lightbox) {
            closeLightbox();
        }
    });

    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
        if (lightbox.style.display === 'flex') {
            if (e.key === 'ArrowRight') {
                nextLightboxImage();
                e.stopPropagation();
            } else if (e.key === 'ArrowLeft') {
                prevLightboxImage();
                e.stopPropagation();
            } else if (e.key === 'Escape') {
                closeLightbox();
                e.stopPropagation();
            }
        }
    }, true); // use capture to ensure we get it before the Dimension theme handles ESC
}

// Robust initialization to handle dynamic includeHTML() injection
function initializeGalleryWhenReady() {
    if (document.getElementById('gallery-thumbnails')) {
        initGallery();
    } else {
        setTimeout(initializeGalleryWhenReady, 100);
    }
}

// Start polling
initializeGalleryWhenReady();
