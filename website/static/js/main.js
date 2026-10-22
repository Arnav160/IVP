/**
 * Main JavaScript for CamoSeg Website
 * Handles navigation, uploads, predictions, toasts, and UI interactions
 */

// ========================================
// Utility Functions
// ========================================

function showToast(message, type = 'info', duration = 4000) {
    const container = getToastContainer();
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <span class="toast-message">${message}</span>
        <button class="toast-close" aria-label="Dismiss">&times;</button>
    `;
    container.appendChild(toast);

    const closeBtn = toast.querySelector('.toast-close');
    closeBtn.addEventListener('click', () => toast.remove());

    setTimeout(() => {
        if (toast.parentNode) toast.remove();
    }, duration);
}

function getToastContainer() {
    let container = document.querySelector('.toast-container');
    if (!container) {
        container = document.createElement('div');
        container.className = 'toast-container';
        document.body.appendChild(container);
    }
    return container;
}

function formatNumber(num, decimals = 2) {
    return Number(num).toFixed(decimals);
}

function debounce(fn, delay) {
    let timeoutId;
    return (...args) => {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => fn(...args), delay);
    };
}

// ========================================
// Navigation Active State
// ========================================

document.addEventListener('DOMContentLoaded', () => {
    const currentPath = window.location.pathname;
    document.querySelectorAll('.nav-links a').forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        }
    });
});

// ========================================
// File Upload Handling (Predict Page)
// ========================================

function initFileUpload() {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');

    if (!uploadArea || !fileInput) return;

    // Click to upload
    uploadArea.addEventListener('click', () => fileInput.click());

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        if (files.length) handleFiles(files);
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) handleFiles(e.target.files);
    });

    async function handleFiles(files) {
        const file = files[0];
        if (!file.type.startsWith('image/')) {
            showToast('Please select an image file (PNG, JPG, WebP)', 'error');
            return;
        }

        if (file.size > 10 * 1024 * 1024) {
            showToast('File size must be less than 10MB', 'error');
            return;
        }

        // Show preview immediately
        showPreview(file);

        // TODO: Send to backend for prediction
        // await predictImage(file);
    }

    function showPreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            uploadArea.style.display = 'none';
            previewContainer.style.display = 'grid';
            previewContainer.innerHTML = `
                <div class="preview-card">
                    <div class="preview-header">Original Image</div>
                    <img class="preview-image" src="${e.target.result}" alt="Uploaded image">
                </div>
                <div class="preview-card">
                    <div class="preview-header">Predicted Mask</div>
                    <div class="empty-state">
                        <svg class="empty-state-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                            <rect x="3" y="3" width="18" height="18" rx="2"/>
                            <path d="M9 12l2 2 4-4"/>
                        </svg>
                        <p>Click "Run Segmentation" to generate prediction</p>
                    </div>
                </div>
                <div class="preview-card">
                    <div class="preview-header">Overlay</div>
                    <div class="empty-state">
                        <svg class="empty-state-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                            <rect x="3" y="3" width="18" height="18" rx="2"/>
                            <path d="M9 12l2 2 4-4"/>
                        </svg>
                        <p>Overlay will appear after prediction</p>
                    </div>
                </div>
            `;

            // Add run button
            const runBtn = document.createElement('button');
            runBtn.className = 'btn btn-primary btn-lg';
            runBtn.style.marginTop = 'var(--space-lg)';
            runBtn.style.gridColumn = '1 / -1';
            runBtn.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> Run Segmentation';
            runBtn.addEventListener('click', () => predictImage(file));
            previewContainer.appendChild(runBtn);
        };
        reader.readAsDataURL(file);
    }

    async function predictImage(file) {
        const formData = new FormData();
        formData.append('image', file);

        const runBtn = previewContainer.querySelector('.btn-primary');
        const originalBtnText = runBtn.innerHTML;
        runBtn.disabled = true;
        runBtn.innerHTML = '<div class="loading-spinner" style="width:20px;height:20px;border-width:2px"></div> Processing...';

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.status === 'success') {
                // Update preview with actual results
                const previewCards = previewContainer.querySelectorAll('.preview-card');

                // Original image (already shown, but update with processed version)
                previewCards[0].innerHTML = `
                    <div class="preview-header">Original Image (352×352)</div>
                    <img class="preview-image" src="${data.rgb}" alt="Processed input">
                `;

                // Predicted mask
                previewCards[1].innerHTML = `
                    <div class="preview-header">Predicted Mask</div>
                    <img class="preview-image mask" src="${data.mask}" alt="Predicted camouflage mask">
                `;

                // Overlay
                previewCards[2].innerHTML = `
                    <div class="preview-header">Overlay (Mask on Original)</div>
                    <img class="preview-image" src="${data.overlay}" alt="Overlay">
                `;

                showToast('Segmentation complete!', 'success');
            } else {
                showToast(data.message || 'Prediction failed', 'error');
            }
        } catch (error) {
            console.error('Prediction error:', error);
            showToast('Failed to connect to server', 'error');
        } finally {
            runBtn.disabled = false;
            runBtn.innerHTML = originalBtnText;
        }
    }
}

// ========================================
// Comparison Slider (Results Page)
// ========================================

function initComparisonSlider() {
    const sliders = document.querySelectorAll('.comparison-slider');

    sliders.forEach(slider => {
        const handle = slider.querySelector('.comparison-handle');
        const overlay = slider.querySelector('.comparison-overlay');
        let isDragging = false;

        function updateSlider(clientX) {
            const rect = slider.getBoundingClientRect();
            let position = ((clientX - rect.left) / rect.width) * 100;
            position = Math.max(0, Math.min(100, position));
            overlay.style.width = `${position}%`;
            handle.style.left = `${position}%`;
        }

        handle.addEventListener('mousedown', (e) => {
            isDragging = true;
            e.preventDefault();
        });

        document.addEventListener('mousemove', (e) => {
            if (isDragging) updateSlider(e.clientX);
        });

        document.addEventListener('mouseup', () => {
            isDragging = false;
        });

        // Touch support
        handle.addEventListener('touchstart', (e) => {
            isDragging = true;
        }, { passive: true });

        document.addEventListener('touchmove', (e) => {
            if (isDragging) updateSlider(e.touches[0].clientX);
        }, { passive: true });

        document.addEventListener('touchend', () => {
            isDragging = false;
        });

        // Click on slider to move handle
        slider.addEventListener('click', (e) => {
            if (e.target !== handle) updateSlider(e.clientX);
        });
    });
}

// ========================================
// Smooth Scroll for Anchor Links
// ========================================

document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                e.preventDefault();
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });
});

// ========================================
// Initialize Page-Specific Features
// ========================================

document.addEventListener('DOMContentLoaded', () => {
    initFileUpload();
    initComparisonSlider();

    // Add entrance animations
    const animatedElements = document.querySelectorAll('.card, .metric-card, .dataset-card, .chart-container');
    animatedElements.forEach((el, i) => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = `opacity 0.5s ease ${i * 50}ms, transform 0.5s ease ${i * 50}ms`;
        requestAnimationFrame(() => {
            el.style.opacity = '1';
            el.style.transform = 'translateY(0)';
        });
    });
});

// ========================================
// Model Config Badge Helper (for Results page)
// ========================================

function createModelBadge(config) {
    const configs = {
        'rgb': { label: 'RGB', class: 'config-rgb' },
        'canny': { label: '+Canny', class: 'config-canny' },
        'depth': { label: '+Depth', class: 'config-depth' },
        'combined': { label: '+Canny+Depth', class: 'config-combined' }
    };

    return configs[config] ? `<span class="model-config ${configs[config].class}">${configs[config].label}</span>` : '';
}

// Export for use in inline scripts
window.CamoSeg = {
    showToast,
    formatNumber,
    createModelBadge
};