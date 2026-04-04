// static/js/main.js
document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('file-upload');
    const previewContainer = document.getElementById('preview-container');
    const previewImage = document.getElementById('image-preview');
    const uploadForm = document.getElementById('upload-form');
    const loader = document.getElementById('loader');
    const resultsSection = document.getElementById('results-section');

    // 1. Show image preview immediately when selected
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            const file = this.files[0];
            if (file) {
                const reader = new FileReader();
                reader.addEventListener('load', function() {
                    previewImage.setAttribute('src', this.result);
                    previewContainer.style.display = 'block';
                });
                reader.readAsDataURL(file);
            }
        });
    }

    // 2. Show loading spinner on form submit
    if (uploadForm) {
        uploadForm.addEventListener('submit', function() {
            // Hide previous results if they exist
            if (resultsSection) {
                resultsSection.style.display = 'none';
            }
            // Show the loading spinner
            loader.style.display = 'block';
        });
    }
});