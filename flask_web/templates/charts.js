document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('file-input');
    const previewImage = document.getElementById('preview-image');
    const predictButton = document.getElementById('predict-button');
    const resultsContainer = document.getElementById('results');

    fileInput.addEventListener('change', () => {
        const file = fileInput.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImage.src = e.target.result;
                previewImage.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }
    });

    predictButton.addEventListener('click', () => {
        const file = fileInput.files[0];
        if (!file) {
            alert('Por favor, selecciona una imagen.');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            resultsContainer.innerHTML = '';
            if (data.error) {
                resultsContainer.innerHTML = `<p style="color: red;">${data.error}</p>`;
            } else {
                const sortedResults = Object.entries(data).sort((a, b) => b[1] - a[1]);
                sortedResults.forEach(([className, probability]) => {
                    const resultElement = document.createElement('p');
                    resultElement.textContent = `${className}: ${(probability * 100).toFixed(2)}%`;
                    resultsContainer.appendChild(resultElement);
                });
            }
        })
        .catch(error => {
            console.error('Error:', error);
            resultsContainer.innerHTML = `<p style="color: red;">Ocurrió un error al procesar la solicitud.</p>`;
        });
    });
});
