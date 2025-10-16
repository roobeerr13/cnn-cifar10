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

    // Fetch history and render charts
    fetch('/history')
        .then(response => response.json())
        .then(history => {
            renderCharts(history);
        })
        .catch(error => console.error('Error fetching history:', error));

    function renderCharts(history) {
        const accuracyCtx = document.getElementById('accuracy-chart').getContext('2d');
        const lossCtx = document.getElementById('loss-chart').getContext('2d');

        const epochs = Array.from({ length: history.accuracy.length }, (_, i) => i + 1);

        new Chart(accuracyCtx, {
            type: 'line',
            data: {
                labels: epochs,
                datasets: [
                    {
                        label: 'Precisión de Entrenamiento',
                        data: history.accuracy,
                        borderColor: '#1e88e5',
                        backgroundColor: 'rgba(30, 136, 229, 0.2)',
                        fill: true
                    },
                    {
                        label: 'Precisión de Validación',
                        data: history.val_accuracy,
                        borderColor: '#f4511e',
                        backgroundColor: 'rgba(244, 81, 30, 0.2)',
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Precisión del Modelo'
                    },
                    legend: {
                        position: 'top',
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Época'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Precisión'
                        }
                    }
                }
            }
        });

        new Chart(lossCtx, {
            type: 'line',
            data: {
                labels: epochs,
                datasets: [
                    {
                        label: 'Pérdida de Entrenamiento',
                        data: history.loss,
                        borderColor: '#1e88e5',
                        backgroundColor: 'rgba(30, 136, 229, 0.2)',
                        fill: true
                    },
                    {
                        label: 'Pérdida de Validación',
                        data: history.val_loss,
                        borderColor: '#f4511e',
                        backgroundColor: 'rgba(244, 81, 30, 0.2)',
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Pérdida del Modelo'
                    },
                    legend: {
                        position: 'top',
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Época'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Pérdida'
                        }
                    }
                }
            }
        });
    }
});
