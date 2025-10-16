document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('file-input');
    const previewImage = document.getElementById('preview-image');
    const predictButton = document.getElementById('predict-button');
    const resultsContainer = document.getElementById('results');
    const datasetGrid = document.querySelector('.dataset-grid');

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
                resultsContainer.innerHTML = `<p style="color: #ff3b30;">${data.error}</p>`;
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
            resultsContainer.innerHTML = `<p style="color: #ff3b30;">Ocurrió un error al procesar la solicitud.</p>`;
        });
    });

    // Fetch history and render charts
    fetch('/history')
        .then(response => response.json())
        .then(history => {
            renderCharts(history);
        })
        .catch(error => console.error('Error fetching history:', error));

    // Fetch dataset subset and display images
    fetch('/dataset_subset')
        .then(response => response.json())
        .then(data => {
            displayDataset(data);
        })
        .catch(error => console.error('Error fetching dataset subset:', error));

    function displayDataset(data) {
        datasetGrid.innerHTML = '';
        data.images.forEach((img_str, index) => {
            const item = document.createElement('div');
            item.classList.add('dataset-item');

            const img = document.createElement('img');
            img.src = `data:image/png;base64,${img_str}`;
            img.alt = data.labels[index];

            const label = document.createElement('p');
            label.textContent = data.labels[index];

            item.appendChild(img);
            item.appendChild(label);
            datasetGrid.appendChild(item);
        });
    }

    function renderCharts(history) {
        const accuracyCtx = document.getElementById('accuracy-chart').getContext('2d');
        const lossCtx = document.getElementById('loss-chart').getContext('2d');

        const epochs = Array.from({ length: history.accuracy.length }, (_, i) => i + 1);

        const chartOptions = {
            responsive: true,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                title: {
                    display: true,
                    font: {
                        size: 18
                    },
                    color: '#1d1d1f'
                },
                legend: {
                    labels: {
                        color: '#1d1d1f'
                    }
                },
                tooltip: {
                    enabled: true,
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#ffffff',
                    bodyColor: '#ffffff',
                    borderColor: '#d1d1d6',
                    borderWidth: 1
                },
                zoom: {
                    pan: {
                        enabled: false, // Disabled
                    },
                    zoom: {
                        wheel: {
                            enabled: false, // Disabled
                        },
                        pinch: {
                            enabled: false, // Disabled
                        },
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Época',
                        color: '#1d1d1f'
                    },
                    ticks: {
                        color: '#1d1d1f'
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Valor',
                        color: '#1d1d1f'
                    },
                    ticks: {
                        color: '#1d1d1f'
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                }
            }
        };

        new Chart(accuracyCtx, {
            type: 'line',
            data: {
                labels: epochs,
                datasets: [
                    {
                        label: 'Precisión de Entrenamiento',
                        data: history.accuracy,
                        borderColor: '#007aff',
                        backgroundColor: 'rgba(0, 122, 255, 0.1)',
                        fill: true,
                        tension: 0.4,
                        pointBackgroundColor: '#007aff',
                        pointBorderColor: '#ffffff',
                        pointHoverRadius: 7,
                        pointHoverBackgroundColor: '#ffffff',
                        pointHoverBorderColor: '#007aff'
                    },
                    {
                        label: 'Precisión de Validación',
                        data: history.val_accuracy,
                        borderColor: '#ff3b30',
                        backgroundColor: 'rgba(255, 59, 48, 0.1)',
                        fill: true,
                        tension: 0.4,
                        pointBackgroundColor: '#ff3b30',
                        pointBorderColor: '#ffffff',
                        pointHoverRadius: 7,
                        pointHoverBackgroundColor: '#ffffff',
                        pointHoverBorderColor: '#ff3b30'
                    }
                ]
            },
            options: {
                ...chartOptions,
                plugins: {
                    ...chartOptions.plugins,
                    title: {
                        ...chartOptions.plugins.title,
                        text: 'Precisión del Modelo'
                    }
                },
                 scales: {
                    ...chartOptions.scales,
                    y: {
                        ...chartOptions.scales.y,
                        title: {
                            ...chartOptions.scales.y.title,
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
                        borderColor: '#007aff',
                        backgroundColor: 'rgba(0, 122, 255, 0.1)',
                        fill: true,
                        tension: 0.4,
                        pointBackgroundColor: '#007aff',
                        pointBorderColor: '#ffffff',
                        pointHoverRadius: 7,
                        pointHoverBackgroundColor: '#ffffff',
                        pointHoverBorderColor: '#007aff'
                    },
                    {
                        label: 'Pérdida de Validación',
                        data: history.val_loss,
                        borderColor: '#ff3b30',
                        backgroundColor: 'rgba(255, 59, 48, 0.1)',
                        fill: true,
                        tension: 0.4,
                        pointBackgroundColor: '#ff3b30',
                        pointBorderColor: '#ffffff',
                        pointHoverRadius: 7,
                        pointHoverBackgroundColor: '#ffffff',
                        pointHoverBorderColor: '#ff3b30'
                    }
                ]
            },
            options: {
                ...chartOptions,
                plugins: {
                    ...chartOptions.plugins,
                    title: {
                        ...chartOptions.plugins.title,
                        text: 'Pérdida del Modelo'
                    }
                },
                scales: {
                    ...chartOptions.scales,
                    y: {
                        ...chartOptions.scales.y,
                        title: {
                            ...chartOptions.scales.y.title,
                            text: 'Pérdida'
                        }
                    }
                }
            }
        });
    }
});
