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
                resultsContainer.innerHTML = `<p style="color: #ff3b3b;">${data.error}</p>`;
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
            resultsContainer.innerHTML = `<p style="color: #ff3b3b;">Ocurrió un error al procesar la solicitud.</p>`;
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
                    color: '#00e5ff'
                },
                legend: {
                    labels: {
                        color: '#e0e0e0'
                    }
                },
                tooltip: {
                    enabled: true,
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#00e5ff',
                    bodyColor: '#e0e0e0',
                    borderColor: '#00e5ff',
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
                        color: '#00e5ff'
                    },
                    ticks: {
                        color: '#e0e0e0'
                    },
                    grid: {
                        color: 'rgba(0, 229, 255, 0.2)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Valor',
                        color: '#00e5ff'
                    },
                    ticks: {
                        color: '#e0e0e0'
                    },
                    grid: {
                        color: 'rgba(0, 229, 255, 0.2)'
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
                        borderColor: '#00e5ff',
                        backgroundColor: 'rgba(0, 229, 255, 0.2)',
                        fill: true,
                        tension: 0.4,
                        pointBackgroundColor: '#00e5ff',
                        pointBorderColor: '#fff',
                        pointHoverRadius: 7,
                        pointHoverBackgroundColor: '#fff',
                        pointHoverBorderColor: '#00e5ff'
                    },
                    {
                        label: 'Precisión de Validación',
                        data: history.val_accuracy,
                        borderColor: '#ff3b3b',
                        backgroundColor: 'rgba(255, 59, 59, 0.2)',
                        fill: true,
                        tension: 0.4,
                        pointBackgroundColor: '#ff3b3b',
                        pointBorderColor: '#fff',
                        pointHoverRadius: 7,
                        pointHoverBackgroundColor: '#fff',
                        pointHoverBorderColor: '#ff3b3b'
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
                        borderColor: '#00e5ff',
                        backgroundColor: 'rgba(0, 229, 255, 0.2)',
                        fill: true,
                        tension: 0.4,
                        pointBackgroundColor: '#00e5ff',
                        pointBorderColor: '#fff',
                        pointHoverRadius: 7,
                        pointHoverBackgroundColor: '#fff',
                        pointHoverBorderColor: '#00e5ff'
                    },
                    {
                        label: 'Pérdida de Validación',
                        data: history.val_loss,
                        borderColor: '#ff3b3b',
                        backgroundColor: 'rgba(255, 59, 59, 0.2)',
                        fill: true,
                        tension: 0.4,
                        pointBackgroundColor: '#ff3b3b',
                        pointBorderColor: '#fff',
                        pointHoverRadius: 7,
                        pointHoverBackgroundColor: '#fff',
                        pointHoverBorderColor: '#ff3b3b'
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
