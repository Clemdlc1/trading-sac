/**
 * validation.js
 * Gestion de l'interface de validation
 */

const socket = io();

// État de la validation
const validationState = {
    isRunning: false,
    currentResults: null,
    charts: {
        equity: null,
        returns: null
    }
};

// ============================================================================
// INITIALISATION
// ============================================================================

document.addEventListener('DOMContentLoaded', function() {
    console.log('Page validation initialisée');
    
    // Charger les modèles disponibles
    loadAvailableModels();
    
    // Charger l'historique des validations
    loadValidationHistory();
    
    // Setup event listeners
    setupEventListeners();
    
    // Initialiser les graphiques
    initCharts();
});

// ============================================================================
// SOCKET.IO HANDLERS
// ============================================================================

socket.on('validation_complete', function(data) {
    console.log('Validation terminée:', data);
    validationState.isRunning = false;
    validationState.currentResults = data;
    
    hideValidationProgress();
    displayResults(data);
    showNotification('Succès', 'Validation terminée', 'success');
    
    // Recharger l'historique
    setTimeout(() => loadValidationHistory(), 1000);
});

socket.on('validation_error', function(data) {
    console.error('Erreur validation:', data);
    validationState.isRunning = false;
    
    hideValidationProgress();
    showNotification('Erreur', data.error, 'danger');
});

// ============================================================================
// CHARGEMENT DES DONNÉES
// ============================================================================

async function loadAvailableModels() {
    try {
        const response = await fetch('/api/models/list');
        const data = await response.json();
        
        const select = document.getElementById('model-select');
        select.innerHTML = '<option value="">-- Choisir un modèle --</option>';
        
        // Ajouter les modèles de production
        if (data.production && data.production.length > 0) {
            const optgroup = document.createElement('optgroup');
            optgroup.label = 'Production';
            
            data.production.forEach(model => {
                const option = document.createElement('option');
                option.value = model.path;
                option.textContent = model.name;
                optgroup.appendChild(option);
            });
            
            select.appendChild(optgroup);
        }
        
        // Ajouter les checkpoints
        if (data.checkpoints && data.checkpoints.length > 0) {
            const optgroup = document.createElement('optgroup');
            optgroup.label = 'Checkpoints';
            
            data.checkpoints.forEach(model => {
                const option = document.createElement('option');
                option.value = model.path;
                option.textContent = model.name;
                optgroup.appendChild(option);
            });
            
            select.appendChild(optgroup);
        }
    } catch (error) {
        console.error('Erreur chargement modèles:', error);
        showNotification('Erreur', 'Impossible de charger les modèles', 'danger');
    }
}

async function loadValidationHistory() {
    // TODO: Implémenter l'API pour charger l'historique
    const tbody = document.querySelector('#validation-history-table tbody');
    tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">Aucune validation dans l\'historique</td></tr>';
}

// ============================================================================
// EVENT LISTENERS
// ============================================================================

function setupEventListeners() {
    // Formulaire de validation
    document.getElementById('validation-form').addEventListener('submit', async function(e) {
        e.preventDefault();
        await startValidation();
    });
    
    // Bouton arrêt validation
    document.getElementById('btn-stop-validation').addEventListener('click', function() {
        stopValidation();
    });
    
    // Boutons d'export
    document.getElementById('btn-export-results')?.addEventListener('click', function() {
        exportResults();
    });
    
    document.getElementById('btn-view-report')?.addEventListener('click', function() {
        viewFullReport();
    });
}

// ============================================================================
// VALIDATION
// ============================================================================

async function startValidation() {
    const modelPath = document.getElementById('model-select').value;
    const validationType = document.getElementById('validation-type').value;
    const nFolds = parseInt(document.getElementById('n-folds').value);
    
    if (!modelPath) {
        showNotification('Erreur', 'Veuillez sélectionner un modèle', 'warning');
        return;
    }
    
    try {
        const response = await fetch('/api/validation/run', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model_path: modelPath,
                validation_type: validationType,
                n_folds: nFolds
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            validationState.isRunning = true;
            showValidationProgress();
            showNotification('Info', data.message, 'info');
        } else {
            showNotification('Erreur', data.error, 'danger');
        }
    } catch (error) {
        console.error('Erreur démarrage validation:', error);
        showNotification('Erreur', 'Impossible de démarrer la validation', 'danger');
    }
}

function stopValidation() {
    // TODO: Implémenter l'arrêt de la validation
    validationState.isRunning = false;
    hideValidationProgress();
    showNotification('Info', 'Validation arrêtée', 'info');
}

// ============================================================================
// INTERFACE DE PROGRESSION
// ============================================================================

function showValidationProgress() {
    document.getElementById('validation-progress').style.display = 'block';
    
    const progressBar = document.getElementById('val-progress-bar');
    progressBar.style.width = '0%';
    progressBar.textContent = '0%';
    
    const info = document.getElementById('validation-info');
    info.textContent = 'Validation en cours...';
}

function hideValidationProgress() {
    document.getElementById('validation-progress').style.display = 'none';
}

// ============================================================================
// AFFICHAGE DES RÉSULTATS
// ============================================================================

function displayResults(data) {
    // Masquer le message "pas de résultats"
    document.getElementById('no-results').style.display = 'none';
    
    // Afficher le container de résultats
    const container = document.getElementById('results-container');
    container.style.display = 'block';
    
    // Mettre à jour les métriques
    if (data.statistics) {
        document.getElementById('val-sharpe').textContent = 
            data.statistics.sharpe_ratio?.toFixed(2) || '-';
        
        document.getElementById('val-dd').textContent = 
            data.statistics.max_drawdown ? 
            (data.statistics.max_drawdown * 100).toFixed(2) + '%' : '-';
        
        document.getElementById('val-winrate').textContent = 
            data.statistics.win_rate ? 
            (data.statistics.win_rate * 100).toFixed(2) + '%' : '-';
        
        document.getElementById('val-pf').textContent = 
            data.statistics.profit_factor?.toFixed(2) || '-';
    }
    
    // Mettre à jour les graphiques
    if (data.results) {
        updateEquityChart(data.results);
        updateReturnsChart(data.results);
    }
    
    // Mettre à jour les tests statistiques
    if (data.statistics) {
        updateStatisticalTests(data.statistics);
    }
}

function updateEquityChart(results) {
    if (!validationState.charts.equity) return;
    
    // Préparer les données pour chaque fold
    const datasets = [];
    
    if (results.folds) {
        results.folds.forEach((fold, index) => {
            if (fold.equity_curve) {
                datasets.push({
                    label: `Fold ${index + 1}`,
                    data: fold.equity_curve,
                    borderColor: getColorForIndex(index),
                    backgroundColor: getColorForIndex(index, 0.1),
                    tension: 0.1,
                    fill: false
                });
            }
        });
    }
    
    validationState.charts.equity.data.datasets = datasets;
    validationState.charts.equity.update();
}

function updateReturnsChart(results) {
    if (!validationState.charts.returns) return;
    
    // Calculer la distribution des returns
    let allReturns = [];
    
    if (results.folds) {
        results.folds.forEach(fold => {
            if (fold.returns) {
                allReturns = allReturns.concat(fold.returns);
            }
        });
    }
    
    // Créer un histogramme
    const bins = 50;
    const histogram = createHistogram(allReturns, bins);
    
    validationState.charts.returns.data.labels = histogram.labels;
    validationState.charts.returns.data.datasets[0].data = histogram.values;
    validationState.charts.returns.update();
}

function updateStatisticalTests(statistics) {
    const tbody = document.getElementById('stats-tests-tbody');
    tbody.innerHTML = '';
    
    const tests = [
        {
            name: 'Deflated Sharpe Ratio',
            value: statistics.dsr?.toFixed(3) || '-',
            threshold: '> 0.05',
            passed: statistics.dsr > 0.05
        },
        {
            name: 'Probabilistic Sharpe Ratio',
            value: statistics.psr?.toFixed(3) || '-',
            threshold: '> 0.95',
            passed: statistics.psr > 0.95
        },
        {
            name: 'Max Drawdown Test',
            value: statistics.max_drawdown ? 
                   (statistics.max_drawdown * 100).toFixed(2) + '%' : '-',
            threshold: '< 15%',
            passed: statistics.max_drawdown < 0.15
        }
    ];
    
    tests.forEach(test => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${test.name}</td>
            <td><strong>${test.value}</strong></td>
            <td>${test.threshold}</td>
            <td>
                ${test.passed ? 
                    '<span class="badge bg-success"><i class="fas fa-check"></i> Passé</span>' :
                    '<span class="badge bg-danger"><i class="fas fa-times"></i> Échoué</span>'
                }
            </td>
        `;
        tbody.appendChild(row);
    });
}

// ============================================================================
// INITIALISATION DES GRAPHIQUES
// ============================================================================

function initCharts() {
    // Graphique d'equity par fold
    const equityCtx = document.getElementById('equity-by-fold-chart');
    if (equityCtx) {
        validationState.charts.equity = new Chart(equityCtx.getContext('2d'), {
            type: 'line',
            data: {
                labels: [],
                datasets: []
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Steps'
                        }
                    },
                    y: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Equity ($)'
                        }
                    }
                }
            }
        });
    }
    
    // Graphique de distribution des returns
    const returnsCtx = document.getElementById('returns-dist-chart');
    if (returnsCtx) {
        validationState.charts.returns = new Chart(returnsCtx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Fréquence',
                    data: [],
                    backgroundColor: 'rgba(75, 192, 192, 0.6)',
                    borderColor: 'rgb(75, 192, 192)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Returns'
                        }
                    },
                    y: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Fréquence'
                        }
                    }
                }
            }
        });
    }
}

// ============================================================================
// EXPORT ET RAPPORTS
// ============================================================================

function exportResults() {
    if (!validationState.currentResults) {
        showNotification('Erreur', 'Aucun résultat à exporter', 'warning');
        return;
    }
    
    const dataStr = JSON.stringify(validationState.currentResults, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `validation_results_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    showNotification('Succès', 'Résultats exportés', 'success');
}

function viewFullReport() {
    if (!validationState.currentResults) {
        showNotification('Erreur', 'Aucun rapport disponible', 'warning');
        return;
    }
    
    // Ouvrir dans un nouvel onglet
    const reportPath = validationState.currentResults.report_path;
    if (reportPath) {
        window.open(reportPath, '_blank');
    } else {
        showNotification('Erreur', 'Chemin du rapport non disponible', 'warning');
    }
}

// ============================================================================
// FONCTIONS UTILITAIRES
// ============================================================================

function createHistogram(data, bins) {
    if (!data || data.length === 0) {
        return { labels: [], values: [] };
    }
    
    const min = Math.min(...data);
    const max = Math.max(...data);
    const binWidth = (max - min) / bins;
    
    const histogram = new Array(bins).fill(0);
    const labels = [];
    
    data.forEach(value => {
        const binIndex = Math.min(
            Math.floor((value - min) / binWidth),
            bins - 1
        );
        histogram[binIndex]++;
    });
    
    for (let i = 0; i < bins; i++) {
        const binStart = min + i * binWidth;
        labels.push(binStart.toFixed(4));
    }
    
    return { labels, values: histogram };
}

function getColorForIndex(index, alpha = 1) {
    const colors = [
        `rgba(75, 192, 192, ${alpha})`,
        `rgba(255, 99, 132, ${alpha})`,
        `rgba(54, 162, 235, ${alpha})`,
        `rgba(255, 206, 86, ${alpha})`,
        `rgba(153, 102, 255, ${alpha})`
    ];
    
    return colors[index % colors.length];
}

function showNotification(title, message, type = 'info') {
    const toastHtml = `
        <div class="toast align-items-center text-white bg-${type} border-0" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="d-flex">
                <div class="toast-body">
                    <strong>${title}</strong><br>
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `;
    
    let toastContainer = document.querySelector('.toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
        document.body.appendChild(toastContainer);
    }
    
    toastContainer.insertAdjacentHTML('beforeend', toastHtml);
    
    const toastElement = toastContainer.lastElementChild;
    const toast = new bootstrap.Toast(toastElement, { delay: 5000 });
    toast.show();
    
    toastElement.addEventListener('hidden.bs.toast', function() {
        toastElement.remove();
    });
}
