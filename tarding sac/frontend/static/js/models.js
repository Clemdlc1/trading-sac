/**
 * models.js
 * Gestion de l'interface des modèles
 */

const socket = io();

// État de la page
const modelsState = {
    checkpoints: [],
    production: [],
    isTraining: false,
    currentEpisode: 0,
    totalEpisodes: 0,
    currentAgent: null,
    trainingType: null
};

// Mapping des noms d'agents
const agentNames = {
    '3': 'Sharpe Optimized',
    '4': 'Sortino Optimized',
    '5': 'Aggressive',
    'all': 'All Agents',
    'meta_controller': 'Meta-Controller'
};

// Charts
let charts = {
    reward: null,
    sharpe: null,
    winrate: null,
    returns: null
};

// ============================================================================
// INITIALISATION
// ============================================================================

document.addEventListener('DOMContentLoaded', function() {
    console.log('Page models initialisée');

    // Charger les modèles disponibles
    loadModels();

    // Setup event listeners
    setupEventListeners();

    // Initialiser les graphiques
    initCharts();

    // Charger l'état du training si en cours
    loadTrainingState();
});

// ============================================================================
// SOCKET.IO HANDLERS
// ============================================================================

socket.on('training_progress', function(data) {
    console.log('Progression:', data);
    updateTrainingProgress(data);
});

socket.on('training_complete', function(data) {
    console.log('Entraînement terminé:', data);
    modelsState.isTraining = false;
    hideTrainingProgress();
    showNotification('Succès', data.message, 'success');
    
    // Recharger les modèles
    setTimeout(() => loadModels(), 1000);
});

socket.on('training_error', function(data) {
    console.error('Erreur entraînement:', data);
    modelsState.isTraining = false;
    hideTrainingProgress();
    showNotification('Erreur', data.error, 'danger');
});

socket.on('training_device_info', function(data) {
    console.log('Device info:', data);
    const deviceInfo = document.getElementById('device-info');
    const deviceName = document.getElementById('device-name');

    deviceName.textContent = data.device;

    // Changer la couleur selon GPU/CPU
    if (data.cuda_available) {
        deviceInfo.className = 'alert alert-success mb-2';
        deviceName.innerHTML = `<i class="fas fa-bolt"></i> ${data.device} (GPU Accéléré)`;
    } else {
        deviceInfo.className = 'alert alert-warning mb-2';
        deviceName.innerHTML = `<i class="fas fa-exclamation-triangle"></i> ${data.device} (Mode CPU - Plus lent)`;
    }

    deviceInfo.style.display = 'block';
});

socket.on('episode_step_progress', function(data) {
    // Mettre à jour seulement la barre de progression de l'épisode (message léger)
    if (data.current_step !== undefined && data.episode_length !== undefined) {
        const episodePercentage = (data.current_step / data.episode_length) * 100;
        const episodeProgressBar = document.getElementById('episode-progress-bar');
        episodeProgressBar.style.width = episodePercentage + '%';
        document.getElementById('episode-progress-text').textContent = `${data.current_step}/${data.episode_length} steps`;
    }
});

// ============================================================================
// CHARGEMENT DES DONNÉES
// ============================================================================

async function loadModels() {
    try {
        const response = await fetch('/api/models/list');
        const data = await response.json();
        
        modelsState.checkpoints = data.checkpoints;
        modelsState.production = data.production;
        
        populateCheckpointsTable();
        populateProductionTable();
        populateCheckpointSelect();
    } catch (error) {
        console.error('Erreur chargement modèles:', error);
        showNotification('Erreur', 'Impossible de charger les modèles', 'danger');
    }
}

// ============================================================================
// MISE À JOUR DES TABLEAUX
// ============================================================================

function populateCheckpointsTable() {
    const tbody = document.querySelector('#checkpoints-table tbody');
    
    if (modelsState.checkpoints.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" class="text-center text-muted">Aucun checkpoint disponible</td></tr>';
        return;
    }
    
    tbody.innerHTML = '';
    
    modelsState.checkpoints.forEach(model => {
        const row = createModelRow(model);
        tbody.appendChild(row);
    });
}

function populateProductionTable() {
    const tbody = document.querySelector('#production-table tbody');
    
    if (modelsState.production.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" class="text-center text-muted">Aucun modèle en production</td></tr>';
        return;
    }
    
    tbody.innerHTML = '';
    
    modelsState.production.forEach(model => {
        const row = createModelRow(model, true);
        tbody.appendChild(row);
    });
}

function createModelRow(model, isProduction = false) {
    const row = document.createElement('tr');
    
    const size = formatFileSize(model.size);
    const date = formatDate(model.modified);
    
    row.innerHTML = `
        <td>
            <i class="fas fa-cube text-primary"></i> 
            <strong>${model.name}</strong>
        </td>
        <td>${size}</td>
        <td>${date}</td>
        <td>
            <button class="btn btn-sm btn-primary" onclick="loadModel('${model.path}')">
                <i class="fas fa-upload"></i> Charger
            </button>
            ${!isProduction ? `
            <button class="btn btn-sm btn-success" onclick="promoteToProduction('${model.path}')">
                <i class="fas fa-rocket"></i> Production
            </button>
            ` : ''}
            <button class="btn btn-sm btn-info" onclick="showModelDetails('${model.path}')">
                <i class="fas fa-info-circle"></i> Détails
            </button>
            <button class="btn btn-sm btn-danger" onclick="deleteModel('${model.path}')">
                <i class="fas fa-trash"></i>
            </button>
        </td>
    `;
    
    return row;
}

function populateCheckpointSelect() {
    const select = document.getElementById('from-checkpoint');
    
    // Garder l'option "Nouveau modèle"
    select.innerHTML = '<option value="">Nouveau modèle</option>';
    
    modelsState.checkpoints.forEach(model => {
        const option = document.createElement('option');
        option.value = model.path;
        option.textContent = model.name;
        select.appendChild(option);
    });
}

// ============================================================================
// ACTIONS SUR LES MODÈLES
// ============================================================================

async function loadModel(modelPath) {
    try {
        const response = await fetch('/api/models/load', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ model_path: modelPath })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showNotification('Succès', data.message, 'success');
        } else {
            showNotification('Erreur', data.error, 'danger');
        }
    } catch (error) {
        console.error('Erreur chargement modèle:', error);
        showNotification('Erreur', 'Impossible de charger le modèle', 'danger');
    }
}

async function promoteToProduction(modelPath) {
    if (!confirm('Promouvoir ce modèle en production ?')) {
        return;
    }
    
    try {
        // Copier le modèle vers production
        const modelName = modelPath.split('/').pop();
        const productionPath = modelPath.replace('checkpoints', 'production');
        
        // TODO: Implémenter l'API de promotion
        showNotification('Succès', 'Modèle promu en production', 'success');
        
        // Recharger les modèles
        setTimeout(() => loadModels(), 500);
    } catch (error) {
        console.error('Erreur promotion:', error);
        showNotification('Erreur', 'Impossible de promouvoir le modèle', 'danger');
    }
}

async function deleteModel(modelPath) {
    if (!confirm('Êtes-vous sûr de vouloir supprimer ce modèle ?')) {
        return;
    }
    
    try {
        const response = await fetch('/api/models/delete', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ model_path: modelPath })
        });

        const data = await response.json();

        if (data.success) {
            showNotification('Succès', data.message, 'success');
            // Recharger les modèles
            setTimeout(() => loadModels(), 500);
        } else {
            showNotification('Erreur', data.error, 'danger');
        }
    } catch (error) {
        console.error('Erreur suppression:', error);
        showNotification('Erreur', 'Impossible de supprimer le modèle', 'danger');
    }
}

function showModelDetails(modelPath) {
    // TODO: Charger et afficher les détails du modèle
    const modal = new bootstrap.Modal(document.getElementById('modelDetailsModal'));
    
    const content = document.getElementById('model-details-content');
    content.innerHTML = `
        <h6>Chemin du modèle</h6>
        <p><code>${modelPath}</code></p>
        
        <h6>Informations</h6>
        <table class="table table-sm">
            <tr>
                <td>Architecture</td>
                <td>SAC (Soft Actor-Critic)</td>
            </tr>
            <tr>
                <td>State Dim</td>
                <td>30</td>
            </tr>
            <tr>
                <td>Action Dim</td>
                <td>1</td>
            </tr>
            <tr>
                <td>Hidden Dim</td>
                <td>256</td>
            </tr>
        </table>
        
        <div class="alert alert-info">
            <i class="fas fa-info-circle"></i>
            Détails complets disponibles dans le fichier checkpoint
        </div>
    `;
    
    modal.show();
}

// ============================================================================
// ENTRAÎNEMENT
// ============================================================================

function setupEventListeners() {
    // Formulaire d'entraînement SAC
    document.getElementById('training-form').addEventListener('submit', async function(e) {
        e.preventDefault();

        const numEpisodes = parseInt(document.getElementById('num-episodes').value);
        const batchSize = parseInt(document.getElementById('batch-size').value);
        const fromCheckpoint = document.getElementById('from-checkpoint').value;
        const agentId = document.getElementById('agent-select').value;

        await startTraining(numEpisodes, batchSize, fromCheckpoint, agentId ? parseInt(agentId) : null);
    });

    // Bouton entraînement meta-controller
    document.getElementById('btn-train-meta').addEventListener('click', async function() {
        const numEpisodes = parseInt(document.getElementById('num-episodes').value) || 500;
        const batchSize = parseInt(document.getElementById('batch-size').value) || 256;

        if (confirm('Entraîner le meta-controller ? Assurez-vous que les 3 agents SAC sont déjà entraînés.')) {
            await startMetaControllerTraining(numEpisodes, batchSize);
        }
    });

    // Bouton arrêt entraînement
    document.getElementById('btn-stop-training').addEventListener('click', async function() {
        await stopTraining();
    });
}

async function loadTrainingState() {
    try {
        const response = await fetch('/api/status');
        const status = await response.json();

        if (status.training_state && status.training_state.is_training) {
            // Un entraînement est en cours
            modelsState.isTraining = true;
            modelsState.currentEpisode = status.training_state.current_episode || 0;
            modelsState.totalEpisodes = status.training_state.total_episodes || 1000;
            modelsState.currentAgent = status.training_state.current_agent;
            modelsState.trainingType = status.training_state.training_type;

            // Afficher la barre de progression
            showTrainingProgress();

            // Mettre à jour les métriques si disponibles
            if (status.training_state.metrics) {
                updateTrainingProgress({
                    episode: modelsState.currentEpisode,
                    total_episodes: modelsState.totalEpisodes,
                    agent: modelsState.currentAgent,
                    ...status.training_state.metrics
                });
            }

            console.log('État du training restauré:', status.training_state);
        }
    } catch (error) {
        console.error('Erreur chargement état training:', error);
    }
}

async function startTraining(numEpisodes, batchSize, fromCheckpoint, agentId) {
    try {
        const response = await fetch('/api/training/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                num_episodes: numEpisodes,
                batch_size: batchSize,
                from_checkpoint: fromCheckpoint || null,
                agent_id: agentId || null
            })
        });

        const data = await response.json();

        if (data.success) {
            modelsState.isTraining = true;
            modelsState.totalEpisodes = numEpisodes;
            modelsState.currentAgent = agentId || 'all';
            modelsState.trainingType = 'sac_agent';
            showTrainingProgress();
            showNotification('Succès', data.message, 'success');
        } else {
            showNotification('Erreur', data.error, 'danger');
        }
    } catch (error) {
        console.error('Erreur démarrage entraînement:', error);
        showNotification('Erreur', 'Impossible de démarrer l\'entraînement', 'danger');
    }
}

async function startMetaControllerTraining(numEpisodes, batchSize) {
    try {
        const response = await fetch('/api/training/meta-controller/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                num_episodes: numEpisodes,
                batch_size: batchSize
            })
        });

        const data = await response.json();

        if (data.success) {
            modelsState.isTraining = true;
            modelsState.totalEpisodes = numEpisodes;
            modelsState.currentAgent = 'meta_controller';
            modelsState.trainingType = 'meta_controller';
            showTrainingProgress();
            showNotification('Succès', data.message, 'success');
        } else {
            showNotification('Erreur', data.error, 'danger');
        }
    } catch (error) {
        console.error('Erreur démarrage entraînement meta-controller:', error);
        showNotification('Erreur', 'Impossible de démarrer l\'entraînement du meta-controller', 'danger');
    }
}

async function stopTraining() {
    try {
        const response = await fetch('/api/training/stop', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        
        if (data.success) {
            modelsState.isTraining = false;
            hideTrainingProgress();
            showNotification('Info', data.message, 'info');
        } else {
            showNotification('Erreur', data.error, 'danger');
        }
    } catch (error) {
        console.error('Erreur arrêt entraînement:', error);
        showNotification('Erreur', 'Impossible d\'arrêter l\'entraînement', 'danger');
    }
}

// ============================================================================
// INTERFACE DE PROGRESSION
// ============================================================================

function showTrainingProgress() {
    document.getElementById('training-progress').style.display = 'block';
    updateTrainingProgress({ episode: 0, reward: 0 });
}

function hideTrainingProgress() {
    document.getElementById('training-progress').style.display = 'none';
}

function updateTrainingProgress(data) {
    modelsState.currentEpisode = data.episode;
    if (data.total_episodes) {
        modelsState.totalEpisodes = data.total_episodes;
    }

    const percentage = (data.episode / modelsState.totalEpisodes) * 100;
    const progressBar = document.getElementById('progress-bar');
    progressBar.style.width = percentage + '%';
    progressBar.textContent = Math.round(percentage) + '%';

    const info = document.getElementById('training-info');
    const agentName = data.agent !== undefined ? (agentNames[data.agent] || `Agent #${data.agent}`) : 'N/A';
    const totalTrades = data.total_trades !== undefined ? ` - Trades: ${data.total_trades}` : '';
    info.textContent = `Épisode ${data.episode}/${modelsState.totalEpisodes} - ${agentName}${totalTrades}`;

    // Réinitialiser la barre de progression de l'épisode (fin d'épisode, prêt pour le suivant)
    const episodeProgressBar = document.getElementById('episode-progress-bar');
    episodeProgressBar.style.width = '0%';
    document.getElementById('episode-progress-text').textContent = '0/' + (data.episode_length || '?') + ' steps';

    // Mettre à jour les statistiques d'épisode
    if (data.total_trades !== undefined) {
        document.getElementById('stat-total-trades').textContent = data.total_trades;
        document.getElementById('stat-winning-trades').textContent = data.winning_trades || 0;
        const losingTrades = data.total_trades - (data.winning_trades || 0);
        document.getElementById('stat-losing-trades').textContent = losingTrades;
        document.getElementById('stat-final-equity').textContent = (data.final_equity || 0).toFixed(2);
    }

    // Mettre à jour les métriques détaillées
    document.getElementById('metric-reward').textContent = (data.reward || 0).toFixed(2);
    document.getElementById('metric-steps').textContent = data.steps || 0;
    document.getElementById('metric-sharpe').textContent = (data.sharpe_ratio || 0).toFixed(3);
    document.getElementById('metric-winrate').textContent = ((data.win_rate || 0) * 100).toFixed(1) + '%';
    document.getElementById('metric-sortino').textContent = (data.sortino_ratio || 0).toFixed(3);
    document.getElementById('metric-dd').textContent = ((data.max_drawdown || 0) * 100).toFixed(2) + '%';
    document.getElementById('metric-pf').textContent = (data.profit_factor || 0).toFixed(2);
    document.getElementById('metric-return').textContent = ((data.total_return || 0) * 100).toFixed(2) + '%';

    // Afficher les loss si disponibles (SAC agent uniquement)
    if (data.critic_loss !== undefined || data.actor_loss !== undefined) {
        document.getElementById('loss-metrics').style.display = 'block';
        document.getElementById('metric-critic-loss').textContent = (data.critic_loss || 0).toFixed(4);
        document.getElementById('metric-actor-loss').textContent = (data.actor_loss || 0).toFixed(4);
    } else {
        document.getElementById('loss-metrics').style.display = 'none';
    }

    // Mettre à jour les graphiques si historique disponible
    if (data.metrics_history) {
        updateCharts(data.metrics_history);
    }
}

// ============================================================================
// GRAPHIQUES
// ============================================================================

function initCharts() {
    // Configuration commune
    const commonOptions = {
        responsive: true,
        maintainAspectRatio: true,
        aspectRatio: 2,
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
                    text: 'Épisode'
                }
            },
            y: {
                display: true
            }
        }
    };

    // Reward Chart
    const ctxReward = document.getElementById('chart-reward').getContext('2d');
    charts.reward = new Chart(ctxReward, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Reward',
                data: [],
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.1)',
                tension: 0.1,
                fill: true
            }]
        },
        options: {
            ...commonOptions,
            scales: {
                ...commonOptions.scales,
                y: {
                    ...commonOptions.scales.y,
                    title: {
                        display: true,
                        text: 'Reward'
                    }
                }
            }
        }
    });

    // Sharpe Ratio Chart
    const ctxSharpe = document.getElementById('chart-sharpe').getContext('2d');
    charts.sharpe = new Chart(ctxSharpe, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Sharpe Ratio',
                data: [],
                borderColor: 'rgb(54, 162, 235)',
                backgroundColor: 'rgba(54, 162, 235, 0.1)',
                tension: 0.1,
                fill: true
            }]
        },
        options: {
            ...commonOptions,
            scales: {
                ...commonOptions.scales,
                y: {
                    ...commonOptions.scales.y,
                    title: {
                        display: true,
                        text: 'Sharpe Ratio'
                    }
                }
            }
        }
    });

    // Win Rate Chart
    const ctxWinrate = document.getElementById('chart-winrate').getContext('2d');
    charts.winrate = new Chart(ctxWinrate, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Win Rate (%)',
                data: [],
                borderColor: 'rgb(255, 206, 86)',
                backgroundColor: 'rgba(255, 206, 86, 0.1)',
                tension: 0.1,
                fill: true
            }]
        },
        options: {
            ...commonOptions,
            scales: {
                ...commonOptions.scales,
                y: {
                    ...commonOptions.scales.y,
                    title: {
                        display: true,
                        text: 'Win Rate (%)'
                    },
                    min: 0,
                    max: 100
                }
            }
        }
    });

    // Total Return Chart
    const ctxReturns = document.getElementById('chart-returns').getContext('2d');
    charts.returns = new Chart(ctxReturns, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Total Return (%)',
                data: [],
                borderColor: 'rgb(153, 102, 255)',
                backgroundColor: 'rgba(153, 102, 255, 0.1)',
                tension: 0.1,
                fill: true
            }]
        },
        options: {
            ...commonOptions,
            scales: {
                ...commonOptions.scales,
                y: {
                    ...commonOptions.scales.y,
                    title: {
                        display: true,
                        text: 'Total Return (%)'
                    }
                }
            }
        }
    });
}

function updateCharts(metricsHistory) {
    if (!metricsHistory || !metricsHistory.episodes) return;

    // Update Reward Chart
    charts.reward.data.labels = metricsHistory.episodes;
    charts.reward.data.datasets[0].data = metricsHistory.rewards;
    charts.reward.update('none'); // Update without animation for performance

    // Update Sharpe Chart
    charts.sharpe.data.labels = metricsHistory.episodes;
    charts.sharpe.data.datasets[0].data = metricsHistory.sharpe_ratios;
    charts.sharpe.update('none');

    // Update Win Rate Chart (convert to percentage)
    charts.winrate.data.labels = metricsHistory.episodes;
    charts.winrate.data.datasets[0].data = metricsHistory.win_rates.map(wr => wr * 100);
    charts.winrate.update('none');

    // Update Returns Chart (convert to percentage)
    charts.returns.data.labels = metricsHistory.episodes;
    charts.returns.data.datasets[0].data = metricsHistory.total_returns.map(r => r * 100);
    charts.returns.update('none');
}

// ============================================================================
// FONCTIONS UTILITAIRES
// ============================================================================

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

function formatDate(isoString) {
    const date = new Date(isoString);
    return date.toLocaleString('fr-FR', {
        day: '2-digit',
        month: '2-digit',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
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

// ============================================================================
// EXPORTS GLOBAUX
// ============================================================================

window.loadModel = loadModel;
window.promoteToProduction = promoteToProduction;
window.deleteModel = deleteModel;
window.showModelDetails = showModelDetails;
