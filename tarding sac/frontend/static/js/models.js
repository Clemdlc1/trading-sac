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
    totalEpisodes: 0
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
        // TODO: Implémenter l'API de suppression
        showNotification('Succès', 'Modèle supprimé', 'info');
        
        // Recharger les modèles
        setTimeout(() => loadModels(), 500);
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
    // Formulaire d'entraînement
    document.getElementById('training-form').addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const numEpisodes = parseInt(document.getElementById('num-episodes').value);
        const batchSize = parseInt(document.getElementById('batch-size').value);
        const fromCheckpoint = document.getElementById('from-checkpoint').value;
        
        await startTraining(numEpisodes, batchSize, fromCheckpoint);
    });
    
    // Bouton arrêt entraînement
    document.getElementById('btn-stop-training').addEventListener('click', async function() {
        await stopTraining();
    });
}

async function startTraining(numEpisodes, batchSize, fromCheckpoint) {
    try {
        const response = await fetch('/api/training/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                num_episodes: numEpisodes,
                batch_size: batchSize,
                from_checkpoint: fromCheckpoint || null
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            modelsState.isTraining = true;
            modelsState.totalEpisodes = numEpisodes;
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
    
    const percentage = (data.episode / modelsState.totalEpisodes) * 100;
    const progressBar = document.getElementById('progress-bar');
    progressBar.style.width = percentage + '%';
    progressBar.textContent = Math.round(percentage) + '%';
    
    const info = document.getElementById('training-info');
    info.textContent = `Épisode ${data.episode}/${modelsState.totalEpisodes} - Reward: ${data.reward.toFixed(2)}`;
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
