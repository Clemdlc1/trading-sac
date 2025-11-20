/**
 * dashboard.js
 * Gestion de l'interface du dashboard principal
 */

// Connexion Socket.IO
const socket = io();

// Variables globales
let equityChart = null;
let updateInterval = null;

// État de l'application
const appState = {
    isTrading: false,
    isTraining: false,
    currentModel: null,
    equity: 10000,
    position: 0,
    pnl: 0,
    trades: []
};

// ============================================================================
// INITIALISATION
// ============================================================================

document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard initialisé');
    
    // Initialiser le graphique d'equity
    initEquityChart();
    
    // Charger le statut initial
    loadStatus();
    
    // Configurer les event listeners
    setupEventListeners();
    
    // Démarrer les mises à jour périodiques
    startPeriodicUpdates();
});

// ============================================================================
// SOCKET.IO HANDLERS
// ============================================================================

socket.on('connect', function() {
    console.log('Connecté au serveur');
    updateConnectionStatus(true);
});

socket.on('disconnect', function() {
    console.log('Déconnecté du serveur');
    updateConnectionStatus(false);
});

socket.on('status_update', function(data) {
    console.log('Mise à jour statut:', data);
    updateDashboard(data);
});

socket.on('new_trade', function(trade) {
    console.log('Nouveau trade:', trade);
    addTradeToTable(trade);
    showNotification('Nouveau trade', trade.action.toUpperCase(), 'info');
});

socket.on('training_progress', function(data) {
    console.log('Progression entraînement:', data);
    // Géré sur la page models
});

socket.on('model_loaded', function(data) {
    console.log('Modèle chargé:', data);
    appState.currentModel = data.model_name;
    document.getElementById('current-model').value = data.model_name;
    showNotification('Succès', `Modèle ${data.model_name} chargé`, 'success');
});

socket.on('risk_alert', function(data) {
    console.log('Alerte risque:', data);
    showNotification('Alerte Risque', `${data.type}: ${(data.value * 100).toFixed(2)}%`, 'danger');
});

// ============================================================================
// FONCTIONS DE CHARGEMENT DES DONNÉES
// ============================================================================

async function loadStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        appState.isTrading = data.is_trading;
        appState.isTraining = data.is_training;
        appState.currentModel = data.current_model;
        appState.equity = data.equity;
        appState.position = data.position;
        appState.pnl = data.pnl;
        
        updateDashboard(data);
        updateTradingButtons();
    } catch (error) {
        console.error('Erreur chargement statut:', error);
    }
}

async function loadPerformance() {
    try {
        const response = await fetch('/api/performance');
        const data = await response.json();
        
        updatePerformanceMetrics(data);
    } catch (error) {
        console.error('Erreur chargement performance:', error);
    }
}

async function loadTrades() {
    try {
        const response = await fetch('/api/trades?limit=50');
        const data = await response.json();
        
        appState.trades = data.trades;
        populateTradesTable();
    } catch (error) {
        console.error('Erreur chargement trades:', error);
    }
}

async function loadEquityCurve() {
    try {
        const response = await fetch('/api/equity-curve');
        const data = await response.json();
        
        updateEquityChart(data.equity);
    } catch (error) {
        console.error('Erreur chargement equity curve:', error);
    }
}

// ============================================================================
// FONCTIONS DE MISE À JOUR DE L'INTERFACE
// ============================================================================

function updateDashboard(data) {
    // Mettre à jour les cartes de statut
    document.getElementById('equity-value').textContent = formatCurrency(data.equity);
    document.getElementById('pnl-value').textContent = formatCurrency(data.pnl);
    document.getElementById('position-value').textContent = data.position.toFixed(2);
    document.getElementById('trades-count').textContent = data.daily_trades || 0;
    
    // Changer la couleur du P&L
    const pnlElement = document.getElementById('pnl-value');
    if (data.pnl > 0) {
        pnlElement.classList.add('text-positive');
        pnlElement.classList.remove('text-negative');
    } else if (data.pnl < 0) {
        pnlElement.classList.add('text-negative');
        pnlElement.classList.remove('text-positive');
    }
    
    // Mettre à jour le modèle actuel
    if (data.current_model) {
        document.getElementById('current-model').value = data.current_model;
    }
}

function updatePerformanceMetrics(metrics) {
    document.getElementById('sharpe-ratio').textContent = 
        metrics.sharpe_ratio ? metrics.sharpe_ratio.toFixed(2) : '-';
    
    document.getElementById('max-drawdown').textContent = 
        metrics.max_drawdown ? (metrics.max_drawdown * 100).toFixed(2) + '%' : '-';
    
    document.getElementById('win-rate').textContent = 
        metrics.win_rate ? (metrics.win_rate * 100).toFixed(2) + '%' : '-';
    
    document.getElementById('profit-factor').textContent = 
        metrics.profit_factor ? metrics.profit_factor.toFixed(2) : '-';
}

function updateConnectionStatus(connected) {
    const statusIcon = document.getElementById('connection-status');
    const statusText = document.getElementById('connection-text');
    
    if (connected) {
        statusIcon.classList.add('connected');
        statusIcon.classList.remove('disconnected');
        statusText.textContent = 'Connecté';
    } else {
        statusIcon.classList.remove('connected');
        statusIcon.classList.add('disconnected');
        statusText.textContent = 'Déconnecté';
    }
}

function updateTradingButtons() {
    const btnStart = document.getElementById('btn-start-trading');
    const btnStop = document.getElementById('btn-stop-trading');
    const tradingStatus = document.getElementById('trading-status');
    
    if (appState.isTrading) {
        btnStart.disabled = true;
        btnStop.disabled = false;
        tradingStatus.innerHTML = '<i class="fas fa-play-circle"></i> Trading Actif';
        tradingStatus.className = 'badge bg-success';
    } else {
        btnStart.disabled = false;
        btnStop.disabled = true;
        tradingStatus.innerHTML = '<i class="fas fa-stop-circle"></i> Non actif';
        tradingStatus.className = 'badge bg-secondary';
    }
}

// ============================================================================
// GESTION DU TABLEAU DES TRADES
// ============================================================================

function populateTradesTable() {
    const tbody = document.querySelector('#trades-table tbody');
    
    if (appState.trades.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">Aucun trade récent</td></tr>';
        return;
    }
    
    tbody.innerHTML = '';
    
    appState.trades.slice(0, 10).forEach(trade => {
        const row = createTradeRow(trade);
        tbody.appendChild(row);
    });
}

function addTradeToTable(trade) {
    const tbody = document.querySelector('#trades-table tbody');
    
    // Supprimer le message "Aucun trade"
    if (tbody.querySelector('td[colspan="7"]')) {
        tbody.innerHTML = '';
    }
    
    // Ajouter le nouveau trade en haut
    const row = createTradeRow(trade);
    row.classList.add('new-trade');
    tbody.insertBefore(row, tbody.firstChild);
    
    // Limiter à 10 trades affichés
    while (tbody.children.length > 10) {
        tbody.removeChild(tbody.lastChild);
    }
    
    // Ajouter aux trades
    appState.trades.unshift(trade);
}

function createTradeRow(trade) {
    const row = document.createElement('tr');
    row.classList.add('trade-row');
    
    const typeClass = trade.action === 'long' ? 'trade-type-long' : 'trade-type-short';
    const typeIcon = trade.action === 'long' ? 'fa-arrow-up' : 'fa-arrow-down';
    
    row.innerHTML = `
        <td>${formatTimestamp(trade.timestamp)}</td>
        <td class="${typeClass}">
            <i class="fas ${typeIcon}"></i> ${trade.action.toUpperCase()}
        </td>
        <td>${trade.size.toFixed(2)}</td>
        <td>${trade.price ? trade.price.toFixed(5) : '-'}</td>
        <td>${trade.sl ? trade.sl.toFixed(5) : '-'}</td>
        <td>${trade.tp ? trade.tp.toFixed(5) : '-'}</td>
        <td><span class="badge bg-info">Ouvert</span></td>
    `;
    
    return row;
}

// ============================================================================
// GESTION DU GRAPHIQUE
// ============================================================================

function initEquityChart() {
    const ctx = document.getElementById('equity-chart').getContext('2d');
    
    equityChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Equity',
                data: [],
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.1)',
                tension: 0.1,
                fill: true
            }]
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
                        text: 'Temps'
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

function updateEquityChart(equityData) {
    if (!equityChart || !equityData) return;
    
    const labels = equityData.map(point => formatTimestamp(point.timestamp));
    const data = equityData.map(point => point.equity);
    
    equityChart.data.labels = labels;
    equityChart.data.datasets[0].data = data;
    equityChart.update();
}

// ============================================================================
// EVENT LISTENERS
// ============================================================================

function setupEventListeners() {
    // Bouton démarrer trading
    document.getElementById('btn-start-trading').addEventListener('click', async function() {
        if (!appState.currentModel) {
            showNotification('Erreur', 'Aucun modèle chargé. Veuillez charger un modèle d\'abord.', 'danger');
            return;
        }
        
        if (confirm('Êtes-vous sûr de vouloir démarrer le trading live ?')) {
            await startTrading();
        }
    });
    
    // Bouton arrêter trading
    document.getElementById('btn-stop-trading').addEventListener('click', async function() {
        if (confirm('Êtes-vous sûr de vouloir arrêter le trading ?')) {
            await stopTrading();
        }
    });
}

async function startTrading() {
    try {
        const response = await fetch('/api/trading/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        
        if (data.success) {
            appState.isTrading = true;
            updateTradingButtons();
            showNotification('Succès', 'Trading démarré avec succès', 'success');
        } else {
            showNotification('Erreur', data.error, 'danger');
        }
    } catch (error) {
        console.error('Erreur démarrage trading:', error);
        showNotification('Erreur', 'Impossible de démarrer le trading', 'danger');
    }
}

async function stopTrading() {
    try {
        const response = await fetch('/api/trading/stop', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        
        if (data.success) {
            appState.isTrading = false;
            updateTradingButtons();
            showNotification('Succès', 'Trading arrêté', 'info');
        } else {
            showNotification('Erreur', data.error, 'danger');
        }
    } catch (error) {
        console.error('Erreur arrêt trading:', error);
        showNotification('Erreur', 'Impossible d\'arrêter le trading', 'danger');
    }
}

// ============================================================================
// FONCTIONS UTILITAIRES
// ============================================================================

function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(value);
}

function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleString('fr-FR', {
        day: '2-digit',
        month: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function showNotification(title, message, type = 'info') {
    // Créer une notification toast
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
    
    // Ajouter au container de toasts (créer si nécessaire)
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
    
    // Supprimer après fermeture
    toastElement.addEventListener('hidden.bs.toast', function() {
        toastElement.remove();
    });
}

// ============================================================================
// MISES À JOUR PÉRIODIQUES
// ============================================================================

function startPeriodicUpdates() {
    // Mettre à jour toutes les 5 secondes
    updateInterval = setInterval(async function() {
        if (appState.isTrading) {
            await loadStatus();
            await loadPerformance();
            await loadEquityCurve();
        }
    }, 5000);
    
    // Charger les trades toutes les 30 secondes
    setInterval(async function() {
        if (appState.isTrading) {
            await loadTrades();
        }
    }, 30000);
}

// ============================================================================
// CLEANUP
// ============================================================================

window.addEventListener('beforeunload', function() {
    if (updateInterval) {
        clearInterval(updateInterval);
    }
    socket.disconnect();
});
