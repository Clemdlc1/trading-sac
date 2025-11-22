/**
 * dashboard.js - Version 3.0
 * Premium Trading Dashboard with Enhanced Visualizations
 */

// ============================================================================
// CONFIGURATION & GLOBALS
// ============================================================================

const socket = io();

// Chart instances
let equityChart = null;
let pnlDistributionChart = null;
let positionSizeChart = null;

// Update intervals
let updateInterval = null;

// Application state
const appState = {
    isTrading: false,
    isTraining: false,
    currentModel: null,
    equity: 10000,
    initialEquity: 10000,
    position: 0,
    pnl: 0,
    dailyTrades: 0,
    trades: [],
    equityHistory: [],
    selectedPeriod: '1d'
};

// Chart color scheme
const chartColors = {
    primary: '#667eea',
    success: '#11998e',
    danger: '#ff6b6b',
    warning: '#f093fb',
    info: '#4facfe',
    gradient: {
        primary: ['rgba(102, 126, 234, 0.8)', 'rgba(118, 75, 162, 0.8)'],
        success: ['rgba(17, 153, 142, 0.8)', 'rgba(56, 239, 125, 0.8)'],
        danger: ['rgba(255, 107, 107, 0.8)', 'rgba(238, 90, 111, 0.8)']
    }
};

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Premium Trading Dashboard initialized');

    // Initialize all charts
    initEquityChart();
    initPnlDistributionChart();
    initPositionSizeChart();

    // Load initial data
    loadStatus();
    loadPerformance();
    loadEquityCurve();
    loadTrades();

    // Setup event listeners
    setupEventListeners();

    // Start periodic updates
    startPeriodicUpdates();

    // Add smooth scroll behavior
    document.documentElement.style.scrollBehavior = 'smooth';
});

// ============================================================================
// SOCKET.IO HANDLERS
// ============================================================================

socket.on('connect', function() {
    console.log('✅ Connected to server');
    updateConnectionStatus(true);
});

socket.on('disconnect', function() {
    console.log('❌ Disconnected from server');
    updateConnectionStatus(false);
});

socket.on('status_update', function(data) {
    console.log('📊 Status update:', data);
    updateDashboard(data);
});

socket.on('new_trade', function(trade) {
    console.log('💰 New trade:', trade);
    addTradeToTable(trade);
    showNotification('Nouveau Trade', `${trade.action.toUpperCase()} - Taille: ${trade.size.toFixed(2)}`, 'info');

    // Update charts
    setTimeout(() => {
        loadEquityCurve();
        loadPerformance();
    }, 500);
});

socket.on('training_progress', function(data) {
    console.log('🎓 Training progress:', data);
    updateTrainingStatus(data);
});

socket.on('model_loaded', function(data) {
    console.log('🧠 Model loaded:', data);
    appState.currentModel = data.model_name;
    document.getElementById('current-model').value = data.model_name;
    showNotification('Succès', `Modèle ${data.model_name} chargé`, 'success');
});

socket.on('risk_alert', function(data) {
    console.log('⚠️ Risk alert:', data);
    showNotification('Alerte Risque', `${data.type}: ${(data.value * 100).toFixed(2)}%`, 'danger');
});

// ============================================================================
// DATA LOADING FUNCTIONS
// ============================================================================

async function loadStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();

        appState.isTrading = data.is_trading || false;
        appState.isTraining = data.is_training || false;
        appState.currentModel = data.current_model;
        appState.equity = data.equity || 10000;
        appState.position = data.position || 0;
        appState.pnl = data.pnl || 0;
        appState.dailyTrades = data.daily_trades || 0;

        updateDashboard(data);
        updateTradingButtons();
    } catch (error) {
        console.error('Error loading status:', error);
    }
}

async function loadPerformance() {
    try {
        const response = await fetch('/api/performance');
        const data = await response.json();

        updatePerformanceMetrics(data);
        updateFooterStats(data);
    } catch (error) {
        console.error('Error loading performance:', error);
    }
}

async function loadTrades() {
    try {
        const response = await fetch('/api/trades?limit=50');
        const data = await response.json();

        appState.trades = data.trades || [];
        populateTradesTable();
    } catch (error) {
        console.error('Error loading trades:', error);
    }
}

async function loadEquityCurve() {
    try {
        const response = await fetch(`/api/equity-curve?period=${appState.selectedPeriod}`);
        const data = await response.json();

        appState.equityHistory = data.equity || [];
        updateEquityChart(appState.equityHistory);
        updatePositionSizeChart(data.positions || []);
    } catch (error) {
        console.error('Error loading equity curve:', error);
    }
}

// ============================================================================
// UI UPDATE FUNCTIONS
// ============================================================================

function updateDashboard(data) {
    // Update equity card
    const equity = data.equity || appState.equity;
    const equityChange = ((equity - appState.initialEquity) / appState.initialEquity * 100);
    document.getElementById('equity-value').textContent = formatCurrency(equity);
    document.getElementById('equity-change').textContent = `${equityChange >= 0 ? '+' : ''}${equityChange.toFixed(2)}%`;

    // Update P&L card
    const pnl = data.pnl || 0;
    const pnlPercent = (pnl / appState.initialEquity * 100);
    const pnlElement = document.getElementById('pnl-value');
    pnlElement.textContent = formatCurrency(pnl);
    document.getElementById('pnl-percent').textContent = `${pnlPercent >= 0 ? '+' : ''}${pnlPercent.toFixed(2)}%`;

    // Color coding for P&L
    const pnlCard = pnlElement.closest('.card');
    if (pnl > 0) {
        pnlCard.classList.remove('bg-danger');
        pnlCard.classList.add('bg-success');
    } else if (pnl < 0) {
        pnlCard.classList.remove('bg-success');
        pnlCard.classList.add('bg-danger');
    }

    // Update position card
    const position = data.position || 0;
    document.getElementById('position-value').textContent = Math.abs(position).toFixed(2);
    const positionType = position > 0 ? 'Long' : position < 0 ? 'Short' : 'Aucune';
    document.getElementById('position-type').textContent = positionType;

    // Update trades card
    const dailyTrades = data.daily_trades || 0;
    document.getElementById('trades-count').textContent = dailyTrades;

    // Update win rate display
    const winRate = data.win_rate || 0;
    document.getElementById('win-rate-display').textContent = `Win Rate: ${(winRate * 100).toFixed(0)}%`;

    // Update model
    if (data.current_model) {
        document.getElementById('current-model').value = data.current_model;
    }
}

function updatePerformanceMetrics(metrics) {
    // Sharpe Ratio
    const sharpeElement = document.getElementById('sharpe-ratio');
    if (metrics.sharpe_ratio !== undefined && metrics.sharpe_ratio !== null) {
        sharpeElement.textContent = metrics.sharpe_ratio.toFixed(2);
        sharpeElement.className = 'metric-value ' + (metrics.sharpe_ratio > 0 ? 'metric-positive' : 'metric-negative');
    } else {
        sharpeElement.textContent = '-';
        sharpeElement.className = 'metric-value';
    }

    // Max Drawdown
    const ddElement = document.getElementById('max-drawdown');
    if (metrics.max_drawdown !== undefined && metrics.max_drawdown !== null) {
        ddElement.textContent = (metrics.max_drawdown * 100).toFixed(2) + '%';
    } else {
        ddElement.textContent = '-';
    }

    // Win Rate
    const wrElement = document.getElementById('win-rate');
    if (metrics.win_rate !== undefined && metrics.win_rate !== null) {
        wrElement.textContent = (metrics.win_rate * 100).toFixed(2) + '%';
        wrElement.className = 'metric-value ' + (metrics.win_rate > 0.5 ? 'metric-positive' : '');
    } else {
        wrElement.textContent = '-';
        wrElement.className = 'metric-value';
    }

    // Profit Factor
    const pfElement = document.getElementById('profit-factor');
    if (metrics.profit_factor !== undefined && metrics.profit_factor !== null) {
        pfElement.textContent = metrics.profit_factor.toFixed(2);
    } else {
        pfElement.textContent = '-';
    }

    // Update P&L distribution chart
    if (metrics.pnl_distribution) {
        updatePnlDistributionChart(metrics.pnl_distribution);
    }
}

function updateFooterStats(metrics) {
    document.getElementById('total-trades').textContent = metrics.total_trades || 0;
    document.getElementById('avg-win').textContent = formatCurrency(metrics.avg_win || 0);
    document.getElementById('avg-loss').textContent = formatCurrency(metrics.avg_loss || 0);
    document.getElementById('best-trade').textContent = formatCurrency(metrics.best_trade || 0);
    document.getElementById('worst-trade').textContent = formatCurrency(metrics.worst_trade || 0);

    // Format duration
    const avgDurationMinutes = metrics.avg_duration || 0;
    const hours = Math.floor(avgDurationMinutes / 60);
    const minutes = Math.floor(avgDurationMinutes % 60);
    document.getElementById('avg-duration').textContent = `${hours}h ${minutes}m`;
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

function updateTrainingStatus(data) {
    const trainingStatus = document.getElementById('training-status');

    if (data.is_training) {
        const progress = (data.progress || 0).toFixed(1);
        trainingStatus.innerHTML = `<i class="fas fa-spinner fa-spin"></i> Entraînement ${progress}%`;
        trainingStatus.className = 'badge bg-warning';
    } else {
        trainingStatus.innerHTML = '<i class="fas fa-stop-circle"></i> Non actif';
        trainingStatus.className = 'badge bg-secondary';
    }
}

// ============================================================================
// TRADES TABLE
// ============================================================================

function populateTradesTable() {
    const tbody = document.querySelector('#trades-table tbody');

    if (appState.trades.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="8" class="text-center text-muted py-5">
                    <i class="fas fa-inbox fa-3x mb-3 d-block opacity-25"></i>
                    <p class="mb-0">Aucun trade récent</p>
                </td>
            </tr>
        `;
        return;
    }

    tbody.innerHTML = '';

    appState.trades.slice(0, 20).forEach(trade => {
        const row = createTradeRow(trade);
        tbody.appendChild(row);
    });
}

function addTradeToTable(trade) {
    const tbody = document.querySelector('#trades-table tbody');

    // Remove "no trades" message
    const noTradesRow = tbody.querySelector('td[colspan="8"]');
    if (noTradesRow) {
        tbody.innerHTML = '';
    }

    // Add new trade at the top
    const row = createTradeRow(trade);
    row.classList.add('new-trade');
    tbody.insertBefore(row, tbody.firstChild);

    // Limit to 20 trades displayed
    while (tbody.children.length > 20) {
        tbody.removeChild(tbody.lastChild);
    }

    // Add to state
    appState.trades.unshift(trade);
}

function createTradeRow(trade) {
    const row = document.createElement('tr');
    row.classList.add('trade-row');

    const typeClass = trade.action === 'long' ? 'trade-type-long' : 'trade-type-short';
    const typeIcon = trade.action === 'long' ? 'fa-arrow-up' : 'fa-arrow-down';

    const pnl = trade.pnl || 0;
    const pnlClass = pnl > 0 ? 'metric-positive' : pnl < 0 ? 'metric-negative' : '';
    const pnlSign = pnl > 0 ? '+' : '';

    row.innerHTML = `
        <td>${formatTimestamp(trade.timestamp)}</td>
        <td class="${typeClass}">
            <i class="fas ${typeIcon}"></i> ${trade.action.toUpperCase()}
        </td>
        <td>${trade.size ? trade.size.toFixed(2) : '-'}</td>
        <td>${trade.price ? trade.price.toFixed(5) : '-'}</td>
        <td>${trade.sl ? trade.sl.toFixed(5) : '-'}</td>
        <td>${trade.tp ? trade.tp.toFixed(5) : '-'}</td>
        <td class="${pnlClass}">${pnlSign}${formatCurrency(pnl)}</td>
        <td><span class="badge ${trade.status === 'closed' ? 'bg-secondary' : 'bg-info'}">${trade.status || 'Ouvert'}</span></td>
    `;

    return row;
}

// ============================================================================
// CHARTS
// ============================================================================

function initEquityChart() {
    const ctx = document.getElementById('equity-chart').getContext('2d');

    const gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, 'rgba(102, 126, 234, 0.5)');
    gradient.addColorStop(1, 'rgba(102, 126, 234, 0.0)');

    equityChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Equity',
                data: [],
                borderColor: chartColors.primary,
                backgroundColor: gradient,
                borderWidth: 3,
                tension: 0.4,
                fill: true,
                pointRadius: 0,
                pointHoverRadius: 6,
                pointHoverBackgroundColor: chartColors.primary,
                pointHoverBorderColor: '#fff',
                pointHoverBorderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(45, 55, 72, 0.95)',
                    padding: 12,
                    titleFont: { size: 14, weight: 'bold' },
                    bodyFont: { size: 13 },
                    borderColor: chartColors.primary,
                    borderWidth: 1,
                    displayColors: false,
                    callbacks: {
                        label: function(context) {
                            return 'Equity: ' + formatCurrency(context.parsed.y);
                        }
                    }
                }
            },
            scales: {
                x: {
                    display: true,
                    grid: {
                        display: false
                    },
                    ticks: {
                        font: { size: 11 },
                        maxRotation: 0
                    }
                },
                y: {
                    display: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    },
                    ticks: {
                        font: { size: 11 },
                        callback: function(value) {
                            return '$' + value.toLocaleString();
                        }
                    }
                }
            }
        }
    });
}

function updateEquityChart(equityData) {
    if (!equityChart || !equityData || equityData.length === 0) return;

    const labels = equityData.map(point => formatTimestamp(point.timestamp, 'short'));
    const data = equityData.map(point => point.equity);

    equityChart.data.labels = labels;
    equityChart.data.datasets[0].data = data;
    equityChart.update('none'); // No animation for smoother updates
}

function initPnlDistributionChart() {
    const ctx = document.getElementById('pnl-distribution-chart').getContext('2d');

    pnlDistributionChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Nombre de Trades',
                data: [],
                backgroundColor: function(context) {
                    const value = context.parsed.y;
                    return value >= 0 ? 'rgba(17, 153, 142, 0.8)' : 'rgba(255, 107, 107, 0.8)';
                },
                borderColor: function(context) {
                    const value = context.parsed.y;
                    return value >= 0 ? 'rgba(17, 153, 142, 1)' : 'rgba(255, 107, 107, 1)';
                },
                borderWidth: 2,
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(45, 55, 72, 0.95)',
                    padding: 12,
                    borderColor: chartColors.primary,
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { font: { size: 11 } }
                },
                y: {
                    grid: { color: 'rgba(0, 0, 0, 0.05)' },
                    ticks: { font: { size: 11 } }
                }
            }
        }
    });
}

function updatePnlDistributionChart(distribution) {
    if (!pnlDistributionChart || !distribution) return;

    pnlDistributionChart.data.labels = distribution.labels || [];
    pnlDistributionChart.data.datasets[0].data = distribution.data || [];
    pnlDistributionChart.update('none');
}

function initPositionSizeChart() {
    const ctx = document.getElementById('position-size-chart').getContext('2d');

    positionSizeChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Taille Position',
                data: [],
                borderColor: chartColors.warning,
                backgroundColor: 'rgba(240, 147, 251, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                fill: true,
                pointRadius: 2,
                pointHoverRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(45, 55, 72, 0.95)',
                    padding: 12,
                    borderColor: chartColors.warning,
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { font: { size: 11 } }
                },
                y: {
                    grid: { color: 'rgba(0, 0, 0, 0.05)' },
                    ticks: { font: { size: 11 } }
                }
            }
        }
    });
}

function updatePositionSizeChart(positions) {
    if (!positionSizeChart || !positions || positions.length === 0) return;

    const labels = positions.map(p => formatTimestamp(p.timestamp, 'short'));
    const data = positions.map(p => Math.abs(p.size));

    positionSizeChart.data.labels = labels;
    positionSizeChart.data.datasets[0].data = data;
    positionSizeChart.update('none');
}

// ============================================================================
// EVENT LISTENERS
// ============================================================================

function setupEventListeners() {
    // Start trading button
    document.getElementById('btn-start-trading').addEventListener('click', async function() {
        if (!appState.currentModel) {
            showNotification('Erreur', 'Aucun modèle chargé. Veuillez charger un modèle d\'abord.', 'danger');
            return;
        }

        if (confirm('Êtes-vous sûr de vouloir démarrer le trading live ?')) {
            showLoadingOverlay('Démarrage du trading...');
            await startTrading();
            hideLoadingOverlay();
        }
    });

    // Stop trading button
    document.getElementById('btn-stop-trading').addEventListener('click', async function() {
        if (confirm('Êtes-vous sûr de vouloir arrêter le trading ?')) {
            showLoadingOverlay('Arrêt du trading...');
            await stopTrading();
            hideLoadingOverlay();
        }
    });

    // Refresh trades button
    document.getElementById('btn-refresh-trades').addEventListener('click', async function() {
        const btn = this;
        const icon = btn.querySelector('i');
        icon.classList.add('fa-spin');
        await loadTrades();
        setTimeout(() => icon.classList.remove('fa-spin'), 500);
    });

    // Period selection buttons
    document.querySelectorAll('[data-period]').forEach(btn => {
        btn.addEventListener('click', function() {
            // Update active state
            document.querySelectorAll('[data-period]').forEach(b => b.classList.remove('active'));
            this.classList.add('active');

            // Update period and reload data
            appState.selectedPeriod = this.dataset.period;
            loadEquityCurve();
        });
    });
}

// ============================================================================
// API CALLS
// ============================================================================

async function startTrading() {
    try {
        const response = await fetch('/api/trading/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        const data = await response.json();

        if (data.success) {
            appState.isTrading = true;
            updateTradingButtons();
            showNotification('Succès', 'Trading démarré avec succès', 'success');
        } else {
            showNotification('Erreur', data.error || 'Échec du démarrage', 'danger');
        }
    } catch (error) {
        console.error('Error starting trading:', error);
        showNotification('Erreur', 'Impossible de démarrer le trading', 'danger');
    }
}

async function stopTrading() {
    try {
        const response = await fetch('/api/trading/stop', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        const data = await response.json();

        if (data.success) {
            appState.isTrading = false;
            updateTradingButtons();
            showNotification('Succès', 'Trading arrêté', 'info');
        } else {
            showNotification('Erreur', data.error || 'Échec de l\'arrêt', 'danger');
        }
    } catch (error) {
        console.error('Error stopping trading:', error);
        showNotification('Erreur', 'Impossible d\'arrêter le trading', 'danger');
    }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(value);
}

function formatTimestamp(timestamp, format = 'full') {
    const date = new Date(timestamp);

    if (format === 'short') {
        return date.toLocaleString('fr-FR', {
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit'
        });
    }

    return date.toLocaleString('fr-FR', {
        day: '2-digit',
        month: '2-digit',
        year: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
}

function showNotification(title, message, type = 'info') {
    const typeColors = {
        success: 'bg-success',
        danger: 'bg-danger',
        warning: 'bg-warning',
        info: 'bg-info'
    };

    const toastHtml = `
        <div class="toast align-items-center text-white ${typeColors[type]} border-0" role="alert" aria-live="assertive" aria-atomic="true">
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

function showLoadingOverlay(message = 'Chargement...') {
    let overlay = document.querySelector('.loading-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.className = 'loading-overlay';
        overlay.innerHTML = `
            <div class="loading-spinner"></div>
            <div class="loading-text">${message}</div>
        `;
        document.body.appendChild(overlay);
    } else {
        overlay.querySelector('.loading-text').textContent = message;
        overlay.style.display = 'flex';
    }
}

function hideLoadingOverlay() {
    const overlay = document.querySelector('.loading-overlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
}

// ============================================================================
// PERIODIC UPDATES
// ============================================================================

function startPeriodicUpdates() {
    // Update status every 5 seconds
    updateInterval = setInterval(async function() {
        if (appState.isTrading) {
            await loadStatus();
            await loadPerformance();
            await loadEquityCurve();
        }
    }, 5000);

    // Update trades every 30 seconds
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

// ============================================================================
// EXPORT FOR DEBUGGING
// ============================================================================

window.dashboardDebug = {
    appState,
    charts: { equityChart, pnlDistributionChart, positionSizeChart },
    reload: () => {
        loadStatus();
        loadPerformance();
        loadEquityCurve();
        loadTrades();
    }
};

console.log('💎 Premium Dashboard ready! Type dashboardDebug in console for debug tools.');
