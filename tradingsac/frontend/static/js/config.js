/**
 * config.js
 * Gestion de la configuration du système
 */

// Configuration actuelle
let currentConfig = {};

// ============================================================================
// INITIALISATION
// ============================================================================

document.addEventListener('DOMContentLoaded', function() {
    console.log('Page configuration initialisée');
    
    // Charger la configuration
    loadConfiguration();
    
    // Setup event listeners
    setupEventListeners();
});

// ============================================================================
// CHARGEMENT DE LA CONFIGURATION
// ============================================================================

async function loadConfiguration() {
    try {
        const response = await fetch('/api/config');
        const data = await response.json();
        
        currentConfig = data;
        populateForm(data);
    } catch (error) {
        console.error('Erreur chargement configuration:', error);
        showStatus('Erreur de chargement de la configuration', 'danger');
    }
}

function populateForm(config) {
    // Data configuration
    document.getElementById('data-dir').value = config.data?.data_dir || 'data';
    document.getElementById('timeframe').value = config.data?.timeframe || '5min';
    
    // Trading configuration
    document.getElementById('initial-capital').value = config.trading?.initial_capital || 10000;
    document.getElementById('risk-per-trade').value = (config.trading?.risk_per_trade || 0.02) * 100;
    document.getElementById('max-daily-loss').value = (config.trading?.max_daily_loss || 0.05) * 100;
    document.getElementById('max-drawdown').value = (config.trading?.max_drawdown || 0.15) * 100;
    
    // Model configuration
    document.getElementById('checkpoint-dir').value = config.model?.checkpoint_dir || 'models/checkpoints';
    document.getElementById('production-dir').value = config.model?.production_dir || 'models/production';
    document.getElementById('ensemble-size').value = config.model?.ensemble_size || 3;
    
    // MT5 configuration
    document.getElementById('mt5-server').value = config.mt5?.server || 'ICMarketsEU-Demo';
    document.getElementById('mt5-symbol').value = config.mt5?.symbol || 'EURUSD';
    document.getElementById('mt5-login').value = config.mt5?.login || 0;
    document.getElementById('mt5-password').value = config.mt5?.password || '';
}

// ============================================================================
// EVENT LISTENERS
// ============================================================================

function setupEventListeners() {
    // Formulaire de configuration
    document.getElementById('config-form').addEventListener('submit', async function(e) {
        e.preventDefault();
        await saveConfiguration();
    });
    
    // Bouton réinitialiser
    document.getElementById('btn-reset').addEventListener('click', function() {
        if (confirm('Êtes-vous sûr de vouloir réinitialiser la configuration ?')) {
            populateForm(currentConfig);
            showStatus('Configuration réinitialisée', 'info');
        }
    });
    
    // Bouton exporter
    document.getElementById('btn-export').addEventListener('click', function() {
        exportConfiguration();
    });
}

// ============================================================================
// SAUVEGARDE DE LA CONFIGURATION
// ============================================================================

async function saveConfiguration() {
    try {
        // Récupérer les valeurs du formulaire
        const newConfig = {
            data: {
                data_dir: document.getElementById('data-dir').value,
                timeframe: document.getElementById('timeframe').value,
                pairs: currentConfig.data?.pairs || ['EURUSD', 'USDJPY', 'GBPUSD', 'USDCAD']
            },
            trading: {
                initial_capital: parseFloat(document.getElementById('initial-capital').value),
                risk_per_trade: parseFloat(document.getElementById('risk-per-trade').value) / 100,
                max_daily_loss: parseFloat(document.getElementById('max-daily-loss').value) / 100,
                max_drawdown: parseFloat(document.getElementById('max-drawdown').value) / 100
            },
            model: {
                checkpoint_dir: document.getElementById('checkpoint-dir').value,
                production_dir: document.getElementById('production-dir').value,
                ensemble_size: parseInt(document.getElementById('ensemble-size').value)
            },
            mt5: {
                server: document.getElementById('mt5-server').value,
                symbol: document.getElementById('mt5-symbol').value,
                login: parseInt(document.getElementById('mt5-login').value),
                password: document.getElementById('mt5-password').value
            }
        };
        
        // Valider la configuration
        if (!validateConfiguration(newConfig)) {
            return;
        }
        
        // Envoyer au serveur
        const response = await fetch('/api/config', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(newConfig)
        });
        
        const data = await response.json();
        
        if (data.success) {
            currentConfig = newConfig;
            showStatus('Configuration sauvegardée avec succès', 'success');
        } else {
            showStatus('Erreur: ' + data.error, 'danger');
        }
    } catch (error) {
        console.error('Erreur sauvegarde configuration:', error);
        showStatus('Erreur lors de la sauvegarde', 'danger');
    }
}

// ============================================================================
// VALIDATION
// ============================================================================

function validateConfiguration(config) {
    const errors = [];
    
    // Validation trading
    if (config.trading.initial_capital <= 0) {
        errors.push('Le capital initial doit être positif');
    }
    
    if (config.trading.risk_per_trade <= 0 || config.trading.risk_per_trade > 0.1) {
        errors.push('Le risque par trade doit être entre 0% et 10%');
    }
    
    if (config.trading.max_daily_loss <= 0 || config.trading.max_daily_loss > 0.5) {
        errors.push('La perte journalière max doit être entre 0% et 50%');
    }
    
    if (config.trading.max_drawdown <= 0 || config.trading.max_drawdown > 0.5) {
        errors.push('Le drawdown max doit être entre 0% et 50%');
    }
    
    // Validation model
    if (config.model.ensemble_size < 1 || config.model.ensemble_size > 10) {
        errors.push('La taille de l\'ensemble doit être entre 1 et 10');
    }
    
    // Validation MT5
    if (!config.mt5.server) {
        errors.push('Le serveur MT5 est requis');
    }
    
    if (!config.mt5.symbol) {
        errors.push('Le symbole de trading est requis');
    }
    
    if (errors.length > 0) {
        showStatus('Erreurs de validation:\n' + errors.join('\n'), 'danger');
        return false;
    }
    
    return true;
}

// ============================================================================
// EXPORT
// ============================================================================

function exportConfiguration() {
    const config = {
        data: {
            data_dir: document.getElementById('data-dir').value,
            timeframe: document.getElementById('timeframe').value,
            pairs: currentConfig.data?.pairs || []
        },
        trading: {
            initial_capital: parseFloat(document.getElementById('initial-capital').value),
            risk_per_trade: parseFloat(document.getElementById('risk-per-trade').value) / 100,
            max_daily_loss: parseFloat(document.getElementById('max-daily-loss').value) / 100,
            max_drawdown: parseFloat(document.getElementById('max-drawdown').value) / 100
        },
        model: {
            checkpoint_dir: document.getElementById('checkpoint-dir').value,
            production_dir: document.getElementById('production-dir').value,
            ensemble_size: parseInt(document.getElementById('ensemble-size').value)
        },
        mt5: {
            server: document.getElementById('mt5-server').value,
            symbol: document.getElementById('mt5-symbol').value,
            login: parseInt(document.getElementById('mt5-login').value),
            password: '***' // Ne pas exporter le mot de passe
        }
    };
    
    const yamlString = convertToYAML(config);
    const blob = new Blob([yamlString], { type: 'text/yaml' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = 'system_config.yaml';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    showStatus('Configuration exportée', 'success');
}

function convertToYAML(obj, indent = 0) {
    let yaml = '';
    const spaces = '  '.repeat(indent);
    
    for (const key in obj) {
        const value = obj[key];
        
        if (typeof value === 'object' && !Array.isArray(value)) {
            yaml += `${spaces}${key}:\n`;
            yaml += convertToYAML(value, indent + 1);
        } else if (Array.isArray(value)) {
            yaml += `${spaces}${key}:\n`;
            value.forEach(item => {
                yaml += `${spaces}  - ${item}\n`;
            });
        } else {
            yaml += `${spaces}${key}: ${value}\n`;
        }
    }
    
    return yaml;
}

// ============================================================================
// INTERFACE
// ============================================================================

function showStatus(message, type = 'info') {
    const statusDiv = document.getElementById('config-status');
    const alert = statusDiv.querySelector('.alert');
    
    alert.className = `alert alert-${type}`;
    alert.innerHTML = `<i class="fas fa-info-circle"></i> ${message.replace(/\n/g, '<br>')}`;
    
    statusDiv.style.display = 'block';
    
    // Masquer après 5 secondes pour les succès
    if (type === 'success' || type === 'info') {
        setTimeout(() => {
            statusDiv.style.display = 'none';
        }, 5000);
    }
}

// ============================================================================
// VALIDATION EN TEMPS RÉEL
// ============================================================================

// Ajouter des validations en temps réel sur les champs
document.addEventListener('DOMContentLoaded', function() {
    // Capital initial
    const initialCapital = document.getElementById('initial-capital');
    initialCapital.addEventListener('blur', function() {
        if (parseFloat(this.value) <= 0) {
            this.classList.add('is-invalid');
        } else {
            this.classList.remove('is-invalid');
        }
    });
    
    // Risque par trade
    const riskPerTrade = document.getElementById('risk-per-trade');
    riskPerTrade.addEventListener('blur', function() {
        const value = parseFloat(this.value);
        if (value <= 0 || value > 10) {
            this.classList.add('is-invalid');
        } else {
            this.classList.remove('is-invalid');
        }
    });
    
    // Perte journalière max
    const maxDailyLoss = document.getElementById('max-daily-loss');
    maxDailyLoss.addEventListener('blur', function() {
        const value = parseFloat(this.value);
        if (value <= 0 || value > 50) {
            this.classList.add('is-invalid');
        } else {
            this.classList.remove('is-invalid');
        }
    });
    
    // Drawdown max
    const maxDrawdown = document.getElementById('max-drawdown');
    maxDrawdown.addEventListener('blur', function() {
        const value = parseFloat(this.value);
        if (value <= 0 || value > 50) {
            this.classList.add('is-invalid');
        } else {
            this.classList.remove('is-invalid');
        }
    });
    
    // Ensemble size
    const ensembleSize = document.getElementById('ensemble-size');
    ensembleSize.addEventListener('blur', function() {
        const value = parseInt(this.value);
        if (value < 1 || value > 10) {
            this.classList.add('is-invalid');
        } else {
            this.classList.remove('is-invalid');
        }
    });
});
