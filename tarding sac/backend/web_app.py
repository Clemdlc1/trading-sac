"""
web_app.py

Interface web complète pour le système de trading SAC EUR/USD.
Fournit un dashboard temps réel, gestion des modèles, configuration et monitoring.

Features:
- Dashboard temps réel avec SocketIO
- REST API pour contrôle du système
- Gestion des modèles (load, train, retrain)
- Éditeur de configuration
- Visualisation des performances
- Monitoring des trades live

Author: Clément
Version: 1.0
"""

import os
import sys
import json
import yaml
import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import traceback

import numpy as np
import pandas as pd
import torch
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import plotly
import plotly.graph_objs as go

# Import modules du système
from backend.data_pipeline import DataPipeline
from backend.feature_engineering import FeatureEngineer
from backend.hmm_detector import HMMRegimeDetector
from backend.trading_env import TradingEnvironment
from backend.sac_agent import SACAgent, SACConfig
from backend.auxiliary_task import AuxiliaryTaskLearner
from backend.ensemble_meta import EnsembleMetaController
from backend.validation import ValidationFramework
from backend.mt5_connector import MT5Connector
from backend.risk_manager import RiskManager

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/web_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
TEMPLATE_DIR = FRONTEND_DIR / "templates"
STATIC_DIR = FRONTEND_DIR / "static"

# Configuration Flask
app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR)
)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
CORS(app)

# Configuration SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Variables globales pour le système
class TradingSystemState:
    """État global du système de trading"""
    def __init__(self):
        self.is_training = False
        self.is_trading = False
        self.current_model = None
        self.ensemble_controller = None
        self.mt5_connector = None
        self.risk_manager = None
        self.data_pipeline = None
        self.feature_engineer = None
        self.hmm_detector = None

        # Métriques en temps réel
        self.current_equity = 10000.0
        self.current_position = 0.0
        self.current_pnl = 0.0
        self.daily_trades = []
        self.performance_metrics = {}

        # Historique
        self.equity_history = []
        self.trade_history = []
        self.signal_history = []

        # Configuration
        self.config = self.load_config()

        # Threading
        self.trading_thread = None
        self.monitoring_thread = None
        self.stop_event = threading.Event()

        # État du training
        self.training_state = self.load_training_state()
        
    def load_config(self) -> Dict:
        """Charge la configuration depuis YAML"""
        config_path = Path('config/system_config.yaml')
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Configuration par défaut
            return {
                'data': {
                    'data_dir': 'data',
                    'timeframe': '5min',
                    'pairs': ['EURUSD', 'USDJPY', 'GBPUSD', 'USDCAD']
                },
                'trading': {
                    'initial_capital': 10000.0,
                    'risk_per_trade': 0.02,
                    'max_daily_loss': 0.05,
                    'max_drawdown': 0.15
                },
                'model': {
                    'checkpoint_dir': 'models/checkpoints',
                    'production_dir': 'models/production',
                    'ensemble_size': 3
                },
                'mt5': {
                    'server': 'ICMarketsEU-Demo',
                    'login': 0,
                    'password': '',
                    'symbol': 'EURUSD'
                }
            }
    
    def save_config(self):
        """Sauvegarde la configuration"""
        config_path = Path('config/system_config.yaml')
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)

    def load_training_state(self) -> Dict:
        """Charge l'état du training depuis JSON"""
        state_path = Path('logs/training_state.json')
        if state_path.exists():
            with open(state_path, 'r') as f:
                return json.load(f)
        return {
            'is_training': False,
            'current_agent': None,
            'current_episode': 0,
            'total_episodes': 0,
            'start_time': None,
            'metrics': {},
            'training_type': None  # 'sac_agent' or 'meta_controller'
        }

    def save_training_state(self):
        """Sauvegarde l'état du training"""
        state_path = Path('logs/training_state.json')
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(state_path, 'w') as f:
            json.dump(self.training_state, f, indent=2)

# État global
system_state = TradingSystemState()

# ============================================================================
# ROUTES WEB INTERFACE
# ============================================================================

@app.route('/')
def index():
    """Page principale du dashboard"""
    return render_template('index.html')

@app.route('/models')
def models_page():
    """Page de gestion des modèles"""
    return render_template('models.html')

@app.route('/config')
def config_page():
    """Page de configuration"""
    return render_template('config.html')

@app.route('/validation')
def validation_page():
    """Page de validation et backtesting"""
    return render_template('validation.html')

@app.route('/dataset')
def dataset_page():
    """Page de gestion du dataset"""
    return render_template('dataset.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Servir les fichiers statiques"""
    return send_from_directory(app.static_folder, filename)

# ============================================================================
# REST API ENDPOINTS
# ============================================================================

@app.route('/api/status', methods=['GET'])
def get_status():
    """Obtenir le statut du système"""
    return jsonify({
        'is_training': system_state.is_training,
        'is_trading': system_state.is_trading,
        'current_model': system_state.current_model,
        'equity': system_state.current_equity,
        'position': system_state.current_position,
        'pnl': system_state.current_pnl,
        'daily_trades': len(system_state.daily_trades),
        'training_state': system_state.training_state,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/config', methods=['GET', 'POST'])
def manage_config():
    """Gérer la configuration du système"""
    if request.method == 'GET':
        return jsonify(system_state.config)
    
    elif request.method == 'POST':
        try:
            new_config = request.json
            system_state.config.update(new_config)
            system_state.save_config()
            
            # Émettre notification de mise à jour
            socketio.emit('config_updated', new_config)
            
            return jsonify({
                'success': True,
                'message': 'Configuration mise à jour avec succès'
            })
        except Exception as e:
            logger.error(f"Erreur mise à jour configuration: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

@app.route('/api/models/list', methods=['GET'])
def list_models():
    """Lister les modèles disponibles"""
    try:
        checkpoint_dir = Path(system_state.config['model']['checkpoint_dir'])
        production_dir = Path(system_state.config['model']['production_dir'])
        
        checkpoints = []
        if checkpoint_dir.exists():
            for path in checkpoint_dir.glob('*.pt'):
                stat = path.stat()
                checkpoints.append({
                    'name': path.stem,
                    'path': path.name,  # Envoyer seulement le nom du fichier
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'type': 'checkpoint'
                })

        production = []
        if production_dir.exists():
            for path in production_dir.glob('*.pt'):
                stat = path.stat()
                production.append({
                    'name': path.stem,
                    'path': path.name,  # Envoyer seulement le nom du fichier
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'type': 'production'
                })
        
        return jsonify({
            'checkpoints': checkpoints,
            'production': production
        })
    
    except Exception as e:
        logger.error(f"Erreur listage modèles: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/load', methods=['POST'])
def load_model():
    """Charger un modèle"""
    try:
        data = request.json
        model_filename = data.get('model_path')  # C'est maintenant juste le nom du fichier

        if not model_filename:
            return jsonify({
                'success': False,
                'error': 'Nom de modèle manquant'
            }), 400

        # Chercher le fichier dans checkpoint_dir et production_dir
        checkpoint_dir = Path(system_state.config['model']['checkpoint_dir'])
        production_dir = Path(system_state.config['model']['production_dir'])

        model_path = None
        if (checkpoint_dir / model_filename).exists():
            model_path = checkpoint_dir / model_filename
        elif (production_dir / model_filename).exists():
            model_path = production_dir / model_filename

        if not model_path:
            return jsonify({
                'success': False,
                'error': 'Modèle non trouvé'
            }), 404
        
        # Initialiser l'ensemble controller
        logger.info(f"Chargement du modèle: {model_path}")
        
        # Charger le data pipeline et feature engineer
        system_state.data_pipeline = DataPipeline()
        
        system_state.feature_engineer = FeatureEngineer()
        
        # Charger l'ensemble
        system_state.ensemble_controller = EnsembleMetaController(
            state_dim=30,
            action_dim=1,
            num_agents=system_state.config['model']['ensemble_size']
        )
        
        # Charger depuis checkpoint
        # PyTorch 2.6+ nécessite weights_only=False pour charger des classes personnalisées
        checkpoint = torch.load(model_path, weights_only=False)
        system_state.ensemble_controller.load(model_path)
        
        system_state.current_model = Path(model_path).stem
        
        # Émettre notification
        socketio.emit('model_loaded', {
            'model_name': system_state.current_model,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"Modèle chargé avec succès: {system_state.current_model}")
        
        return jsonify({
            'success': True,
            'message': f'Modèle {system_state.current_model} chargé avec succès'
        })
    
    except Exception as e:
        logger.error(f"Erreur chargement modèle: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models/delete', methods=['POST'])
def delete_model():
    """Supprimer un modèle"""
    try:
        data = request.json
        model_filename = data.get('model_path')  # C'est le nom du fichier

        if not model_filename:
            return jsonify({
                'success': False,
                'error': 'Nom de modèle manquant'
            }), 400

        # Chercher le fichier dans checkpoint_dir et production_dir
        checkpoint_dir = Path(system_state.config['model']['checkpoint_dir'])
        production_dir = Path(system_state.config['model']['production_dir'])

        model_path = None
        if (checkpoint_dir / model_filename).exists():
            model_path = checkpoint_dir / model_filename
        elif (production_dir / model_filename).exists():
            model_path = production_dir / model_filename

        if not model_path:
            return jsonify({
                'success': False,
                'error': 'Modèle non trouvé'
            }), 404

        # Supprimer le fichier
        model_path.unlink()
        logger.info(f"Modèle supprimé: {model_path}")

        return jsonify({
            'success': True,
            'message': f'Modèle {model_filename} supprimé avec succès'
        })

    except Exception as e:
        logger.error(f"Erreur suppression modèle: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Démarrer l'entraînement"""
    if system_state.is_training:
        return jsonify({
            'success': False,
            'error': 'Entraînement déjà en cours'
        }), 400

    try:
        data = request.json

        # Paramètres d'entraînement
        num_episodes = data.get('num_episodes', 1000)
        batch_size = data.get('batch_size', 512)  # Augmenté pour mieux utiliser le GPU
        from_checkpoint = data.get('from_checkpoint', None)
        agent_id = data.get('agent_id', None)  # None = tous les agents, sinon l'ID spécifique

        # Initialiser l'état du training
        system_state.training_state = {
            'is_training': True,
            'current_agent': agent_id if agent_id is not None else 'all',
            'current_episode': 0,
            'total_episodes': num_episodes,
            'start_time': datetime.now().isoformat(),
            'metrics': {},
            'training_type': 'sac_agent'
        }
        system_state.save_training_state()

        # Lancer l'entraînement dans un thread séparé
        training_thread = threading.Thread(
            target=run_training,
            args=(num_episodes, batch_size, from_checkpoint, agent_id)
        )
        training_thread.daemon = True
        training_thread.start()

        system_state.training_thread = training_thread
        system_state.is_training = True

        return jsonify({
            'success': True,
            'message': f'Entraînement démarré (Agent: {agent_id if agent_id is not None else "tous"})'
        })

    except Exception as e:
        logger.error(f"Erreur démarrage entraînement: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    """Arrêter l'entraînement"""
    if not system_state.is_training:
        return jsonify({
            'success': False,
            'error': 'Aucun entraînement en cours'
        }), 400

    system_state.stop_event.set()
    system_state.is_training = False

    # Mettre à jour l'état du training
    system_state.training_state['is_training'] = False
    system_state.save_training_state()

    return jsonify({
        'success': True,
        'message': 'Entraînement arrêté'
    })

@app.route('/api/training/meta-controller/start', methods=['POST'])
def start_meta_controller_training():
    """Démarrer l'entraînement du meta-controller"""
    if system_state.is_training:
        return jsonify({
            'success': False,
            'error': 'Entraînement déjà en cours'
        }), 400

    try:
        data = request.json

        # Paramètres d'entraînement
        num_episodes = data.get('num_episodes', 500)
        batch_size = data.get('batch_size', 256)

        # Initialiser l'état du training
        system_state.training_state = {
            'is_training': True,
            'current_agent': 'meta_controller',
            'current_episode': 0,
            'total_episodes': num_episodes,
            'start_time': datetime.now().isoformat(),
            'metrics': {},
            'training_type': 'meta_controller'
        }
        system_state.save_training_state()

        # Lancer l'entraînement dans un thread séparé
        training_thread = threading.Thread(
            target=run_meta_controller_training,
            args=(num_episodes, batch_size)
        )
        training_thread.daemon = True
        training_thread.start()

        system_state.training_thread = training_thread
        system_state.is_training = True

        return jsonify({
            'success': True,
            'message': 'Entraînement du meta-controller démarré'
        })

    except Exception as e:
        logger.error(f"Erreur démarrage entraînement meta-controller: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/trading/start', methods=['POST'])
def start_trading():
    """Démarrer le trading live"""
    if system_state.is_trading:
        return jsonify({
            'success': False,
            'error': 'Trading déjà actif'
        }), 400
    
    if not system_state.current_model:
        return jsonify({
            'success': False,
            'error': 'Aucun modèle chargé'
        }), 400
    
    try:
        # Initialiser MT5
        from backend.mt5_connector import MT5Config
        mt5_config = MT5Config(
            account=system_state.config.get('mt5', {}).get('login', 0),
            password=system_state.config.get('mt5', {}).get('password', ''),
            server=system_state.config.get('mt5', {}).get('server', '')
        )
        system_state.mt5_connector = MT5Connector(config=mt5_config)
        
        if not system_state.mt5_connector.connect():
            return jsonify({
                'success': False,
                'error': 'Impossible de se connecter à MT5'
            }), 500
        
        # Initialiser risk manager
        from backend.risk_manager import RiskConfig
        risk_config = RiskConfig(
            risk_per_trade=system_state.config.get('trading', {}).get('risk_per_trade', 0.02),
            daily_loss_limit=system_state.config.get('trading', {}).get('max_daily_loss', 0.05),
            max_drawdown_limit=system_state.config.get('trading', {}).get('max_drawdown', 0.15)
        )
        system_state.risk_manager = RiskManager(
            initial_equity=system_state.config.get('trading', {}).get('initial_capital', 100000.0),
            config=risk_config
        )
        
        # Lancer le trading dans un thread
        trading_thread = threading.Thread(target=run_trading)
        trading_thread.daemon = True
        trading_thread.start()
        
        system_state.trading_thread = trading_thread
        system_state.is_trading = True
        
        return jsonify({
            'success': True,
            'message': 'Trading démarré'
        })
    
    except Exception as e:
        logger.error(f"Erreur démarrage trading: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/dataset/status', methods=['GET'])
def dataset_status():
    """Obtenir le statut du dataset"""
    try:
        data_pipeline = DataPipeline()
        processed_file = data_pipeline.config.processed_data_dir / "processed_data.h5"
        
        status = {
            'processed_exists': processed_file.exists(),
            'processed_path': str(processed_file),
            'raw_data_dir': str(data_pipeline.config.raw_data_dir),
            'processed_data_dir': str(data_pipeline.config.processed_data_dir),
            'pairs': data_pipeline.config.pairs
        }
        
        if processed_file.exists():
            import h5py
            with h5py.File(processed_file, 'r') as f:
                if 'train' in f:
                    train_pairs = list(f['train'].keys())
                    status['train_pairs'] = train_pairs
                    if train_pairs:
                        first_pair = train_pairs[0]
                        status['train_samples'] = len(f[f'train/{first_pair}'])
                if 'validation' in f:
                    val_pairs = list(f['validation'].keys())
                    status['val_pairs'] = val_pairs
                if 'test' in f:
                    test_pairs = list(f['test'].keys())
                    status['test_pairs'] = test_pairs
        
        return jsonify(status)
    
    except Exception as e:
        logger.error(f"Erreur statut dataset: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/dataset/prepare', methods=['POST'])
def prepare_dataset():
    """Préparer le dataset (téléchargement et traitement)"""
    try:
        data = request.json
        force_download = data.get('force_download', False)
        
        def run_preparation():
            try:
                data_pipeline = DataPipeline()
                train_data, val_data, test_data = data_pipeline.run_full_pipeline(
                    force_download=force_download
                )
                
                socketio.emit('dataset_preparation_complete', {
                    'success': True,
                    'train_samples': len(next(iter(train_data.values()))) if train_data else 0,
                    'val_samples': len(next(iter(val_data.values()))) if val_data else 0,
                    'test_samples': len(next(iter(test_data.values()))) if test_data else 0,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Erreur préparation dataset: {e}")
                socketio.emit('dataset_preparation_error', {
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        # Lancer dans un thread séparé
        prep_thread = threading.Thread(target=run_preparation, daemon=True)
        prep_thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Préparation du dataset démarrée'
        })
    
    except Exception as e:
        logger.error(f"Erreur démarrage préparation: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/trading/stop', methods=['POST'])
def stop_trading():
    """Arrêter le trading live"""
    if not system_state.is_trading:
        return jsonify({
            'success': False,
            'error': 'Trading non actif'
        }), 400
    
    system_state.stop_event.set()
    system_state.is_trading = False
    
    # Fermer toutes les positions
    if system_state.mt5_connector:
        system_state.mt5_connector.close_all_positions()
        system_state.mt5_connector.disconnect()
    
    return jsonify({
        'success': True,
        'message': 'Trading arrêté'
    })

@app.route('/api/performance', methods=['GET'])
def get_performance():
    """Obtenir les métriques de performance"""
    return jsonify(system_state.performance_metrics)

@app.route('/api/trades', methods=['GET'])
def get_trades():
    """Obtenir l'historique des trades"""
    limit = request.args.get('limit', 100, type=int)
    return jsonify({
        'trades': system_state.trade_history[-limit:]
    })

@app.route('/api/equity-curve', methods=['GET'])
def get_equity_curve():
    """Obtenir la courbe d'equity"""
    return jsonify({
        'equity': system_state.equity_history
    })

@app.route('/api/training/metrics', methods=['GET'])
def get_training_metrics():
    """Obtenir l'historique complet des métriques d'entraînement"""
    try:
        metrics_file = Path('logs/training_metrics.json')

        if not metrics_file.exists():
            return jsonify({
                'success': False,
                'error': 'Aucune métrique d\'entraînement trouvée'
            }), 404

        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        # Statistiques sur les métriques
        num_episodes = len(metrics.get('episodes', []))

        return jsonify({
            'success': True,
            'num_episodes': num_episodes,
            'metrics': metrics,
            'available_metrics': list(metrics.keys()),
            'file_path': str(metrics_file),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Erreur récupération métriques: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/validation/run', methods=['POST'])
def run_validation():
    """Exécuter une validation walk-forward"""
    try:
        data = request.json
        model_path = data.get('model_path')
        
        if not model_path:
            return jsonify({
                'success': False,
                'error': 'Chemin du modèle requis'
            }), 400
        
        # Créer le validateur
        validator = ValidationFramework(
            data_dir=system_state.config['data']['data_dir']
        )
        
        # Lancer la validation dans un thread
        validation_thread = threading.Thread(
            target=run_validation_process,
            args=(validator, model_path)
        )
        validation_thread.daemon = True
        validation_thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Validation démarrée'
        })
    
    except Exception as e:
        logger.error(f"Erreur validation: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============================================================================
# SOCKETIO HANDLERS
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """Nouveau client connecté"""
    logger.info(f"Client connecté: {request.sid}")
    emit('connection_response', {
        'status': 'connected',
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Client déconnecté"""
    logger.info(f"Client déconnecté: {request.sid}")

@socketio.on('request_update')
def handle_update_request():
    """Client demande une mise à jour"""
    emit('status_update', {
        'equity': system_state.current_equity,
        'position': system_state.current_position,
        'pnl': system_state.current_pnl,
        'timestamp': datetime.now().isoformat()
    })

# ============================================================================
# FONCTIONS DE BACKGROUND
# ============================================================================

def run_training(num_episodes: int, batch_size: int, from_checkpoint: Optional[str], agent_id: Optional[int] = None):
    """Exécuter l'entraînement en background"""
    try:
        # Afficher les informations du device
        cuda_available = torch.cuda.is_available()
        device_name = str(torch.cuda.get_device_name(0)) if cuda_available else 'CPU'

        logger.info("="*80)
        logger.info(f"Démarrage entraînement: {num_episodes} épisodes, Agent: {agent_id if agent_id is not None else 'tous'}")
        logger.info(f"Device: {device_name}")
        if cuda_available:
            logger.info(f"GPU Count: {torch.cuda.device_count()}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            logger.warning("⚠️ CUDA non disponible - Entraînement sur CPU (sera plus lent)")
        logger.info("="*80)

        # Notifier le frontend du device utilisé
        socketio.emit('training_device_info', {
            'device': device_name,
            'cuda_available': cuda_available,
            'timestamp': datetime.now().isoformat()
        })

        # Charger les données
        data_pipeline = DataPipeline()
        train_data, val_data, test_data = data_pipeline.get_processed_data()

        # Utiliser FeaturePipeline avec cache pour normaliser les features
        from backend.feature_engineering import FeaturePipeline
        feature_pipeline = FeaturePipeline()
        train_features, val_features, test_features = feature_pipeline.run_full_pipeline(
            train_data, val_data, test_data, force_recalculate=False
        )

        logger.info(f"Features chargées: {len(train_features)} samples d'entraînement")

        # Retirer la colonne timestamp si présente (non numérique)
        if 'timestamp' in train_features.columns:
            train_features = train_features.drop(columns=['timestamp'])

        logger.info(f"Features après nettoyage: {train_features.shape}")

        # Utiliser les données d'entraînement pour l'agent
        eurusd_data = train_data.get('EURUSD')
        if eurusd_data is None:
            raise ValueError("Données EURUSD non trouvées dans le dataset d'entraînement")

        # Créer l'environnement
        from backend.trading_env import TradingEnvConfig
        env_config = TradingEnvConfig(
            initial_capital=system_state.config.get('trading', {}).get('initial_capital', 100000.0)
        )
        env = TradingEnvironment(
            data=eurusd_data,
            features=train_features,
            config=env_config
        )

        # Créer les agents selon agent_id
        agents = []
        if agent_id is not None:
            # Entraîner un seul agent
            sac_config = SACConfig(
                state_dim=30,
                action_dim=1,
                hidden_dims=[256, 256]
            )
            agent = SACAgent(config=sac_config, agent_id=agent_id)
            if from_checkpoint:
                # Extraire seulement le nom du fichier (agent.load() ajoute le préfixe du dossier)
                checkpoint_filename = Path(from_checkpoint).name
                agent.load(checkpoint_filename)
            agents.append(agent)
            agent_indices = [agent_id]
        else:
            # Entraîner tous les agents
            for i in range(system_state.config['model']['ensemble_size']):
                sac_config = SACConfig(
                    state_dim=30,
                    action_dim=1,
                    hidden_dims=[256, 256]
                )
                # Use agent_id >= 3 to avoid regime feature requirements
                agent = SACAgent(config=sac_config, agent_id=i+3)
                if from_checkpoint:
                    checkpoint_path = from_checkpoint.replace('.pt', f'_agent{i}.pt')
                    # Extraire seulement le nom du fichier (agent.load() ajoute le préfixe du dossier)
                    checkpoint_filename = Path(checkpoint_path).name
                    checkpoint_full_path = Path(system_state.config['model']['checkpoint_dir']) / checkpoint_filename
                    if checkpoint_full_path.exists():
                        agent.load(checkpoint_filename)
                agents.append(agent)
            agent_indices = list(range(len(agents)))

        logger.info(f"Agents créés: {len(agents)}")

        # Historique des métriques pour les graphiques (SANS LIMITE - tout l'historique)
        # Charger l'historique existant s'il existe
        metrics_file = Path('logs/training_metrics.json')
        if metrics_file.exists() and not from_checkpoint:
            try:
                with open(metrics_file, 'r') as f:
                    metrics_history = json.load(f)
                logger.info(f"Métriques chargées: {len(metrics_history.get('episodes', []))} épisodes précédents")
            except:
                metrics_history = initialize_metrics_history()
        else:
            metrics_history = initialize_metrics_history()

        def initialize_metrics_history():
            """Initialise toutes les métriques à tracker (style TensorBoard)"""
            return {
                # Episode info
                'episodes': [],
                'timestamps': [],

                # Rewards
                'episode_rewards': [],
                'episode_rewards_mean': [],  # Moving average
                'episode_rewards_std': [],

                # Performance metrics
                'sharpe_ratios': [],
                'sortino_ratios': [],
                'win_rates': [],
                'max_drawdowns': [],
                'total_returns': [],
                'final_equities': [],
                'profit_factors': [],
                'total_trades': [],
                'winning_trades': [],
                'losing_trades': [],

                # Losses
                'critic_losses': [],
                'actor_losses': [],
                'alpha_losses': [],

                # SAC specific
                'alpha_values': [],
                'target_entropies': [],

                # Learning rates
                'actor_lr': [],
                'critic_lr': [],

                # Buffer stats
                'buffer_sizes': [],
                'buffer_winning_ratio': [],
                'buffer_losing_ratio': [],
                'buffer_neutral_ratio': [],

                # Exploration stats
                'action_mean': [],
                'action_std': [],

                # Step info
                'episode_steps': [],
                'total_steps': [],

                # Gradient stats (si disponible)
                'actor_grad_norm': [],
                'critic_grad_norm': [],
            }

        # Réinitialiser si besoin
        if not metrics_file.exists() or not from_checkpoint:
            metrics_history = initialize_metrics_history()

        # Logger pour les transitions (pour CSV)
        transitions_log = []

        # Entraîner chaque agent
        for episode in range(num_episodes):
            # Vider le log de transitions au début de chaque épisode pour ne sauvegarder que l'épisode du checkpoint
            transitions_log.clear()

            # Logger les transitions seulement pour les épisodes qui seront sauvegardés (optimisation performance)
            should_log_transitions = (episode % 100 == 0 and episode > 0)

            if system_state.stop_event.is_set():
                logger.info("Arrêt manuel détecté, sauvegarde des agents...")
                # Sauvegarder les agents avant d'arrêter
                for i, agent in enumerate(agents):
                    current_agent_id = agent_indices[i] if agent_id is None else agent_id
                    filename = f'checkpoint_ep{episode}_agent{current_agent_id}_manual_stop.pt'
                    agent.save(filename)
                    logger.info(f"Checkpoint d'arrêt manuel sauvegardé: {filename}")
                break

            for agent_idx, agent in enumerate(agents):
                state = env.reset()
                episode_reward = 0
                episode_steps = 0
                done = False
                # Utiliser moyenne courante au lieu d'accumuler toutes les pertes (économie mémoire)
                critic_loss_sum = 0.0
                actor_loss_sum = 0.0
                alpha_loss_sum = 0.0
                update_count = 0

                # Tracking pour actions (exploration)
                episode_actions = []

                # Capturer la date de début d'épisode
                episode_start_time = datetime.now()

                while not done:
                    if system_state.stop_event.is_set():
                        break

                    # Sélectionner action
                    action = agent.select_action(state, deterministic=False)

                    # Tracker l'action pour statistiques
                    action_value = float(action[0]) if hasattr(action, '__len__') else float(action)
                    episode_actions.append(action_value)

                    # Step environment
                    next_state, reward, done, info = env.step(action)

                    # Logger la transition pour le CSV seulement si c'est un épisode de checkpoint (optimisation)
                    if should_log_transitions:
                        transition_data = {
                            'episode': episode + 1,
                            'agent_id': agent_indices[agent_idx] if agent_id is None else agent_id,
                            'step': episode_steps,
                            'action': float(action[0]) if hasattr(action, '__len__') else float(action),
                            'reward': float(reward),
                            'done': int(done),
                            'equity': float(info.get('equity', 0)),
                            'position': float(info.get('position', 0)),
                            'cumulative_reward': episode_reward + reward,
                            'episode_start_time': episode_start_time.isoformat()
                        }

                        # Ajouter les observations (state) - toutes les features
                        if hasattr(state, '__len__'):
                            for i, obs_value in enumerate(state):
                                transition_data[f'obs_{i}'] = float(obs_value)

                        transitions_log.append(transition_data)

                    # Stocker dans replay buffer
                    agent.replay_buffer.push(state, action, reward, next_state, done)

                    # Update agent: REDUCED from 4 to 2 updates per step to prevent overfitting
                    if len(agent.replay_buffer) > batch_size:
                        # Faire 2 updates consécutifs (REDUCED from 4)
                        # Reasoning: 4×512 = 2048 samples/step was too aggressive
                        # Now: 2×256 = 512 samples/step (much better ratio)
                        for _ in range(2):
                            losses = agent.update()
                            if losses:
                                critic_loss_sum += losses.get('critic_loss', 0)
                                actor_loss_sum += losses.get('actor_loss', 0)
                                alpha_loss_sum += losses.get('alpha_loss', 0)
                                update_count += 1

                    episode_reward += reward
                    episode_steps += 1
                    agent.total_steps += 1  # CRITICAL: Increment total steps for LR scheduling
                    state = next_state

                    # Émettre la progression de l'épisode tous les 20 steps (optimisation performance)
                    if episode_steps % 20 == 0:
                        socketio.emit('episode_step_progress', {
                            'current_step': int(episode_steps),
                            'episode_length': int(env.episode_length),
                            'episode': int(episode + 1)
                        })

                # Capturer la date de fin d'épisode et l'ajouter à la dernière transition
                if should_log_transitions:
                    episode_end_time = datetime.now()
                    if len(transitions_log) > 0:
                        transitions_log[-1]['episode_end_time'] = episode_end_time.isoformat()

                # Calculer métriques de l'épisode
                env_metrics = env.get_episode_metrics()

                # Calculer statistiques des actions
                action_mean = float(np.mean(episode_actions)) if episode_actions else 0.0
                action_std = float(np.std(episode_actions)) if episode_actions else 0.0

                # Calculer moving average des rewards (100 derniers épisodes)
                recent_rewards = metrics_history['episode_rewards'][-99:] + [episode_reward]
                reward_mean = float(np.mean(recent_rewards))
                reward_std = float(np.std(recent_rewards))

                # Récupérer les stats du buffer
                buffer_size = len(agent.replay_buffer)
                buffer_winning = len(agent.replay_buffer.winning_indices) / buffer_size if buffer_size > 0 else 0
                buffer_losing = len(agent.replay_buffer.losing_indices) / buffer_size if buffer_size > 0 else 0
                buffer_neutral = len(agent.replay_buffer.neutral_indices) / buffer_size if buffer_size > 0 else 0

                # Récupérer les learning rates actuels
                actor_lr = float(agent.actor_optimizer.param_groups[0]['lr'])
                critic_lr = float(agent.critic_optimizer.param_groups[0]['lr'])

                # Mettre à jour l'état du training
                current_agent_id = agent_indices[agent_idx] if agent_id is None else agent_id
                system_state.training_state['current_episode'] = episode + 1
                system_state.training_state['metrics'] = {
                    'agent_id': current_agent_id,
                    'episode_reward': float(episode_reward),
                    'episode_steps': episode_steps,
                    'avg_critic_loss': float(critic_loss_sum / update_count) if update_count > 0 else 0,
                    'avg_actor_loss': float(actor_loss_sum / update_count) if update_count > 0 else 0,
                    'avg_alpha_loss': float(alpha_loss_sum / update_count) if update_count > 0 else 0,
                    'sharpe_ratio': float(env_metrics.get('sharpe_ratio', 0)),
                    'sortino_ratio': float(env_metrics.get('sortino_ratio', 0)),
                    'max_drawdown': float(env_metrics.get('max_drawdown', 0)),
                    'win_rate': float(env_metrics.get('win_rate', 0)),
                    'profit_factor': float(env_metrics.get('profit_factor', 0)),
                    'total_return': float(env_metrics.get('total_return', 0))
                }

                # Ajouter TOUTES les métriques à l'historique (SANS LIMITE - stockage complet)
                metrics_history['episodes'].append(int(episode + 1))
                metrics_history['timestamps'].append(datetime.now().isoformat())

                # Rewards
                metrics_history['episode_rewards'].append(float(episode_reward))
                metrics_history['episode_rewards_mean'].append(reward_mean)
                metrics_history['episode_rewards_std'].append(reward_std)

                # Performance metrics
                metrics_history['sharpe_ratios'].append(float(env_metrics.get('sharpe_ratio', 0)))
                metrics_history['sortino_ratios'].append(float(env_metrics.get('sortino_ratio', 0)))
                metrics_history['win_rates'].append(float(env_metrics.get('win_rate', 0)))
                metrics_history['max_drawdowns'].append(float(env_metrics.get('max_drawdown', 0)))
                metrics_history['total_returns'].append(float(env_metrics.get('total_return', 0)))
                metrics_history['final_equities'].append(float(env_metrics.get('final_equity', 0)))
                metrics_history['profit_factors'].append(float(env_metrics.get('profit_factor', 0)))
                metrics_history['total_trades'].append(int(env_metrics.get('total_trades', 0)))
                metrics_history['winning_trades'].append(int(env_metrics.get('winning_trades', 0)))
                metrics_history['losing_trades'].append(int(env_metrics.get('losing_trades', 0)))

                # Losses
                metrics_history['critic_losses'].append(float(critic_loss_sum / update_count) if update_count > 0 else 0)
                metrics_history['actor_losses'].append(float(actor_loss_sum / update_count) if update_count > 0 else 0)
                metrics_history['alpha_losses'].append(float(alpha_loss_sum / update_count) if update_count > 0 else 0)

                # SAC specific
                metrics_history['alpha_values'].append(float(agent.alpha.item()))
                metrics_history['target_entropies'].append(float(agent.target_entropy) if hasattr(agent, 'target_entropy') else 0)

                # Learning rates
                metrics_history['actor_lr'].append(actor_lr)
                metrics_history['critic_lr'].append(critic_lr)

                # Buffer stats
                metrics_history['buffer_sizes'].append(buffer_size)
                metrics_history['buffer_winning_ratio'].append(float(buffer_winning))
                metrics_history['buffer_losing_ratio'].append(float(buffer_losing))
                metrics_history['buffer_neutral_ratio'].append(float(buffer_neutral))

                # Exploration stats
                metrics_history['action_mean'].append(action_mean)
                metrics_history['action_std'].append(action_std)

                # Step info
                metrics_history['episode_steps'].append(episode_steps)
                metrics_history['total_steps'].append(int(agent.total_steps))

                # PAS DE LIMITATION - On garde tout l'historique depuis l'épisode 0

                # Sauvegarder l'état ET les métriques tous les 10 épisodes (optimisation performance)
                if episode % 10 == 0:
                    system_state.save_training_state()

                    # Sauvegarder l'historique complet des métriques dans un fichier JSON
                    try:
                        with open(metrics_file, 'w') as f:
                            json.dump(metrics_history, f, indent=2)
                        logger.info(f"Métriques sauvegardées: {len(metrics_history['episodes'])} épisodes")
                    except Exception as e:
                        logger.error(f"Erreur sauvegarde métriques: {e}")

                # Envoyer TOUT l'historique au frontend (pas de limitation)
                # Le frontend peut gérer l'affichage (zoom, pan, etc.)
                history_to_send = metrics_history

                # Émettre progression à chaque épisode avec TOUTES les métriques
                socketio.emit('training_progress', {
                    # Episode info
                    'episode': int(episode + 1),
                    'total_episodes': int(num_episodes),
                    'agent': int(current_agent_id),
                    'steps': int(episode_steps),
                    'total_steps': int(agent.total_steps),
                    'episode_length': int(env.episode_length),

                    # Current episode metrics
                    'reward': float(episode_reward),
                    'reward_mean': reward_mean,
                    'reward_std': reward_std,

                    # Losses
                    'critic_loss': float(critic_loss_sum / update_count) if update_count > 0 else 0.0,
                    'actor_loss': float(actor_loss_sum / update_count) if update_count > 0 else 0.0,
                    'alpha_loss': float(alpha_loss_sum / update_count) if update_count > 0 else 0.0,

                    # SAC specific
                    'alpha': float(agent.alpha.item()),
                    'target_entropy': float(agent.target_entropy) if hasattr(agent, 'target_entropy') else 0,

                    # Learning rates
                    'actor_lr': actor_lr,
                    'critic_lr': critic_lr,

                    # Buffer stats
                    'buffer_size': buffer_size,
                    'buffer_winning_ratio': float(buffer_winning),
                    'buffer_losing_ratio': float(buffer_losing),
                    'buffer_neutral_ratio': float(buffer_neutral),

                    # Exploration
                    'action_mean': action_mean,
                    'action_std': action_std,

                    # Performance metrics
                    'sharpe_ratio': float(env_metrics.get('sharpe_ratio', 0)),
                    'sortino_ratio': float(env_metrics.get('sortino_ratio', 0)),
                    'max_drawdown': float(env_metrics.get('max_drawdown', 0)),
                    'win_rate': float(env_metrics.get('win_rate', 0)),
                    'profit_factor': float(env_metrics.get('profit_factor', 0)),
                    'total_return': float(env_metrics.get('total_return', 0)),
                    'total_trades': int(env_metrics.get('total_trades', 0)),
                    'winning_trades': int(env_metrics.get('winning_trades', 0)),
                    'losing_trades': int(env_metrics.get('losing_trades', 0)),
                    'final_equity': float(env_metrics.get('final_equity', 0)),

                    # Timestamp
                    'timestamp': datetime.now().isoformat(),

                    # TOUT l'historique des métriques (depuis l'épisode 0)
                    'metrics_history': history_to_send
                })

            # Sauvegarder checkpoint tous les 100 épisodes
            if episode % 100 == 0 and episode > 0:
                for i, agent in enumerate(agents):
                    current_agent_id = agent_indices[i] if agent_id is None else agent_id
                    filename = f'checkpoint_ep{episode}_agent{current_agent_id}.pt'
                    agent.save(filename)
                    logger.info(f"Checkpoint sauvegardé: {filename}")

                # Sauvegarder les transitions dans un CSV (seulement l'épisode du checkpoint)
                if len(transitions_log) > 0:
                    import pandas as pd
                    csv_dir = Path('logs/training_csvs')
                    csv_dir.mkdir(parents=True, exist_ok=True)

                    df = pd.DataFrame(transitions_log)
                    # Nom du fichier indique l'épisode du checkpoint
                    csv_filename = csv_dir / f'training_ep{episode}_agent{current_agent_id}.csv'
                    df.to_csv(csv_filename, index=False)
                    logger.info(f"Training CSV sauvegardé: {csv_filename} ({len(transitions_log)} transitions)")
        
        logger.info("Entraînement terminé")
        system_state.is_training = False
        system_state.stop_event.clear()

        # Mettre à jour l'état final
        system_state.training_state['is_training'] = False
        system_state.save_training_state()

        # Sauvegarde FINALE de toutes les métriques
        try:
            with open(metrics_file, 'w') as f:
                json.dump(metrics_history, f, indent=2)
            logger.info(f"✅ Métriques finales sauvegardées: {len(metrics_history['episodes'])} épisodes au total")
            logger.info(f"   Fichier: {metrics_file}")
        except Exception as e:
            logger.error(f"Erreur sauvegarde métriques finales: {e}")

        socketio.emit('training_complete', {
            'message': 'Entraînement terminé avec succès',
            'agent_id': agent_id if agent_id is not None else 'all',
            'total_episodes': len(metrics_history['episodes']),
            'metrics_file': str(metrics_file),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Erreur pendant l'entraînement: {e}")
        logger.error(traceback.format_exc())
        system_state.is_training = False
        system_state.stop_event.clear()

        # Mettre à jour l'état en cas d'erreur
        system_state.training_state['is_training'] = False
        system_state.save_training_state()

        socketio.emit('training_error', {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        })

def run_meta_controller_training(num_episodes: int, batch_size: int):
    """Exécuter l'entraînement du meta-controller en background"""
    try:
        # Afficher les informations du device
        cuda_available = torch.cuda.is_available()
        device_name = str(torch.cuda.get_device_name(0)) if cuda_available else 'CPU'

        logger.info("="*80)
        logger.info(f"Démarrage entraînement meta-controller: {num_episodes} épisodes")
        logger.info(f"Device: {device_name}")
        if cuda_available:
            logger.info(f"GPU Count: {torch.cuda.device_count()}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            logger.warning("⚠️ CUDA non disponible - Entraînement sur CPU (sera plus lent)")
        logger.info("="*80)

        # Notifier le frontend du device utilisé
        socketio.emit('training_device_info', {
            'device': device_name,
            'cuda_available': cuda_available,
            'timestamp': datetime.now().isoformat()
        })

        # Charger les données
        data_pipeline = DataPipeline()
        train_data, val_data, test_data = data_pipeline.get_processed_data()

        # Utiliser FeaturePipeline avec cache
        from backend.feature_engineering import FeaturePipeline
        feature_pipeline = FeaturePipeline()
        train_features, val_features, test_features = feature_pipeline.run_full_pipeline(
            train_data, val_data, test_data, force_recalculate=False
        )

        # Retirer la colonne timestamp si présente (non numérique)
        if 'timestamp' in train_features.columns:
            train_features = train_features.drop(columns=['timestamp'])

        # Charger les 3 agents SAC pré-entraînés
        agents = []
        checkpoint_dir = Path(system_state.config['model']['checkpoint_dir'])

        for i in range(3):
            sac_config = SACConfig(
                state_dim=30,
                action_dim=1,
                hidden_dims=[256, 256]
            )
            agent = SACAgent(config=sac_config, agent_id=i+3)

            # Trouver le dernier checkpoint pour cet agent
            checkpoints = list(checkpoint_dir.glob(f'checkpoint_*_agent{i+3}.pt'))
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
                agent.load(latest_checkpoint.name)  # Passer seulement le nom du fichier
                logger.info(f"Agent {i+3} chargé depuis: {latest_checkpoint.name}")
            else:
                logger.warning(f"Aucun checkpoint trouvé pour l'agent {i+3}")

            agents.append(agent)

        # Créer le meta-controller
        meta_controller = EnsembleMetaController(
            state_dim=30,
            action_dim=1,
            num_agents=3
        )

        # Créer l'environnement
        eurusd_data = train_data.get('EURUSD')
        from backend.trading_env import TradingEnvConfig
        env_config = TradingEnvConfig(
            initial_capital=system_state.config.get('trading', {}).get('initial_capital', 100000.0)
        )
        env = TradingEnvironment(
            data=eurusd_data,
            features=train_features,
            config=env_config
        )

        # Logger pour les transitions (pour CSV)
        transitions_log = []

        # Entraîner le meta-controller
        for episode in range(num_episodes):
            # Vider le log de transitions au début de chaque épisode pour ne sauvegarder que l'épisode du checkpoint
            transitions_log.clear()

            # Logger les transitions seulement pour les épisodes qui seront sauvegardés (optimisation performance)
            should_log_transitions = (episode % 50 == 0 and episode > 0)

            if system_state.stop_event.is_set():
                break

            state = env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False

            # Capturer la date de début d'épisode
            episode_start_time = datetime.now()

            while not done:
                if system_state.stop_event.is_set():
                    break

                # Obtenir les actions de tous les agents
                agent_actions = []
                for agent in agents:
                    action = agent.select_action(state, deterministic=True)
                    agent_actions.append(action)

                # Le meta-controller apprend à pondérer les actions
                final_action = meta_controller.aggregate_actions(state, agent_actions)

                # Step environment
                next_state, reward, done, info = env.step(final_action)

                # Logger la transition pour le CSV seulement si c'est un épisode de checkpoint (optimisation)
                if should_log_transitions:
                    transition_data = {
                        'episode': episode + 1,
                        'agent_id': 'meta_controller',
                        'step': episode_steps,
                        'final_action': float(final_action[0]) if hasattr(final_action, '__len__') else float(final_action),
                        'reward': float(reward),
                        'done': int(done),
                        'equity': float(info.get('equity', 0)),
                        'position': float(info.get('position', 0)),
                        'cumulative_reward': episode_reward + reward,
                        'episode_start_time': episode_start_time.isoformat()
                    }

                    # Ajouter les actions des agents individuels
                    for i, agent_action in enumerate(agent_actions):
                        action_val = float(agent_action[0]) if hasattr(agent_action, '__len__') else float(agent_action)
                        transition_data[f'agent_{i+3}_action'] = action_val

                    # Ajouter les observations (state)
                    if hasattr(state, '__len__'):
                        for i, obs_value in enumerate(state):
                            transition_data[f'obs_{i}'] = float(obs_value)

                    transitions_log.append(transition_data)

                # Entraîner le meta-controller
                meta_controller.train_step(state, agent_actions, reward, next_state, done)

                episode_reward += reward
                episode_steps += 1
                state = next_state

                # Émettre la progression de l'épisode tous les 20 steps (optimisation performance)
                if episode_steps % 20 == 0:
                    socketio.emit('episode_step_progress', {
                        'current_step': int(episode_steps),
                        'episode_length': int(env.episode_length),
                        'episode': int(episode + 1)
                    })

            # Capturer la date de fin d'épisode et l'ajouter à la dernière transition
            if should_log_transitions:
                episode_end_time = datetime.now()
                if len(transitions_log) > 0:
                    transitions_log[-1]['episode_end_time'] = episode_end_time.isoformat()

            # Calculer métriques de l'épisode
            env_metrics = env.get_episode_metrics()

            # Mettre à jour l'état du training
            system_state.training_state['current_episode'] = episode + 1
            system_state.training_state['metrics'] = {
                'agent_id': 'meta_controller',
                'episode_reward': float(episode_reward),
                'episode_steps': episode_steps,
                'sharpe_ratio': float(env_metrics.get('sharpe_ratio', 0)),
                'sortino_ratio': float(env_metrics.get('sortino_ratio', 0)),
                'max_drawdown': float(env_metrics.get('max_drawdown', 0)),
                'win_rate': float(env_metrics.get('win_rate', 0)),
                'profit_factor': float(env_metrics.get('profit_factor', 0)),
                'total_return': float(env_metrics.get('total_return', 0))
            }

            # Émettre progression à chaque épisode
            system_state.save_training_state()
            socketio.emit('training_progress', {
                'episode': int(episode + 1),
                'total_episodes': int(num_episodes),
                'agent': 'meta_controller',
                'reward': float(episode_reward),
                'steps': int(episode_steps),
                'episode_length': int(env.episode_length),  # Pour la barre de progression d'épisode
                'sharpe_ratio': float(env_metrics.get('sharpe_ratio', 0)),
                'sortino_ratio': float(env_metrics.get('sortino_ratio', 0)),
                'max_drawdown': float(env_metrics.get('max_drawdown', 0)),
                'win_rate': float(env_metrics.get('win_rate', 0)),
                'profit_factor': float(env_metrics.get('profit_factor', 0)),
                'total_return': float(env_metrics.get('total_return', 0)),
                'timestamp': datetime.now().isoformat()
            })

            # Sauvegarder checkpoint tous les 50 épisodes
            if episode % 50 == 0 and episode > 0:
                filename = f'meta_controller_ep{episode}.pt'
                meta_controller.save(filename)
                logger.info(f"Meta-controller checkpoint sauvegardé: {filename}")

                # Sauvegarder les transitions dans un CSV
                if len(transitions_log) > 0:
                    import pandas as pd
                    csv_dir = Path('logs/training_csvs')
                    csv_dir.mkdir(parents=True, exist_ok=True)

                    df = pd.DataFrame(transitions_log)
                    csv_filename = csv_dir / f'training_meta_controller_ep{episode}.csv'
                    df.to_csv(csv_filename, index=False)
                    logger.info(f"Training CSV sauvegardé: {csv_filename} ({len(transitions_log)} transitions)")

        logger.info("Entraînement meta-controller terminé")
        system_state.is_training = False
        system_state.stop_event.clear()

        # Mettre à jour l'état final
        system_state.training_state['is_training'] = False
        system_state.save_training_state()

        socketio.emit('training_complete', {
            'message': 'Entraînement meta-controller terminé avec succès',
            'agent_id': 'meta_controller',
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Erreur pendant l'entraînement meta-controller: {e}")
        logger.error(traceback.format_exc())
        system_state.is_training = False
        system_state.stop_event.clear()

        # Mettre à jour l'état en cas d'erreur
        system_state.training_state['is_training'] = False
        system_state.save_training_state()

        socketio.emit('training_error', {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        })

def run_trading():
    """Exécuter le trading live en background"""
    try:
        logger.info("Démarrage trading live")
        
        # Boucle de trading
        while not system_state.stop_event.is_set():
            try:
                # Vérifier que le marché est ouvert
                if not system_state.mt5_connector.is_market_open():
                    logger.info("Marché fermé, pause...")
                    time.sleep(60)
                    continue
                
                # Obtenir les dernières données
                current_data = system_state.mt5_connector.get_current_data()
                
                # Calculer les features
                features = system_state.feature_engineer.calculate_features(current_data)
                current_state = features[-1]  # Dernière observation
                
                # Obtenir l'action de l'ensemble
                action = system_state.ensemble_controller.predict(current_state)
                
                # Vérifier avec le risk manager
                can_trade, reason = system_state.risk_manager.can_trade(
                    current_equity=system_state.current_equity,
                    current_position=system_state.current_position
                )
                
                if not can_trade:
                    logger.warning(f"Trade refusé: {reason}")
                    time.sleep(60)
                    continue
                
                # Exécuter l'action si significative
                if abs(action) > 0.1:  # Seuil pour éviter micro-trades
                    # Calculer position size
                    position_size = system_state.risk_manager.calculate_position_size(
                        action=action,
                        equity=system_state.current_equity,
                        atr=current_state[6]  # ATR feature
                    )
                    
                    # Passer l'ordre
                    if action > 0:  # Long
                        result = system_state.mt5_connector.open_position(
                            symbol=system_state.config['mt5']['symbol'],
                            order_type='buy',
                            volume=position_size
                        )
                    else:  # Short
                        result = system_state.mt5_connector.open_position(
                            symbol=system_state.config['mt5']['symbol'],
                            order_type='sell',
                            volume=abs(position_size)
                        )
                    
                    if result['success']:
                        logger.info(f"Position ouverte: {result}")
                        
                        # Enregistrer le trade
                        trade = {
                            'timestamp': datetime.now().isoformat(),
                            'action': 'long' if action > 0 else 'short',
                            'size': position_size,
                            'price': result.get('price'),
                            'sl': result.get('sl'),
                            'tp': result.get('tp')
                        }
                        system_state.trade_history.append(trade)
                        system_state.daily_trades.append(trade)
                        
                        # Émettre via SocketIO
                        socketio.emit('new_trade', trade)
                
                # Mettre à jour les métriques
                account_info = system_state.mt5_connector.get_account_info()
                system_state.current_equity = account_info['equity']
                system_state.current_position = system_state.mt5_connector.get_current_position()
                system_state.current_pnl = account_info['profit']
                
                system_state.equity_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'equity': system_state.current_equity
                })
                
                # Émettre mise à jour
                socketio.emit('status_update', {
                    'equity': system_state.current_equity,
                    'position': system_state.current_position,
                    'pnl': system_state.current_pnl,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Attendre prochaine barre (5 minutes)
                time.sleep(300)
            
            except Exception as e:
                logger.error(f"Erreur dans la boucle de trading: {e}")
                logger.error(traceback.format_exc())
                time.sleep(60)
        
        logger.info("Trading arrêté")
        system_state.is_trading = False
        system_state.stop_event.clear()
    
    except Exception as e:
        logger.error(f"Erreur fatale trading: {e}")
        logger.error(traceback.format_exc())
        system_state.is_trading = False
        system_state.stop_event.clear()

def run_validation_process(validator: ValidationFramework, model_path: str):
    """Exécuter le processus de validation"""
    try:
        logger.info(f"Démarrage validation pour: {model_path}")

        # Charger le modèle
        sac_config = SACConfig(
            state_dim=30,
            action_dim=1,
            hidden_dims=[256, 256]
        )
        agent = SACAgent(config=sac_config, agent_id=1)
        agent.load(model_path)

        # Exécuter validation walk-forward
        results = validator.walk_forward_validation(
            agent=agent,
            n_folds=5,
            train_size=0.7
        )
        
        # Calculer métriques statistiques
        stats = validator.compute_statistical_tests(results)
        
        # Sauvegarder résultats
        report_path = Path('reports') / f'validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump({
                'results': results,
                'statistics': stats,
                'model_path': model_path,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        # Émettre résultats
        socketio.emit('validation_complete', {
            'results': results,
            'statistics': stats,
            'report_path': str(report_path),
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"Validation terminée: {report_path}")
    
    except Exception as e:
        logger.error(f"Erreur validation: {e}")
        logger.error(traceback.format_exc())
        
        socketio.emit('validation_error', {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        })

def monitoring_loop():
    """Boucle de monitoring en arrière-plan"""
    while True:
        try:
            if system_state.is_trading:
                # Vérifier état du système
                account_info = system_state.mt5_connector.get_account_info()
                
                # Calculer drawdown
                peak_equity = max([e['equity'] for e in system_state.equity_history] + [system_state.current_equity])
                current_dd = (peak_equity - system_state.current_equity) / peak_equity
                
                # Vérifier limites de risque
                if current_dd > system_state.config['trading']['max_drawdown']:
                    logger.warning(f"Drawdown maximum atteint: {current_dd:.2%}")
                    socketio.emit('risk_alert', {
                        'type': 'max_drawdown',
                        'value': current_dd,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Arrêter le trading
                    stop_trading()
            
            time.sleep(60)  # Check toutes les minutes
        
        except Exception as e:
            logger.error(f"Erreur monitoring: {e}")
            time.sleep(60)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Créer les dossiers nécessaires
    for folder in ['logs', 'models/checkpoints', 'models/production', 'config', 'reports', 'static', 'templates']:
        Path(folder).mkdir(parents=True, exist_ok=True)
    
    # Démarrer le thread de monitoring
    monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
    monitoring_thread.start()
    
    # Lancer l'application
    logger.info("Démarrage de l'application web...")
    logger.info("Interface accessible sur http://localhost:5000")
    
    socketio.run(
        app,
        host='0.0.0.0',
        port=5000,
        debug=False,
        allow_unsafe_werkzeug=True
    )
