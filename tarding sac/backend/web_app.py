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
                    'path': str(path),
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
                    'path': str(path),
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
        model_path = data.get('model_path')
        
        if not model_path or not Path(model_path).exists():
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
        checkpoint = torch.load(model_path)
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
        batch_size = data.get('batch_size', 256)
        from_checkpoint = data.get('from_checkpoint', None)
        
        # Lancer l'entraînement dans un thread séparé
        training_thread = threading.Thread(
            target=run_training,
            args=(num_episodes, batch_size, from_checkpoint)
        )
        training_thread.daemon = True
        training_thread.start()
        
        system_state.training_thread = training_thread
        system_state.is_training = True
        
        return jsonify({
            'success': True,
            'message': 'Entraînement démarré'
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
    
    return jsonify({
        'success': True,
        'message': 'Entraînement arrêté'
    })

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

def run_training(num_episodes: int, batch_size: int, from_checkpoint: Optional[str]):
    """Exécuter l'entraînement en background"""
    try:
        logger.info(f"Démarrage entraînement: {num_episodes} épisodes")
        
        # Charger les données
        data_pipeline = DataPipeline()
        train_data, val_data, test_data = data_pipeline.get_processed_data()
        
        # Utiliser les données d'entraînement pour l'agent
        # Prendre EURUSD comme paire principale
        eurusd_data = train_data.get('EURUSD')
        if eurusd_data is None:
            raise ValueError("Données EURUSD non trouvées dans le dataset d'entraînement")
        
        # Feature engineering
        feature_engineer = FeatureEngineer()
        # Calculer les features pour toutes les paires nécessaires
        all_data = {**train_data, **val_data}
        raw_features = feature_engineer.calculate_all_features(all_data)
        # Normaliser les features
        normalized_features = feature_engineer.normalize_features(raw_features)
        
        # Créer l'environnement
        from backend.trading_env import TradingEnvConfig
        env_config = TradingEnvConfig(
            initial_capital=system_state.config.get('trading', {}).get('initial_capital', 100000.0)
        )
        env = TradingEnvironment(
            data=eurusd_data,
            features=normalized_features,
            config=env_config
        )
        
        # Créer les agents
        agents = []
        for i in range(system_state.config['model']['ensemble_size']):
            # Create SAC config with correct parameters
            sac_config = SACConfig(
                state_dim=30,
                action_dim=1,
                hidden_dims=[256, 256]
            )
            # Use agent_id >= 3 to avoid regime feature requirements for agents 1 & 2
            # (agents 1 & 2 expect state_dim=32 with HMM regime features)
            agent = SACAgent(config=sac_config, agent_id=i+3)

            # Charger depuis checkpoint si spécifié
            if from_checkpoint:
                agent.load(from_checkpoint)

            agents.append(agent)
        
        # Entraîner chaque agent
        for episode in range(num_episodes):
            if system_state.stop_event.is_set():
                break
            
            for agent_idx, agent in enumerate(agents):
                state = env.reset()
                episode_reward = 0
                done = False
                
                while not done:
                    if system_state.stop_event.is_set():
                        break
                    
                    # Sélectionner action
                    action = agent.select_action(state, deterministic=False)
                    
                    # Step environment
                    next_state, reward, done, info = env.step(action)

                    # Stocker dans replay buffer
                    agent.replay_buffer.push(state, action, reward, next_state, done)

                    # Update agent
                    if len(agent.replay_buffer) > batch_size:
                        agent.update(batch_size)
                    
                    episode_reward += reward
                    state = next_state
                
                # Émettre progression
                if episode % 10 == 0:
                    socketio.emit('training_progress', {
                        'episode': episode,
                        'agent': agent_idx,
                        'reward': episode_reward,
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Sauvegarder checkpoint
            if episode % 100 == 0:
                checkpoint_path = Path(system_state.config['model']['checkpoint_dir']) / f'checkpoint_ep{episode}.pt'
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

                for i, agent in enumerate(agents):
                    agent.save(f'checkpoint_ep{episode}_agent{i}.pt')
        
        logger.info("Entraînement terminé")
        system_state.is_training = False
        system_state.stop_event.clear()
        
        socketio.emit('training_complete', {
            'message': 'Entraînement terminé avec succès',
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Erreur pendant l'entraînement: {e}")
        logger.error(traceback.format_exc())
        system_state.is_training = False
        system_state.stop_event.clear()
        
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
