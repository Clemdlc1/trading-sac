"""
Script d'entraînement SAC pour Kaggle
======================================

Ce script permet d'entraîner les agents SAC sur Kaggle avec des données h5 pré-processées.
Il est 100% compatible avec le code local et produit des modèles directement utilisables.

Usage sur Kaggle:
    1. Upload le fichier processed_data.h5 dans /kaggle/input/
    2. Copier tout le dossier backend/ dans le notebook
    3. Exécuter ce script

Auteur: Trading SAC System
Date: 2025-11-23
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch

# Ajouter le chemin backend pour les imports
sys.path.insert(0, str(Path(__file__).parent))

from backend.data_pipeline import DataPipeline
from backend.feature_engineering import FeaturePipeline
from backend.trading_env import TradingEnvironment, TradingEnvConfig
from backend.sac_agent import SACAgent, SACConfig


class KaggleTrainer:
    """Trainer optimisé pour environnement Kaggle"""

    def __init__(
        self,
        h5_path: str = "/kaggle/input/trading-data/processed_data.h5",
        output_dir: str = "/kaggle/working",
        num_episodes: int = 100,
        eval_frequency: int = 10,
        checkpoint_frequency: int = 5,
        agent_id: int = 1,
        device: str = "auto"
    ):
        """
        Args:
            h5_path: Chemin vers le fichier h5 avec les données
            output_dir: Répertoire de sortie pour les modèles
            num_episodes: Nombre d'épisodes d'entraînement
            eval_frequency: Fréquence d'évaluation (en épisodes)
            checkpoint_frequency: Fréquence de sauvegarde (en épisodes)
            agent_id: ID de l'agent (1, 2, ou 3)
            device: Device PyTorch ('cuda', 'cpu', ou 'auto')
        """
        self.h5_path = h5_path
        self.output_dir = Path(output_dir)
        self.num_episodes = num_episodes
        self.eval_frequency = eval_frequency
        self.checkpoint_frequency = checkpoint_frequency
        self.agent_id = agent_id

        # Configuration du device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"🚀 Initialisation KaggleTrainer")
        print(f"   Device: {self.device}")
        print(f"   Agent ID: {self.agent_id}")
        print(f"   Episodes: {self.num_episodes}")
        print(f"   Output: {self.output_dir}")

        # Créer les répertoires de sortie
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.logs_dir = self.output_dir / "logs"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Métriques d'entraînement
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'eval_rewards': [],
            'actor_losses': [],
            'critic_losses': [],
            'alpha_values': [],
        }

    def load_data(self):
        """Charge les données depuis le fichier h5"""
        print("\n📊 Chargement des données...")

        # Vérifier que le fichier existe
        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(
                f"Fichier h5 introuvable: {self.h5_path}\n"
                f"Assurez-vous d'avoir uploadé processed_data.h5 dans Kaggle Input."
            )

        # Charger via DataPipeline (qui gère le cache h5)
        data_pipeline = DataPipeline(processed_data_path=self.h5_path)
        train_data, val_data, test_data = data_pipeline.get_processed_data()

        print(f"✅ Données chargées:")
        print(f"   Train: {len(train_data['EURUSD'])} candles")
        print(f"   Val:   {len(val_data['EURUSD'])} candles")
        print(f"   Test:  {len(test_data['EURUSD'])} candles")

        return train_data, val_data, test_data

    def compute_features(self, train_data, val_data, test_data):
        """Calcule les features avec FeaturePipeline"""
        print("\n🔧 Calcul des features...")

        feature_pipeline = FeaturePipeline()
        train_features, val_features, test_features = feature_pipeline.run_full_pipeline(
            train_data, val_data, test_data
        )

        print(f"✅ Features calculées: {train_features.shape[1]} dimensions")
        print(f"   Train shape: {train_features.shape}")
        print(f"   Val shape:   {val_features.shape}")
        print(f"   Test shape:  {test_features.shape}")

        return train_features, val_features, test_features

    def create_environments(self, train_data, val_data, train_features, val_features):
        """Crée les environnements d'entraînement et de validation"""
        print("\n🌍 Création des environnements...")

        # Configuration de l'environnement (même config que le système local)
        env_config = TradingEnvConfig(
            initial_capital=500000.0,
            risk_per_trade=0.0005,
            max_leverage=2.0,
            sl_atr_multiplier=3.0,
            tp_atr_multiplier=6.0,
            episode_lengths=[3000],  # 3000 steps = ~5 jours
            no_trading_warmup_steps=5000,
            use_simple_reward=True,
        )

        # Environnement d'entraînement
        train_env = TradingEnvironment(
            data=train_data['EURUSD'],
            features=train_features,
            config=env_config,
            eval_mode=False  # Mode aléatoire pour exploration
        )

        # Environnement de validation
        val_env = TradingEnvironment(
            data=val_data['EURUSD'],
            features=val_features,
            config=env_config,
            eval_mode=True  # Mode séquentiel pour évaluation
        )

        print(f"✅ Environnements créés:")
        print(f"   Action space: {train_env.action_space}")
        print(f"   Observation space: {train_env.observation_space}")

        return train_env, val_env, env_config

    def create_agent(self):
        """Crée l'agent SAC avec la configuration appropriée"""
        print("\n🤖 Création de l'agent SAC...")

        # Configuration SAC (identique au système local)
        sac_config = SACConfig(
            state_dim=30,  # 30 features
            action_dim=1,  # 1 action (position sizing)
            hidden_dims=[256, 256],
            actor_lr=3e-4,
            critic_lr=3e-4,
            alpha_lr=3e-5,
            gamma=0.95,
            tau=0.005,
            warmup_steps=5000,
            buffer_capacity=100000,
            batch_size=1024,
            auto_entropy_tuning=True,
            target_entropy=-1.0,
        )

        # Créer l'agent
        agent = SACAgent(
            config=sac_config,
            agent_id=self.agent_id,
            device=self.device
        )

        print(f"✅ Agent créé:")
        print(f"   State dim: {sac_config.state_dim}")
        print(f"   Hidden dims: {sac_config.hidden_dims}")
        print(f"   Warmup steps: {sac_config.warmup_steps}")
        print(f"   Buffer size: {sac_config.buffer_capacity}")

        return agent, sac_config

    def train_episode(self, agent: SACAgent, env: TradingEnvironment, episode: int) -> Dict[str, float]:
        """Entraîne un épisode complet"""
        state = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        episode_losses = {'actor': [], 'critic': [], 'alpha': []}

        start_time = time.time()

        while not done:
            # Sélection de l'action
            if agent.total_steps < agent.config.warmup_steps:
                # Phase de warmup: actions aléatoires
                action = env.action_space.sample()
            else:
                # Phase d'apprentissage: policy apprise
                action = agent.select_action(state, evaluate=False)

            # Exécuter l'action dans l'environnement
            next_state, reward, done, info = env.step(action)

            # Stocker la transition dans le replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # Mettre à jour l'agent (après warmup)
            if agent.total_steps >= agent.config.warmup_steps:
                losses = agent.update()
                if losses:
                    episode_losses['actor'].append(losses.get('actor_loss', 0))
                    episode_losses['critic'].append(losses.get('critic_loss', 0))
                    episode_losses['alpha'].append(losses.get('alpha', 0))

            # Accumuler les métriques
            episode_reward += reward
            episode_length += 1
            state = next_state
            agent.total_steps += 1

        # Incrémenter le compteur d'épisodes
        agent.episode_count += 1

        # Calculer les moyennes des pertes
        avg_actor_loss = np.mean(episode_losses['actor']) if episode_losses['actor'] else 0
        avg_critic_loss = np.mean(episode_losses['critic']) if episode_losses['critic'] else 0
        avg_alpha = np.mean(episode_losses['alpha']) if episode_losses['alpha'] else 0

        elapsed_time = time.time() - start_time

        return {
            'reward': episode_reward,
            'length': episode_length,
            'actor_loss': avg_actor_loss,
            'critic_loss': avg_critic_loss,
            'alpha': avg_alpha,
            'time': elapsed_time,
            'final_equity': info.get('equity', 0),
            'total_return': info.get('total_return', 0),
            'sharpe': info.get('sharpe_ratio', 0),
            'max_drawdown': info.get('max_drawdown', 0),
        }

    def evaluate_agent(self, agent: SACAgent, env: TradingEnvironment, num_episodes: int = 3) -> Dict[str, float]:
        """Évalue l'agent sur plusieurs épisodes"""
        eval_rewards = []
        eval_returns = []
        eval_sharpes = []
        eval_drawdowns = []

        for _ in range(num_episodes):
            state = env.reset()
            done = False
            episode_reward = 0.0

            while not done:
                # Mode évaluation: pas d'exploration
                action = agent.select_action(state, evaluate=True)
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                state = next_state

            eval_rewards.append(episode_reward)
            eval_returns.append(info.get('total_return', 0))
            eval_sharpes.append(info.get('sharpe_ratio', 0))
            eval_drawdowns.append(info.get('max_drawdown', 0))

        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_return': np.mean(eval_returns),
            'mean_sharpe': np.mean(eval_sharpes),
            'mean_drawdown': np.mean(eval_drawdowns),
        }

    def save_checkpoint(self, agent: SACAgent, episode: int, metrics: Dict[str, Any]):
        """Sauvegarde un checkpoint"""
        checkpoint_path = self.checkpoints_dir / f"agent_{self.agent_id}_ep{episode}.pt"
        agent.save(str(checkpoint_path))

        # Sauvegarder aussi les métriques
        metrics_path = self.checkpoints_dir / f"metrics_ep{episode}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"💾 Checkpoint sauvegardé: {checkpoint_path.name}")

    def save_training_stats(self):
        """Sauvegarde les statistiques d'entraînement"""
        stats_path = self.logs_dir / "training_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2)

        # Sauvegarder aussi en format numpy pour analyse
        np.savez(
            self.logs_dir / "training_stats.npz",
            episode_rewards=np.array(self.training_stats['episode_rewards']),
            episode_lengths=np.array(self.training_stats['episode_lengths']),
            eval_rewards=np.array(self.training_stats['eval_rewards']),
            actor_losses=np.array(self.training_stats['actor_losses']),
            critic_losses=np.array(self.training_stats['critic_losses']),
            alpha_values=np.array(self.training_stats['alpha_values']),
        )

    def run_training(self):
        """Boucle principale d'entraînement"""
        print("\n" + "="*70)
        print("🎯 DÉMARRAGE DE L'ENTRAÎNEMENT SAC SUR KAGGLE")
        print("="*70)

        # 1. Charger les données
        train_data, val_data, test_data = self.load_data()

        # 2. Calculer les features
        train_features, val_features, test_features = self.compute_features(
            train_data, val_data, test_data
        )

        # 3. Créer les environnements
        train_env, val_env, env_config = self.create_environments(
            train_data, val_data, train_features, val_features
        )

        # 4. Créer l'agent
        agent, sac_config = self.create_agent()

        # 5. Boucle d'entraînement
        print("\n" + "="*70)
        print("🏋️  ENTRAÎNEMENT EN COURS")
        print("="*70)

        best_eval_reward = -float('inf')

        for episode in range(1, self.num_episodes + 1):
            # Entraîner un épisode
            metrics = self.train_episode(agent, train_env, episode)

            # Logger les métriques
            self.training_stats['episode_rewards'].append(metrics['reward'])
            self.training_stats['episode_lengths'].append(metrics['length'])
            self.training_stats['actor_losses'].append(metrics['actor_loss'])
            self.training_stats['critic_losses'].append(metrics['critic_loss'])
            self.training_stats['alpha_values'].append(metrics['alpha'])

            # Afficher les résultats
            print(f"\n📈 Episode {episode}/{self.num_episodes}")
            print(f"   Reward: {metrics['reward']:.2f}")
            print(f"   Length: {metrics['length']}")
            print(f"   Return: {metrics['total_return']:.2%}")
            print(f"   Sharpe: {metrics['sharpe']:.2f}")
            print(f"   MaxDD: {metrics['max_drawdown']:.2%}")
            print(f"   Actor Loss: {metrics['actor_loss']:.4f}")
            print(f"   Critic Loss: {metrics['critic_loss']:.4f}")
            print(f"   Alpha: {metrics['alpha']:.4f}")
            print(f"   Time: {metrics['time']:.1f}s")
            print(f"   Total Steps: {agent.total_steps}")

            # Évaluation périodique
            if episode % self.eval_frequency == 0:
                print(f"\n🔍 Évaluation (Episode {episode})...")
                eval_metrics = self.evaluate_agent(agent, val_env, num_episodes=3)
                self.training_stats['eval_rewards'].append(eval_metrics['mean_reward'])

                print(f"   Eval Reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
                print(f"   Eval Return: {eval_metrics['mean_return']:.2%}")
                print(f"   Eval Sharpe: {eval_metrics['mean_sharpe']:.2f}")
                print(f"   Eval MaxDD: {eval_metrics['mean_drawdown']:.2%}")

                # Sauvegarder le meilleur modèle
                if eval_metrics['mean_reward'] > best_eval_reward:
                    best_eval_reward = eval_metrics['mean_reward']
                    best_model_path = self.output_dir / f"agent_{self.agent_id}_best.pt"
                    agent.save(str(best_model_path))
                    print(f"   ⭐ Nouveau meilleur modèle sauvegardé!")

            # Sauvegarder des checkpoints périodiques
            if episode % self.checkpoint_frequency == 0:
                self.save_checkpoint(agent, episode, metrics)

        # 6. Sauvegarder le modèle final
        print("\n" + "="*70)
        print("💾 SAUVEGARDE DU MODÈLE FINAL")
        print("="*70)

        final_model_path = self.output_dir / f"agent_{self.agent_id}_final.pt"
        agent.save(str(final_model_path))
        print(f"✅ Modèle final sauvegardé: {final_model_path}")

        # Sauvegarder les statistiques
        self.save_training_stats()
        print(f"✅ Statistiques sauvegardées: {self.logs_dir}")

        # 7. Résumé final
        print("\n" + "="*70)
        print("🎉 ENTRAÎNEMENT TERMINÉ")
        print("="*70)
        print(f"   Total épisodes: {self.num_episodes}")
        print(f"   Total steps: {agent.total_steps}")
        print(f"   Meilleure récompense eval: {best_eval_reward:.2f}")
        print(f"   Récompense moyenne (10 derniers): {np.mean(self.training_stats['episode_rewards'][-10:]):.2f}")
        print(f"   Fichiers sauvegardés dans: {self.output_dir}")
        print("="*70)

        return agent, self.training_stats


def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(description="Entraînement SAC sur Kaggle")

    parser.add_argument(
        '--h5-path',
        type=str,
        default="/kaggle/input/trading-data/processed_data.h5",
        help="Chemin vers le fichier h5 avec les données"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default="/kaggle/working",
        help="Répertoire de sortie"
    )
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=100,
        help="Nombre d'épisodes d'entraînement"
    )
    parser.add_argument(
        '--eval-frequency',
        type=int,
        default=10,
        help="Fréquence d'évaluation (en épisodes)"
    )
    parser.add_argument(
        '--checkpoint-frequency',
        type=int,
        default=5,
        help="Fréquence de sauvegarde (en épisodes)"
    )
    parser.add_argument(
        '--agent-id',
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="ID de l'agent (1, 2, ou 3)"
    )
    parser.add_argument(
        '--device',
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device PyTorch"
    )

    args = parser.parse_args()

    # Créer le trainer
    trainer = KaggleTrainer(
        h5_path=args.h5_path,
        output_dir=args.output_dir,
        num_episodes=args.num_episodes,
        eval_frequency=args.eval_frequency,
        checkpoint_frequency=args.checkpoint_frequency,
        agent_id=args.agent_id,
        device=args.device
    )

    # Lancer l'entraînement
    agent, stats = trainer.run_training()

    print("\n✅ Script terminé avec succès!")


if __name__ == "__main__":
    main()
