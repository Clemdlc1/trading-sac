#!/usr/bin/env python
"""
Script de visualisation des métriques d'entraînement (style TensorBoard)
=========================================================================

Ce script charge les métriques sauvegardées et génère des graphiques détaillés
pour analyser l'entraînement de l'agent SAC.

Usage:
    python visualize_metrics.py [--metrics-file logs/training_metrics.json] [--output-dir reports/plots]

Author: Trading SAC System
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Configuration du style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (15, 8)
plt.rcParams['font.size'] = 10


def load_metrics(metrics_file: Path) -> Dict:
    """Charge les métriques depuis le fichier JSON"""
    if not metrics_file.exists():
        raise FileNotFoundError(f"Fichier de métriques introuvable: {metrics_file}")

    with open(metrics_file, 'r') as f:
        return json.load(f)


def plot_rewards(metrics: Dict, output_dir: Path):
    """Graphique des rewards (avec moyenne mobile)"""
    fig, ax = plt.subplots(figsize=(15, 6))

    episodes = metrics['episodes']
    rewards = metrics['episode_rewards']
    rewards_mean = metrics['episode_rewards_mean']
    rewards_std = metrics['episode_rewards_std']

    # Reward par épisode
    ax.plot(episodes, rewards, alpha=0.3, label='Episode Reward', color='blue', linewidth=0.5)

    # Moyenne mobile
    ax.plot(episodes, rewards_mean, label='Moving Average (100 episodes)', color='red', linewidth=2)

    # Bande de confiance (std)
    rewards_mean_arr = np.array(rewards_mean)
    rewards_std_arr = np.array(rewards_std)
    ax.fill_between(
        episodes,
        rewards_mean_arr - rewards_std_arr,
        rewards_mean_arr + rewards_std_arr,
        alpha=0.2,
        color='red',
        label='±1 Std Dev'
    )

    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Episode Rewards Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'rewards.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Graphique des rewards sauvegardé: {output_dir / 'rewards.png'}")


def plot_losses(metrics: Dict, output_dir: Path):
    """Graphique des losses (critic, actor, alpha)"""
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))

    episodes = metrics['episodes']
    critic_losses = metrics['critic_losses']
    actor_losses = metrics['actor_losses']
    alpha_losses = metrics['alpha_losses']

    # Critic Loss
    axes[0].plot(episodes, critic_losses, color='purple', linewidth=1)
    axes[0].set_ylabel('Critic Loss')
    axes[0].set_title('Critic Loss Over Training')
    axes[0].grid(True, alpha=0.3)

    # Actor Loss
    axes[1].plot(episodes, actor_losses, color='green', linewidth=1)
    axes[1].set_ylabel('Actor Loss')
    axes[1].set_title('Actor Loss Over Training')
    axes[1].grid(True, alpha=0.3)

    # Alpha Loss
    axes[2].plot(episodes, alpha_losses, color='orange', linewidth=1)
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Alpha Loss')
    axes[2].set_title('Alpha Loss Over Training')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'losses.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Graphique des losses sauvegardé: {output_dir / 'losses.png'}")


def plot_sac_params(metrics: Dict, output_dir: Path):
    """Graphique des paramètres SAC (alpha, target entropy, learning rates)"""
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))

    episodes = metrics['episodes']

    # Alpha
    alpha_values = metrics['alpha_values']
    target_entropies = metrics['target_entropies']
    axes[0].plot(episodes, alpha_values, label='Alpha (Temperature)', color='red', linewidth=1.5)
    axes[0].plot(episodes, target_entropies, label='Target Entropy', color='blue', linewidth=1.5, linestyle='--')
    axes[0].set_ylabel('Value')
    axes[0].set_title('SAC Alpha and Target Entropy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Learning Rates
    actor_lr = metrics['actor_lr']
    critic_lr = metrics['critic_lr']
    axes[1].plot(episodes, actor_lr, label='Actor LR', color='green', linewidth=1.5)
    axes[1].plot(episodes, critic_lr, label='Critic LR', color='purple', linewidth=1.5)
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rates Over Training')
    axes[1].legend()
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)

    # Buffer Size
    buffer_sizes = metrics['buffer_sizes']
    axes[2].plot(episodes, buffer_sizes, color='brown', linewidth=1.5)
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Buffer Size')
    axes[2].set_title('Replay Buffer Size')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'sac_params.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Graphique des paramètres SAC sauvegardé: {output_dir / 'sac_params.png'}")


def plot_exploration(metrics: Dict, output_dir: Path):
    """Graphique des métriques d'exploration (action mean/std)"""
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    episodes = metrics['episodes']
    action_mean = metrics['action_mean']
    action_std = metrics['action_std']

    # Action Mean
    axes[0].plot(episodes, action_mean, color='blue', linewidth=1.5)
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Zero (Neutral)')
    axes[0].set_ylabel('Action Mean')
    axes[0].set_title('Action Mean Over Training (Exploration vs Exploitation)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Action Std
    axes[1].plot(episodes, action_std, color='orange', linewidth=1.5)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Action Std Dev')
    axes[1].set_title('Action Standard Deviation (Exploration Level)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'exploration.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Graphique d'exploration sauvegardé: {output_dir / 'exploration.png'}")


def plot_performance(metrics: Dict, output_dir: Path):
    """Graphique des métriques de performance (Sharpe, Win Rate, Drawdown, etc.)"""
    fig, axes = plt.subplots(3, 2, figsize=(18, 14))

    episodes = metrics['episodes']

    # Sharpe Ratio
    axes[0, 0].plot(episodes, metrics['sharpe_ratios'], color='green', linewidth=1.5)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0, 0].set_ylabel('Sharpe Ratio')
    axes[0, 0].set_title('Sharpe Ratio')
    axes[0, 0].grid(True, alpha=0.3)

    # Sortino Ratio
    axes[0, 1].plot(episodes, metrics['sortino_ratios'], color='blue', linewidth=1.5)
    axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].set_ylabel('Sortino Ratio')
    axes[0, 1].set_title('Sortino Ratio')
    axes[0, 1].grid(True, alpha=0.3)

    # Win Rate
    axes[1, 0].plot(episodes, metrics['win_rates'], color='purple', linewidth=1.5)
    axes[1, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% (Random)')
    axes[1, 0].set_ylabel('Win Rate')
    axes[1, 0].set_title('Win Rate')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Max Drawdown
    axes[1, 1].plot(episodes, metrics['max_drawdowns'], color='red', linewidth=1.5)
    axes[1, 1].set_ylabel('Max Drawdown')
    axes[1, 1].set_title('Maximum Drawdown')
    axes[1, 1].grid(True, alpha=0.3)

    # Total Return
    axes[2, 0].plot(episodes, metrics['total_returns'], color='darkgreen', linewidth=1.5)
    axes[2, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[2, 0].set_xlabel('Episode')
    axes[2, 0].set_ylabel('Total Return')
    axes[2, 0].set_title('Total Return')
    axes[2, 0].grid(True, alpha=0.3)

    # Profit Factor
    axes[2, 1].plot(episodes, metrics['profit_factors'], color='orange', linewidth=1.5)
    axes[2, 1].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Break-even')
    axes[2, 1].set_xlabel('Episode')
    axes[2, 1].set_ylabel('Profit Factor')
    axes[2, 1].set_title('Profit Factor')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'performance.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Graphique de performance sauvegardé: {output_dir / 'performance.png'}")


def plot_buffer_composition(metrics: Dict, output_dir: Path):
    """Graphique de la composition du buffer (winning/losing/neutral)"""
    fig, ax = plt.subplots(figsize=(15, 6))

    episodes = metrics['episodes']
    winning = np.array(metrics['buffer_winning_ratio']) * 100
    losing = np.array(metrics['buffer_losing_ratio']) * 100
    neutral = np.array(metrics['buffer_neutral_ratio']) * 100

    # Stacked area chart
    ax.fill_between(episodes, 0, winning, label='Winning Trades', color='green', alpha=0.6)
    ax.fill_between(episodes, winning, winning + losing, label='Losing Trades', color='red', alpha=0.6)
    ax.fill_between(episodes, winning + losing, 100, label='Neutral', color='gray', alpha=0.6)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Replay Buffer Composition Over Training')
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'buffer_composition.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Graphique de composition du buffer sauvegardé: {output_dir / 'buffer_composition.png'}")


def plot_trades_evolution(metrics: Dict, output_dir: Path):
    """Graphique de l'évolution des trades (total, winning, losing)"""
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    episodes = metrics['episodes']
    total_trades = metrics['total_trades']
    winning_trades = metrics['winning_trades']
    losing_trades = metrics['losing_trades']

    # Total trades
    axes[0].plot(episodes, total_trades, label='Total Trades', color='blue', linewidth=1.5)
    axes[0].plot(episodes, winning_trades, label='Winning Trades', color='green', linewidth=1.5)
    axes[0].plot(episodes, losing_trades, label='Losing Trades', color='red', linewidth=1.5)
    axes[0].set_ylabel('Number of Trades')
    axes[0].set_title('Trade Count Evolution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Win rate (redundant but useful to see)
    win_rates = np.array(metrics['win_rates']) * 100
    axes[1].plot(episodes, win_rates, color='purple', linewidth=1.5)
    axes[1].axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% (Random)')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Win Rate (%)')
    axes[1].set_title('Win Rate Percentage')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'trades_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Graphique d'évolution des trades sauvegardé: {output_dir / 'trades_evolution.png'}")


def plot_equity_curve(metrics: Dict, output_dir: Path):
    """Graphique de la courbe d'équité finale"""
    fig, ax = plt.subplots(figsize=(15, 6))

    episodes = metrics['episodes']
    final_equities = metrics['final_equities']

    ax.plot(episodes, final_equities, color='darkblue', linewidth=1.5)
    ax.axhline(y=final_equities[0] if final_equities else 0, color='red', linestyle='--', alpha=0.5, label='Initial Equity')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Final Equity ($)')
    ax.set_title('Final Equity Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    plt.tight_layout()
    plt.savefig(output_dir / 'equity_curve.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Graphique de courbe d'équité sauvegardé: {output_dir / 'equity_curve.png'}")


def generate_summary_stats(metrics: Dict, output_dir: Path):
    """Génère un fichier texte avec les statistiques résumées"""
    num_episodes = len(metrics['episodes'])

    # Calculer statistiques
    final_reward_mean = np.mean(metrics['episode_rewards'][-100:]) if len(metrics['episode_rewards']) >= 100 else np.mean(metrics['episode_rewards'])
    final_sharpe = np.mean(metrics['sharpe_ratios'][-100:]) if len(metrics['sharpe_ratios']) >= 100 else np.mean(metrics['sharpe_ratios'])
    final_win_rate = np.mean(metrics['win_rates'][-100:]) if len(metrics['win_rates']) >= 100 else np.mean(metrics['win_rates'])
    final_drawdown = np.mean(metrics['max_drawdowns'][-100:]) if len(metrics['max_drawdowns']) >= 100 else np.mean(metrics['max_drawdowns'])

    best_episode = int(np.argmax(metrics['episode_rewards']))
    best_reward = float(np.max(metrics['episode_rewards']))

    worst_episode = int(np.argmin(metrics['episode_rewards']))
    worst_reward = float(np.min(metrics['episode_rewards']))

    # Écrire dans un fichier
    stats_file = output_dir / 'training_summary.txt'
    with open(stats_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TRAINING SUMMARY STATISTICS\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total Episodes: {num_episodes}\n")
        f.write(f"Total Steps: {metrics['total_steps'][-1] if metrics['total_steps'] else 0}\n\n")

        f.write("-" * 80 + "\n")
        f.write("FINAL PERFORMANCE (Last 100 episodes average)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Average Reward: {final_reward_mean:.4f}\n")
        f.write(f"Sharpe Ratio: {final_sharpe:.4f}\n")
        f.write(f"Win Rate: {final_win_rate:.2%}\n")
        f.write(f"Max Drawdown: {final_drawdown:.2%}\n\n")

        f.write("-" * 80 + "\n")
        f.write("BEST/WORST EPISODES\n")
        f.write("-" * 80 + "\n")
        f.write(f"Best Episode: {best_episode} (Reward: {best_reward:.4f})\n")
        f.write(f"Worst Episode: {worst_episode} (Reward: {worst_reward:.4f})\n\n")

        f.write("-" * 80 + "\n")
        f.write("FINAL VALUES\n")
        f.write("-" * 80 + "\n")
        f.write(f"Final Alpha: {metrics['alpha_values'][-1]:.6f}\n")
        f.write(f"Final Actor LR: {metrics['actor_lr'][-1]:.6e}\n")
        f.write(f"Final Critic LR: {metrics['critic_lr'][-1]:.6e}\n")
        f.write(f"Final Buffer Size: {metrics['buffer_sizes'][-1]}\n")
        f.write(f"Final Action Std: {metrics['action_std'][-1]:.4f}\n")

    print(f"✓ Statistiques résumées sauvegardées: {stats_file}")


def main():
    parser = argparse.ArgumentParser(description='Visualize training metrics (TensorBoard style)')
    parser.add_argument('--metrics-file', type=str, default='logs/training_metrics.json',
                        help='Path to metrics JSON file')
    parser.add_argument('--output-dir', type=str, default='reports/plots',
                        help='Output directory for plots')

    args = parser.parse_args()

    metrics_file = Path(args.metrics_file)
    output_dir = Path(args.output_dir)

    # Créer le dossier de sortie
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("TRAINING METRICS VISUALIZATION (TensorBoard Style)")
    print("=" * 80)
    print(f"\nChargement des métriques depuis: {metrics_file}")

    try:
        metrics = load_metrics(metrics_file)
    except FileNotFoundError as e:
        print(f"\n❌ Erreur: {e}")
        print("\nAssurez-vous qu'un entraînement a été effectué et que les métriques ont été sauvegardées.")
        sys.exit(1)

    num_episodes = len(metrics.get('episodes', []))
    print(f"✓ Métriques chargées: {num_episodes} épisodes")
    print(f"\nGénération des graphiques dans: {output_dir}\n")

    # Générer tous les graphiques
    plot_rewards(metrics, output_dir)
    plot_losses(metrics, output_dir)
    plot_sac_params(metrics, output_dir)
    plot_exploration(metrics, output_dir)
    plot_performance(metrics, output_dir)
    plot_buffer_composition(metrics, output_dir)
    plot_trades_evolution(metrics, output_dir)
    plot_equity_curve(metrics, output_dir)
    generate_summary_stats(metrics, output_dir)

    print("\n" + "=" * 80)
    print("✅ TOUS LES GRAPHIQUES ONT ÉTÉ GÉNÉRÉS AVEC SUCCÈS!")
    print("=" * 80)
    print(f"\nDossier de sortie: {output_dir}")
    print("\nGraphiques disponibles:")
    print("  - rewards.png: Récompenses et moyenne mobile")
    print("  - losses.png: Losses (critic, actor, alpha)")
    print("  - sac_params.png: Paramètres SAC (alpha, LR, buffer)")
    print("  - exploration.png: Métriques d'exploration")
    print("  - performance.png: Métriques de performance")
    print("  - buffer_composition.png: Composition du replay buffer")
    print("  - trades_evolution.png: Évolution des trades")
    print("  - equity_curve.png: Courbe d'équité")
    print("  - training_summary.txt: Statistiques résumées\n")


if __name__ == '__main__':
    main()
