"""
SAC EUR/USD Trading System - Auxiliary Task Learning
=====================================================

This module implements auxiliary task learning with autoencoder and K-Means
clustering for improved pattern recognition and sample efficiency.

Features:
- Price sequence autoencoder (50 bars lookback)
- K-Means clustering for market regime classification
- Combined loss function (RL + auxiliary)
- Pre-training and fine-tuning support
- Latent representation extraction
- Model persistence

Author: SAC EUR/USD Project
Version: 2.0
"""

import logging
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


@dataclass
class AuxiliaryConfig:
    """Configuration for auxiliary task learning."""
    
    # Autoencoder architecture
    lookback_window: int = 50  # Number of price bars in sequence
    encoder_dims: List[int] = None
    latent_dim: int = 32
    decoder_dims: List[int] = None
    
    # K-Means clustering
    n_clusters: int = 5
    cluster_names: Dict[int, str] = None
    
    # Training parameters
    pretrain_sequences: int = 100000
    pretrain_epochs: int = 50
    pretrain_lr: float = 1e-3
    pretrain_batch_size: int = 256
    
    # Combined loss weights
    rl_weight: float = 0.80
    auxiliary_weight: float = 0.20
    classification_weight_start: float = 0.20
    classification_weight_end: float = 0.05
    classification_decay_steps: int = 500000
    
    # Update frequency
    kmeans_update_interval: int = 10000
    
    # Model paths
    models_dir: Path = Path("models/auxiliary")
    
    def __post_init__(self):
        """Initialize default values and create directories."""
        if self.encoder_dims is None:
            self.encoder_dims = [256, 128, 64]
        if self.decoder_dims is None:
            self.decoder_dims = [64, 128, 256]
        if self.cluster_names is None:
            self.cluster_names = {
                0: 'ranging',
                1: 'trending_up',
                2: 'trending_down',
                3: 'high_volatility',
                4: 'low_volatility'
            }
        self.models_dir.mkdir(parents=True, exist_ok=True)


class PriceSequenceEncoder(nn.Module):
    """Encoder network for price sequences."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int
    ):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.1))
            current_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.latent_layer = nn.Linear(current_dim, latent_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode price sequence to latent representation.
        
        Args:
            x: Input tensor [batch_size, sequence_length]
            
        Returns:
            Latent representation [batch_size, latent_dim]
        """
        features = self.encoder(x)
        latent = self.latent_layer(features)
        return latent


class PriceSequenceDecoder(nn.Module):
    """Decoder network for price sequences."""
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dims: List[int],
        output_dim: int
    ):
        super().__init__()
        
        layers = []
        current_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.1))
            current_dim = hidden_dim
        
        self.decoder = nn.Sequential(*layers)
        self.output_layer = nn.Linear(current_dim, output_dim)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to price sequence.
        
        Args:
            z: Latent tensor [batch_size, latent_dim]
            
        Returns:
            Reconstructed sequence [batch_size, sequence_length]
        """
        features = self.decoder(z)
        reconstruction = self.output_layer(features)
        return reconstruction


class Autoencoder(nn.Module):
    """Complete autoencoder for price sequences."""
    
    def __init__(self, config: AuxiliaryConfig):
        super().__init__()
        
        self.config = config
        
        self.encoder = PriceSequenceEncoder(
            input_dim=config.lookback_window,
            hidden_dims=config.encoder_dims,
            latent_dim=config.latent_dim
        )
        
        self.decoder = PriceSequenceDecoder(
            latent_dim=config.latent_dim,
            hidden_dims=config.decoder_dims,
            output_dim=config.lookback_window
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.
        
        Args:
            x: Input sequence [batch_size, sequence_length]
            
        Returns:
            Tuple of (reconstruction, latent_representation)
        """
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space."""
        return self.decoder(z)


class MarketRegimeClassifier:
    """K-Means based market regime classifier."""
    
    def __init__(self, config: AuxiliaryConfig):
        self.config = config
        self.kmeans = KMeans(
            n_clusters=config.n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        self.is_fitted = False
        self.cluster_centers_ = None
    
    def fit(self, latent_representations: np.ndarray):
        """
        Fit K-Means on latent representations.
        
        Args:
            latent_representations: Array of shape [n_samples, latent_dim]
        """
        logger.info(f"Fitting K-Means with {len(latent_representations)} samples...")
        self.kmeans.fit(latent_representations)
        self.cluster_centers_ = self.kmeans.cluster_centers_
        self.is_fitted = True
        logger.info("K-Means fitting complete")
        
        # Log cluster statistics
        labels = self.kmeans.labels_
        for i in range(self.config.n_clusters):
            count = np.sum(labels == i)
            percentage = count / len(labels) * 100
            cluster_name = self.config.cluster_names.get(i, f"cluster_{i}")
            logger.info(f"  {cluster_name}: {count} samples ({percentage:.1f}%)")
    
    def predict(self, latent_representations: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels.
        
        Args:
            latent_representations: Array of shape [n_samples, latent_dim]
            
        Returns:
            Cluster labels [n_samples]
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction")
        return self.kmeans.predict(latent_representations)
    
    def predict_proba(self, latent_representations: np.ndarray) -> np.ndarray:
        """
        Predict cluster probabilities (soft assignment).
        
        Args:
            latent_representations: Array of shape [n_samples, latent_dim]
            
        Returns:
            Probabilities [n_samples, n_clusters]
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction")
        
        # Calculate distances to all cluster centers
        distances = self.kmeans.transform(latent_representations)
        
        # Convert distances to probabilities (inverse distance weighting)
        # Use temperature parameter for softness
        temperature = 1.0
        scores = -distances / temperature
        probs = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        
        return probs


class AuxiliaryTaskLearner:
    """Main auxiliary task learning orchestrator."""
    
    def __init__(self, config: Optional[AuxiliaryConfig] = None):
        self.config = config or AuxiliaryConfig()
        
        # Initialize autoencoder
        self.autoencoder = Autoencoder(self.config).to(device)
        self.optimizer = optim.Adam(
            self.autoencoder.parameters(),
            lr=self.config.pretrain_lr
        )
        
        # Initialize classifier
        self.classifier = MarketRegimeClassifier(self.config)
        
        # Training state
        self.training_step = 0
        self.encoder_frozen = False
        
        logger.info("Auxiliary Task Learner initialized")
        logger.info(f"  Lookback window: {self.config.lookback_window}")
        logger.info(f"  Latent dim: {self.config.latent_dim}")
        logger.info(f"  N clusters: {self.config.n_clusters}")
    
    def prepare_sequences(
        self,
        prices: np.ndarray,
        num_sequences: int
    ) -> np.ndarray:
        """
        Prepare price sequences for training.
        
        Args:
            prices: Array of close prices
            num_sequences: Number of sequences to generate
            
        Returns:
            Array of shape [num_sequences, lookback_window]
        """
        sequences = []
        
        # Generate random starting points
        max_start = len(prices) - self.config.lookback_window
        start_indices = np.random.randint(0, max_start, size=num_sequences)
        
        for start_idx in start_indices:
            sequence = prices[start_idx:start_idx + self.config.lookback_window]
            
            # Normalize sequence (mean=0, std=1)
            sequence_norm = (sequence - np.mean(sequence)) / (np.std(sequence) + 1e-8)
            sequences.append(sequence_norm)
        
        return np.array(sequences, dtype=np.float32)
    
    def pretrain_autoencoder(
        self,
        prices: np.ndarray,
        validation_split: float = 0.2
    ):
        """
        Pre-train autoencoder on price sequences.
        
        Args:
            prices: Array of close prices
            validation_split: Fraction of data for validation
        """
        logger.info("="*80)
        logger.info("Pre-training Autoencoder")
        logger.info("="*80)
        
        # Prepare sequences
        logger.info(f"Preparing {self.config.pretrain_sequences} sequences...")
        sequences = self.prepare_sequences(prices, self.config.pretrain_sequences)
        
        # Train/validation split
        n_val = int(len(sequences) * validation_split)
        val_sequences = sequences[:n_val]
        train_sequences = sequences[n_val:]
        
        logger.info(f"Training samples: {len(train_sequences)}")
        logger.info(f"Validation samples: {len(val_sequences)}")
        
        # Convert to tensors
        train_tensor = torch.FloatTensor(train_sequences)
        val_tensor = torch.FloatTensor(val_sequences)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(self.config.pretrain_epochs):
            # Training
            self.autoencoder.train()
            train_losses = []
            
            # Create batches
            n_batches = len(train_tensor) // self.config.pretrain_batch_size
            
            for i in range(n_batches):
                start_idx = i * self.config.pretrain_batch_size
                end_idx = start_idx + self.config.pretrain_batch_size
                batch = train_tensor[start_idx:end_idx].to(device)
                
                # Forward pass
                reconstruction, latent = self.autoencoder(batch)
                
                # Calculate loss
                recon_loss = F.mse_loss(reconstruction, batch)
                
                # Backward pass
                self.optimizer.zero_grad()
                recon_loss.backward()
                self.optimizer.step()
                
                train_losses.append(recon_loss.item())
            
            # Validation
            self.autoencoder.eval()
            with torch.no_grad():
                val_reconstruction, _ = self.autoencoder(val_tensor.to(device))
                val_loss = F.mse_loss(val_reconstruction, val_tensor.to(device))
            
            # Log progress
            avg_train_loss = np.mean(train_losses)
            logger.info(
                f"Epoch {epoch+1}/{self.config.pretrain_epochs}: "
                f"Train Loss={avg_train_loss:.6f}, Val Loss={val_loss:.6f}"
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_autoencoder("autoencoder_best.pt")
        
        logger.info(f"\nPre-training complete! Best validation loss: {best_val_loss:.6f}")
        logger.info("="*80)
    
    def fit_classifier(self, prices: np.ndarray, num_samples: int = 50000):
        """
        Fit K-Means classifier on latent representations.
        
        Args:
            prices: Array of close prices
            num_samples: Number of samples for fitting
        """
        logger.info("Fitting K-Means classifier...")
        
        # Prepare sequences
        sequences = self.prepare_sequences(prices, num_samples)
        
        # Extract latent representations
        self.autoencoder.eval()
        with torch.no_grad():
            latent_reps = []
            batch_size = 1000
            
            for i in range(0, len(sequences), batch_size):
                batch = torch.FloatTensor(sequences[i:i+batch_size]).to(device)
                latent = self.autoencoder.encode(batch)
                latent_reps.append(latent.cpu().numpy())
            
            latent_reps = np.concatenate(latent_reps, axis=0)
        
        # Fit classifier
        self.classifier.fit(latent_reps)
    
    def get_latent_representation(self, price_sequence: np.ndarray) -> np.ndarray:
        """
        Get latent representation for a price sequence.
        
        Args:
            price_sequence: Array of shape [lookback_window] or [batch, lookback_window]
            
        Returns:
            Latent representation
        """
        # Normalize sequence
        if price_sequence.ndim == 1:
            price_sequence = price_sequence.reshape(1, -1)
        
        sequences_norm = []
        for seq in price_sequence:
            seq_norm = (seq - np.mean(seq)) / (np.std(seq) + 1e-8)
            sequences_norm.append(seq_norm)
        
        sequences_norm = np.array(sequences_norm, dtype=np.float32)
        
        # Encode
        self.autoencoder.eval()
        with torch.no_grad():
            tensor = torch.FloatTensor(sequences_norm).to(device)
            latent = self.autoencoder.encode(tensor)
            return latent.cpu().numpy()
    
    def get_cluster_label(self, latent_representation: np.ndarray) -> int:
        """
        Get cluster label for latent representation.
        
        Args:
            latent_representation: Array of shape [latent_dim] or [batch, latent_dim]
            
        Returns:
            Cluster label(s)
        """
        if latent_representation.ndim == 1:
            latent_representation = latent_representation.reshape(1, -1)
        
        labels = self.classifier.predict(latent_representation)
        return labels[0] if len(labels) == 1 else labels
    
    def get_cluster_probabilities(self, latent_representation: np.ndarray) -> np.ndarray:
        """
        Get cluster probabilities for latent representation.
        
        Args:
            latent_representation: Array of shape [latent_dim] or [batch, latent_dim]
            
        Returns:
            Cluster probabilities
        """
        if latent_representation.ndim == 1:
            latent_representation = latent_representation.reshape(1, -1)
        
        return self.classifier.predict_proba(latent_representation)
    
    def calculate_auxiliary_loss(
        self,
        price_sequences: torch.Tensor,
        cluster_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate auxiliary loss (reconstruction + classification).
        
        Args:
            price_sequences: Tensor of shape [batch, lookback_window]
            cluster_labels: Tensor of shape [batch] with cluster labels
            
        Returns:
            Tuple of (reconstruction_loss, classification_loss)
        """
        # Forward pass
        reconstruction, latent = self.autoencoder(price_sequences)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstruction, price_sequences)
        
        # Classification loss (distance to assigned cluster center)
        # Get cluster centers as tensor
        cluster_centers = torch.FloatTensor(self.classifier.cluster_centers_).to(device)
        
        # Get assigned cluster centers
        assigned_centers = cluster_centers[cluster_labels]
        
        # Calculate distance loss
        classification_loss = F.mse_loss(latent, assigned_centers)
        
        return recon_loss, classification_loss
    
    def get_classification_weight(self) -> float:
        """
        Get current classification loss weight (decays over training).
        
        Returns:
            Current weight
        """
        if self.training_step >= self.config.classification_decay_steps:
            return self.config.classification_weight_end
        
        # Linear decay
        progress = self.training_step / self.config.classification_decay_steps
        weight = (
            self.config.classification_weight_start +
            (self.config.classification_weight_end - self.config.classification_weight_start) * progress
        )
        
        return weight
    
    def freeze_encoder(self):
        """Freeze encoder weights (after pre-training)."""
        for param in self.autoencoder.encoder.parameters():
            param.requires_grad = False
        self.encoder_frozen = True
        logger.info("Encoder frozen")
    
    def update_classifier(self, prices: np.ndarray):
        """
        Update K-Means classifier (called periodically during training).
        
        Args:
            prices: Recent price data
        """
        logger.info("Updating K-Means classifier...")
        self.fit_classifier(prices, num_samples=20000)
    
    def save_autoencoder(self, filename: str) -> Path:
        """
        Save autoencoder model.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.config.models_dir / filename
        
        torch.save({
            'autoencoder_state_dict': self.autoencoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_step': self.training_step,
            'encoder_frozen': self.encoder_frozen
        }, output_path)
        
        logger.info(f"Autoencoder saved to {output_path}")
        return output_path
    
    def load_autoencoder(self, filename: str):
        """
        Load autoencoder model.
        
        Args:
            filename: Input filename
        """
        input_path = self.config.models_dir / filename
        
        if not input_path.exists():
            raise FileNotFoundError(f"Autoencoder file not found: {input_path}")
        
        checkpoint = torch.load(input_path, map_location=device)
        
        self.autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint['training_step']
        self.encoder_frozen = checkpoint['encoder_frozen']
        
        logger.info(f"Autoencoder loaded from {input_path}")
    
    def save_classifier(self, filename: str = "kmeans_classifier.pkl") -> Path:
        """
        Save K-Means classifier.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.config.models_dir / filename
        
        with open(output_path, 'wb') as f:
            pickle.dump({
                'kmeans': self.classifier.kmeans,
                'is_fitted': self.classifier.is_fitted,
                'cluster_centers': self.classifier.cluster_centers_,
                'config': self.config
            }, f)
        
        logger.info(f"Classifier saved to {output_path}")
        return output_path
    
    def load_classifier(self, filename: str = "kmeans_classifier.pkl"):
        """
        Load K-Means classifier.
        
        Args:
            filename: Input filename
        """
        input_path = self.config.models_dir / filename
        
        if not input_path.exists():
            raise FileNotFoundError(f"Classifier file not found: {input_path}")
        
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
        
        self.classifier.kmeans = data['kmeans']
        self.classifier.is_fitted = data['is_fitted']
        self.classifier.cluster_centers_ = data['cluster_centers']
        
        logger.info(f"Classifier loaded from {input_path}")
    
    def analyze_clusters(self, prices: np.ndarray, num_samples: int = 10000) -> Dict:
        """
        Analyze cluster characteristics.
        
        Args:
            prices: Array of close prices
            num_samples: Number of samples for analysis
            
        Returns:
            Dictionary with cluster analysis
        """
        logger.info("Analyzing clusters...")
        
        # Prepare sequences
        sequences = self.prepare_sequences(prices, num_samples)
        
        # Get latent representations and labels
        latent_reps = self.get_latent_representation(sequences)
        labels = self.classifier.predict(latent_reps)
        
        # Analyze each cluster
        analysis = {}
        
        for cluster_id in range(self.config.n_clusters):
            cluster_mask = labels == cluster_id
            cluster_sequences = sequences[cluster_mask]
            
            if len(cluster_sequences) > 0:
                # Calculate statistics
                mean_sequence = np.mean(cluster_sequences, axis=0)
                std_sequence = np.std(cluster_sequences, axis=0)
                
                # Calculate trend (linear regression slope)
                x = np.arange(len(mean_sequence))
                slope = np.polyfit(x, mean_sequence, 1)[0]
                
                # Calculate volatility
                volatility = np.mean([np.std(seq) for seq in cluster_sequences])
                
                cluster_name = self.config.cluster_names.get(cluster_id, f"cluster_{cluster_id}")
                
                analysis[cluster_name] = {
                    'count': int(np.sum(cluster_mask)),
                    'percentage': float(np.sum(cluster_mask) / len(labels) * 100),
                    'mean_slope': float(slope),
                    'mean_volatility': float(volatility),
                    'mean_sequence': mean_sequence.tolist()
                }
        
        return analysis


def main():
    """Example usage of auxiliary task learning."""
    from backend.data_pipeline import DataPipeline
    
    # Load data
    logger.info("Loading processed data...")
    data_pipeline = DataPipeline()
    train_data, val_data, test_data = data_pipeline.get_processed_data()
    
    # Get EUR/USD close prices
    prices = train_data['EURUSD']['close'].values
    logger.info(f"Loaded {len(prices)} price bars")
    
    # Initialize auxiliary task learner
    config = AuxiliaryConfig(
        lookback_window=50,
        latent_dim=32,
        n_clusters=5,
        pretrain_sequences=100000,
        pretrain_epochs=50
    )
    
    learner = AuxiliaryTaskLearner(config)
    
    # Pre-train autoencoder
    logger.info("\n" + "="*80)
    logger.info("Step 1: Pre-training Autoencoder")
    logger.info("="*80)
    learner.pretrain_autoencoder(prices, validation_split=0.2)
    
    # Fit K-Means classifier
    logger.info("\n" + "="*80)
    logger.info("Step 2: Fitting K-Means Classifier")
    logger.info("="*80)
    learner.fit_classifier(prices, num_samples=50000)
    
    # Freeze encoder
    learner.freeze_encoder()
    
    # Analyze clusters
    logger.info("\n" + "="*80)
    logger.info("Step 3: Analyzing Clusters")
    logger.info("="*80)
    cluster_analysis = learner.analyze_clusters(prices, num_samples=10000)
    
    print("\n" + "="*80)
    print("Cluster Analysis Results")
    print("="*80)
    for cluster_name, stats in cluster_analysis.items():
        print(f"\n{cluster_name}:")
        print(f"  Count: {stats['count']} ({stats['percentage']:.1f}%)")
        print(f"  Mean slope: {stats['mean_slope']:.6f}")
        print(f"  Mean volatility: {stats['mean_volatility']:.6f}")
    
    # Test encoding/decoding
    logger.info("\n" + "="*80)
    logger.info("Step 4: Testing Encoding/Decoding")
    logger.info("="*80)
    
    test_sequence = prices[1000:1050]
    latent_rep = learner.get_latent_representation(test_sequence)
    cluster_label = learner.get_cluster_label(latent_rep)
    cluster_probs = learner.get_cluster_probabilities(latent_rep)
    
    print(f"\nTest sequence latent dim: {latent_rep.shape}")
    print(f"Assigned cluster: {config.cluster_names[cluster_label]}")
    print(f"Cluster probabilities:")
    for i, prob in enumerate(cluster_probs[0]):
        print(f"  {config.cluster_names[i]}: {prob:.3f}")
    
    # Save models
    logger.info("\n" + "="*80)
    logger.info("Step 5: Saving Models")
    logger.info("="*80)
    learner.save_autoencoder("autoencoder_final.pt")
    learner.save_classifier("kmeans_final.pkl")
    
    logger.info("\n" + "="*80)
    logger.info("Auxiliary Task Learning Demo Complete!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
