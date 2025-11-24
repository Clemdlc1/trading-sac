"""
SAC EUR/USD Trading System - Data Pipeline
===========================================

This module handles data acquisition, aggregation, cleaning, and persistence
for the algorithmic trading system.

Features:
- Download FX data from GitHub (philipperemy/FX-1-Minute-Data)
- Aggregate 1-minute bars to 5-minute bars
- Robust outlier detection and cleaning
- OHLC validation
- Temporal alignment across all pairs
- Train/Validation/Test split with proper time-series handling
- HDF5 persistence for fast loading

Author: SAC EUR/USD Project
Version: 2.0
"""

import logging
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import zipfile
import tempfile

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import histdata API (after logger is defined)
try:
    from histdata import download_hist_data as dl
    from histdata.api import Platform as P, TimeFrame as TF
    HISTDATA_AVAILABLE = True
except ImportError:
    HISTDATA_AVAILABLE = False
    logger.warning("histdata package not installed. Install with: pip install histdata")


@dataclass
class DataConfig:
    """Configuration for data pipeline."""
    
    # Data source
    github_base_url: str = "https://github.com/philipperemy/FX-1-Minute-Data"
    
    # Currency pairs to download
    pairs: List[str] = field(default_factory=lambda: [
        'EURUSD',  # Main pair
        'USDJPY',
        'GBPUSD',
        'USDCAD',
        'USDSEK',
        'USDCHF',
        'EURGBP',
        'EURJPY',
        'XAUUSD',  # Gold
        'SPXUSD',  # S&P 500 (available in histdata)
    ])
    
    # Date ranges - training starts in 2012 to avoid 2011 noise
    train_start: str = "2012-01-01"  # Changed from 2011-01-01 (2011 has too much noise)
    train_end: str = "2023-12-31"
    val_start: str = "2019-01-01"  # Intentional overlap
    val_end: str = "2023-12-31"
    test_start: str = "2024-01-01"
    test_end: str = "2025-01-31"

    # Corrupted data periods to exclude (format: list of tuples with start and end dates)
    # Based on analysis of oscillating US session data corruption
    exclude_periods: List[Tuple[str, str]] = field(default_factory=lambda: [
        # Add corrupted periods here - example: ("2023-06-01", "2023-06-15")
        ("2023-02-16", "2023-07-31"),
    ])
    
    # Aggregation parameters
    source_timeframe: str = "1min"
    target_timeframe: str = "5min"
    aggregation_bars: int = 5
    
    # Cleaning parameters
    outlier_sigma: float = 5.0
    min_price: float = 0.0001
    max_price: float = 1000000.0
    
    # File paths
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    normalized_data_dir: Path = Path("data/normalized")
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.normalized_data_dir.mkdir(parents=True, exist_ok=True)


class DataDownloader:
    """Download FX data using histdata API from philipperemy/FX-1-Minute-Data."""
    
    # Mapping from system pair names to histdata pair names
    PAIR_MAPPING = {
        'EURUSD': 'eurusd',
        'USDJPY': 'usdjpy',
        'GBPUSD': 'gbpusd',
        'USDCAD': 'usdcad',
        'USDSEK': 'usdsek',
        'USDCHF': 'usdchf',
        'EURGBP': 'eurgbp',
        'EURJPY': 'eurjpy',
        'XAUUSD': 'xauusd',  # Gold
        'SPXUSD': 'spxusd',  # S&P 500
    }
    
    def __init__(self, config: DataConfig):
        self.config = config
        if not HISTDATA_AVAILABLE:
            raise ImportError(
                "histdata package is required. Install with: pip install histdata\n"
                "See: https://github.com/philipperemy/FX-1-Minute-Data"
            )
    
    def _get_histdata_pair_name(self, pair: str) -> str:
        """Convert system pair name to histdata pair name."""
        return self.PAIR_MAPPING.get(pair.upper(), pair.lower())
    
    def _parse_histdata_csv(self, csv_content: str) -> pd.DataFrame:
        """
        Parse histdata CSV format.
        
        Format: DateTime Stamp;Bar OPEN Bid Quote;Bar HIGH Bid Quote;Bar LOW Bid Quote;Bar CLOSE Bid Quote;Volume
        Timestamp format: YYYYMMDD HHMMSS (EST timezone, no DST)
        
        Args:
            csv_content: CSV content as string
            
        Returns:
            DataFrame with OHLC data
        """
        lines = csv_content.strip().split('\n')
        data = []
        
        for line in lines:
            if not line.strip():
                continue
            parts = line.split(';')
            if len(parts) < 5:
                continue
            
            # Parse timestamp: YYYYMMDD HHMMSS
            timestamp_str = parts[0].strip()
            try:
                timestamp = pd.to_datetime(timestamp_str, format='%Y%m%d %H%M%S')
                # Convert EST to UTC (EST is UTC-5, no DST)
                timestamp = timestamp + pd.Timedelta(hours=5)
            except:
                logger.warning(f"Failed to parse timestamp: {timestamp_str}")
                continue
            
            try:
                open_price = float(parts[1])
                high_price = float(parts[2])
                low_price = float(parts[3])
                close_price = float(parts[4])
                
                data.append({
                    'timestamp': timestamp,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price
                })
            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse line: {line[:50]}... Error: {e}")
                continue
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df
    
    def download_pair_month(self, pair: str, year: int, month: int) -> Optional[pd.DataFrame]:
        """
        Download data for a specific pair, year, and month using histdata API.

        Args:
            pair: Currency pair (e.g., 'EURUSD')
            year: Year to download
            month: Month to download (1-12)

        Returns:
            DataFrame with OHLC data or None if download fails
        """
        try:
            # Convert pair to histdata format
            pair_lower = self._get_histdata_pair_name(pair)

            # Download using histdata API
            logger.debug(f"Downloading {pair} {year}-{month:02d} via histdata API...")

            # Save current directory
            original_cwd = os.getcwd()

            # Create a temporary directory and change to it
            temp_dir = tempfile.mkdtemp()
            temp_path = Path(temp_dir)

            try:
                os.chdir(str(temp_path))

                try:
                    # Log the exact parameters being sent to the API
                    params = {
                        'year': str(year),
                        'month': str(month),
                        'pair': pair_lower,
                        'platform': P.GENERIC_ASCII,
                        'time_frame': TF.ONE_MINUTE
                    }
                    logger.info(f"🔍 DEBUG: Calling histdata API with params: {params}")
                    logger.info(f"🔍 DEBUG: Platform value: {P.GENERIC_ASCII}, TimeFrame value: {TF.ONE_MINUTE}")

                    file_path = dl(
                        year=str(year),
                        month=str(month),
                        pair=pair_lower,
                        platform=P.GENERIC_ASCII,
                        time_frame=TF.ONE_MINUTE
                    )

                    logger.info(f"🔍 DEBUG: API returned file_path: {file_path}, type: {type(file_path)}")
                finally:
                    # Restore original directory
                    os.chdir(original_cwd)

                # Handle file path - histdata may return None or a path
                if file_path is None:
                    logger.info(f"🔍 DEBUG: file_path is None, searching for downloaded files...")

                    # List all files in temp directory
                    temp_files = list(temp_path.glob("*"))
                    logger.info(f"🔍 DEBUG: Files in temp directory: {[f.name for f in temp_files]}")

                    # List all files in current directory
                    current_files = list(Path(original_cwd).glob("*.zip"))
                    logger.info(f"🔍 DEBUG: ZIP files in current directory: {[f.name for f in current_files]}")

                    # Try to find the downloaded file in temp directory
                    pattern = f"*{pair_lower}*{year}{month:02d}*.zip"
                    logger.info(f"🔍 DEBUG: Searching with pattern: {pattern}")
                    matches = list(temp_path.glob(pattern))
                    logger.info(f"🔍 DEBUG: Matches in temp dir: {[f.name for f in matches]}")

                    if matches:
                        file_path = matches[0]
                        logger.info(f"🔍 DEBUG: Found file in temp dir: {file_path}")
                    else:
                        # Also check current directory (histdata might download there)
                        current_dir = Path(original_cwd)
                        matches = list(current_dir.glob(pattern))
                        logger.info(f"🔍 DEBUG: Matches in current dir: {[f.name for f in matches]}")

                        if matches:
                            file_path = matches[0]
                            logger.info(f"🔍 DEBUG: Found file in current dir: {file_path}")
                            # Copy to temp directory
                            import shutil
                            dest_path = temp_path / file_path.name
                            shutil.copy2(file_path, dest_path)
                            file_path = dest_path
                            # Remove from current directory
                            matches[0].unlink()
                            logger.info(f"🔍 DEBUG: Moved file to temp dir: {file_path}")
                        else:
                            logger.warning(f"❌ No file returned for {pair} {year}-{month:02d}")
                            return None

                # Handle both string path and Path object
                if isinstance(file_path, str):
                    # The API returns a relative path like ".\filename.zip" which is relative to temp_path
                    # Since we've changed back to original_cwd, we need to make it relative to temp_path
                    file_path = temp_path / Path(file_path).name
                    logger.info(f"🔍 DEBUG: Converted file_path to: {file_path}")
                elif isinstance(file_path, Path):
                    # If it's already a Path, ensure it's relative to temp_path
                    if not file_path.is_absolute():
                        file_path = temp_path / file_path.name
                        logger.info(f"🔍 DEBUG: Converted Path to: {file_path}")
                else:
                    logger.warning(f"Unexpected file_path type: {type(file_path)}")
                    return None

                # Check if file exists
                if not file_path.exists():
                    logger.warning(f"Downloaded file not found: {file_path}")
                    return None

                # Extract and parse CSV
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    # Find CSV file in zip
                    csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                    if not csv_files:
                        logger.warning(f"No CSV file found in {file_path}")
                        return None

                    csv_file = csv_files[0]
                    csv_content = zip_ref.read(csv_file).decode('utf-8')
                    df = self._parse_histdata_csv(csv_content)

                    if len(df) > 0:
                        logger.info(f"Downloaded {pair} {year}-{month:02d}: {len(df)} rows")
                        return df
                    else:
                        logger.warning(f"Empty DataFrame for {pair} {year}-{month:02d}")
                        return None

            except Exception as e:
                logger.error(f"❌ EXCEPTION during download {pair} {year}-{month:02d}")
                logger.error(f"❌ Exception type: {type(e).__name__}")
                logger.error(f"❌ Exception message: {str(e)}")
                import traceback
                logger.error(f"❌ Full traceback:\n{traceback.format_exc()}")
                return None
            finally:
                # Clean up temporary directory
                try:
                    import shutil
                    import time
                    time.sleep(0.1)  # Small delay to ensure all file handles are closed
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception as e:
                    logger.debug(f"Failed to clean up temp directory: {e}")

        except Exception as e:
            logger.error(f"Error downloading {pair} {year}-{month:02d}: {e}")
            return None

    def download_pair_year(self, pair: str, year: int) -> Optional[pd.DataFrame]:
        """
        Download data for a specific pair and entire year using histdata API with month=None.
        This is used for past years where the histdata library requires month=None.

        Args:
            pair: Currency pair (e.g., 'EURUSD')
            year: Year to download

        Returns:
            DataFrame with OHLC data or None if download fails
        """
        try:
            # Convert pair to histdata format
            pair_lower = self._get_histdata_pair_name(pair)

            # Download using histdata API
            logger.debug(f"Downloading {pair} {year} (full year) via histdata API...")

            # Save current directory
            original_cwd = os.getcwd()

            # Create a temporary directory and change to it
            temp_dir = tempfile.mkdtemp()
            temp_path = Path(temp_dir)

            try:
                os.chdir(str(temp_path))

                try:
                    # Log the exact parameters being sent to the API
                    params = {
                        'year': str(year),
                        'month': None,  # For past years, library requires month=None
                        'pair': pair_lower,
                        'platform': P.GENERIC_ASCII,
                        'time_frame': TF.ONE_MINUTE
                    }
                    logger.info(f"🔍 DEBUG: Calling histdata API for full year with params: {params}")

                    file_path = dl(
                        year=str(year),
                        month=None,  # Past years require month=None
                        pair=pair_lower,
                        platform=P.GENERIC_ASCII,
                        time_frame=TF.ONE_MINUTE
                    )

                    logger.info(f"🔍 DEBUG: API returned file_path: {file_path}, type: {type(file_path)}")
                finally:
                    # Restore original directory
                    os.chdir(original_cwd)

                # Handle file path - histdata may return None or a path
                if file_path is None:
                    logger.info(f"🔍 DEBUG: file_path is None, searching for downloaded files...")

                    # List all files in temp directory
                    temp_files = list(temp_path.glob("*"))
                    logger.info(f"🔍 DEBUG: Files in temp directory: {[f.name for f in temp_files]}")

                    # Try to find the downloaded file in temp directory
                    pattern = f"*{pair_lower}*{year}*.zip"
                    logger.info(f"🔍 DEBUG: Searching with pattern: {pattern}")
                    matches = list(temp_path.glob(pattern))
                    logger.info(f"🔍 DEBUG: Matches in temp dir: {[f.name for f in matches]}")

                    if matches:
                        file_path = matches[0]
                        logger.info(f"🔍 DEBUG: Found file in temp dir: {file_path}")
                    else:
                        # Also check current directory
                        current_dir = Path(original_cwd)
                        matches = list(current_dir.glob(pattern))
                        logger.info(f"🔍 DEBUG: Matches in current dir: {[f.name for f in matches]}")

                        if matches:
                            file_path = matches[0]
                            logger.info(f"🔍 DEBUG: Found file in current dir: {file_path}")
                            # Copy to temp directory
                            import shutil
                            dest_path = temp_path / file_path.name
                            shutil.copy2(file_path, dest_path)
                            file_path = dest_path
                            # Remove from current directory
                            matches[0].unlink()
                            logger.info(f"🔍 DEBUG: Moved file to temp dir: {file_path}")
                        else:
                            logger.warning(f"❌ No file returned for {pair} {year}")
                            return None

                # Handle both string path and Path object
                if isinstance(file_path, str):
                    # The API returns a relative path like ".\filename.zip" which is relative to temp_path
                    # Since we've changed back to original_cwd, we need to make it relative to temp_path
                    file_path = temp_path / Path(file_path).name
                    logger.info(f"🔍 DEBUG: Converted file_path to: {file_path}")
                elif isinstance(file_path, Path):
                    # If it's already a Path, ensure it's relative to temp_path
                    if not file_path.is_absolute():
                        file_path = temp_path / file_path.name
                        logger.info(f"🔍 DEBUG: Converted Path to: {file_path}")
                else:
                    logger.warning(f"Unexpected file_path type: {type(file_path)}")
                    return None

                # Check if file exists
                if not file_path.exists():
                    logger.warning(f"Downloaded file not found: {file_path}")
                    return None

                # Extract and parse CSV
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    # Find CSV file in zip
                    csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                    if not csv_files:
                        logger.warning(f"No CSV file found in {file_path}")
                        return None

                    csv_file = csv_files[0]
                    csv_content = zip_ref.read(csv_file).decode('utf-8')
                    df = self._parse_histdata_csv(csv_content)

                    if len(df) > 0:
                        logger.info(f"Downloaded {pair} {year} (full year): {len(df)} rows")
                        return df
                    else:
                        logger.warning(f"Empty DataFrame for {pair} {year}")
                        return None

            except Exception as e:
                logger.error(f"❌ EXCEPTION during download {pair} {year} (full year)")
                logger.error(f"❌ Exception type: {type(e).__name__}")
                logger.error(f"❌ Exception message: {str(e)}")
                import traceback
                logger.error(f"❌ Full traceback:\n{traceback.format_exc()}")
                return None
            finally:
                # Clean up temporary directory
                try:
                    import shutil
                    import time
                    time.sleep(0.1)  # Small delay to ensure all file handles are closed
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception as e:
                    logger.debug(f"Failed to clean up temp directory: {e}")

        except Exception as e:
            logger.error(f"Error downloading {pair} {year}: {e}")
            return None

    def download_pair(self, pair: str, year: int) -> Optional[pd.DataFrame]:
        """
        Download data for a specific pair and year.

        The histdata library has different requirements based on the year:
        - Past years: Must use month=None to download entire year at once
        - Current year: Must specify month to download month by month

        Args:
            pair: Currency pair (e.g., 'EURUSD')
            year: Year to download

        Returns:
            DataFrame with OHLC data or None if download fails
        """
        from datetime import datetime
        current_year = datetime.now().year

        if year < current_year:
            # For past years, histdata library requires month=None (download entire year)
            logger.info(f"Downloading {pair} {year} as full year (past year)")
            return self.download_pair_year(pair, year)
        elif year == current_year:
            # For current year, download month by month up to current month
            logger.info(f"Downloading {pair} {year} month by month (current year)")
            current_month = datetime.now().month
            dfs = []
            for month in range(1, current_month + 1):
                df = self.download_pair_month(pair, year, month)
                if df is not None and len(df) > 0:
                    dfs.append(df)

            if dfs:
                result = pd.concat(dfs, ignore_index=True)
                result = result.sort_values('timestamp').reset_index(drop=True)
                logger.info(f"Successfully downloaded {pair} {year}: {len(result)} rows ({len(dfs)} months)")
                return result
            else:
                logger.error(f"No data downloaded for {pair} {year}")
                return None
        else:
            # Future years - no data available
            logger.warning(f"Cannot download future year: {year}")
            return None
    
    def download_all_pairs(self, start_year: int = 2011, end_year: int = 2025) -> Dict[str, pd.DataFrame]:
        """
        Download all configured pairs for specified year range.
        
        Args:
            start_year: First year to download
            end_year: Last year to download (inclusive)
            
        Returns:
            Dictionary mapping pair names to DataFrames
        """
        all_data = {}
        
        for pair in tqdm(self.config.pairs, desc="Downloading pairs"):
            pair_dfs = []
            
            for year in range(start_year, end_year + 1):
                # Check if already downloaded
                cache_file = self.config.raw_data_dir / f"{pair}_{year}.parquet"
                
                if cache_file.exists():
                    logger.info(f"Loading cached {pair} {year}")
                    df = pd.read_parquet(cache_file)
                else:
                    df = self.download_pair(pair, year)
                    if df is not None:
                        # Save to cache
                        df.to_parquet(cache_file, index=False)
                
                if df is not None:
                    pair_dfs.append(df)
            
            if pair_dfs:
                combined = pd.concat(pair_dfs, ignore_index=True)
                combined = combined.sort_values('timestamp').reset_index(drop=True)
                all_data[pair] = combined
                logger.info(f"Total {pair} data: {len(combined)} rows")
            else:
                logger.warning(f"No data available for {pair}")
        
        return all_data


class DataAggregator:
    """Aggregate 1-minute bars to 5-minute bars."""
    
    @staticmethod
    def aggregate_to_5min(df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate 1-minute OHLC data to 5-minute bars.

        Standard OHLC aggregation (simplifié):
        - Open: première valeur open dans fenêtre 5min
        - High: max de tous les highs
        - Low: min de tous les lows
        - Close: dernière valeur close dans fenêtre 5min

        Args:
            df: DataFrame with 1-minute OHLC data

        Returns:
            DataFrame with 5-minute OHLC data
        """
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

        # Agrégation standard OHLC (simple et correcte)
        aggregated = df.resample('5min').agg({
            'open': 'first',   # Première valeur open
            'high': 'max',     # Max de tous les highs
            'low': 'min',      # Min de tous les lows
            'close': 'last'    # Dernière valeur close
        })

        # Remove bars with NaN values
        aggregated = aggregated.dropna()

        return aggregated.reset_index()


class DataCleaner:
    """Clean and validate OHLC data."""
    
    def __init__(self, config: DataConfig):
        self.config = config
    
    def validate_ohlc(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate OHLC relationships.
        
        Rules:
        - high >= max(open, close)
        - low <= min(open, close)
        - All prices > 0
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # Check price positivity
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            invalid_mask = (df[col] <= self.config.min_price) | (df[col] >= self.config.max_price)
            n_invalid = invalid_mask.sum()
            if n_invalid > 0:
                logger.warning(f"Found {n_invalid} invalid {col} prices, removing rows")
                df = df[~invalid_mask]
        
        # Validate OHLC relationships
        valid_high = df['high'] >= df[['open', 'close']].max(axis=1)
        valid_low = df['low'] <= df[['open', 'close']].min(axis=1)
        
        invalid_mask = ~(valid_high & valid_low)
        n_invalid = invalid_mask.sum()
        
        if n_invalid > 0:
            logger.warning(f"Found {n_invalid} bars with invalid OHLC relationships, removing")
            df = df[~invalid_mask]
        
        return df
    
    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove or clip outlier returns.
        
        Returns > 5σ are clipped to ±5σ
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # Calculate returns
        returns = df['close'].pct_change()
        
        # Calculate rolling statistics (use expanding to avoid look-ahead)
        mean = returns.expanding(min_periods=100).mean()
        std = returns.expanding(min_periods=100).std()
        
        # Identify outliers
        z_scores = np.abs((returns - mean) / (std + 1e-8))
        outlier_mask = z_scores > self.config.outlier_sigma
        
        n_outliers = outlier_mask.sum()
        if n_outliers > 0:
            logger.warning(f"Found {n_outliers} outlier returns, clipping to ±{self.config.outlier_sigma}σ")
            
            # Clip returns
            returns_clipped = returns.clip(
                lower=mean - self.config.outlier_sigma * std,
                upper=mean + self.config.outlier_sigma * std
            )
            
            # Reconstruct prices from clipped returns
            # This is tricky - we'll just remove extreme outliers instead
            df = df[~(z_scores > self.config.outlier_sigma)]
        
        return df
    
    def clean_pair(self, df: pd.DataFrame, pair: str) -> pd.DataFrame:
        """
        Apply all cleaning steps to a pair.
        
        Args:
            df: DataFrame with OHLC data
            pair: Pair name for logging
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Cleaning {pair}: {len(df)} bars")
        
        original_len = len(df)
        
        # Validate OHLC
        df = self.validate_ohlc(df)
        logger.info(f"  After OHLC validation: {len(df)} bars ({len(df)/original_len*100:.1f}%)")
        
        # Remove outliers
        df = self.remove_outliers(df)
        logger.info(f"  After outlier removal: {len(df)} bars ({len(df)/original_len*100:.1f}%)")
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp'])
        logger.info(f"  After deduplication: {len(df)} bars ({len(df)/original_len*100:.1f}%)")
        
        return df


class DataAligner:
    """Align multiple currency pairs temporally."""
    
    @staticmethod
    def align_pairs(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Align all pairs to common timestamps.
        
        Uses intersection of all timestamps to ensure all pairs have data
        at each timestamp.
        
        Args:
            data_dict: Dictionary mapping pair names to DataFrames
            
        Returns:
            Dictionary with aligned DataFrames
        """
        if not data_dict:
            return {}
        
        logger.info("Aligning pairs to common timestamps...")
        
        # Get all timestamp sets
        timestamp_sets = []
        for pair, df in data_dict.items():
            timestamps = set(df['timestamp'])
            timestamp_sets.append(timestamps)
            logger.info(f"  {pair}: {len(timestamps)} unique timestamps")
        
        # Find intersection
        common_timestamps = set.intersection(*timestamp_sets)
        logger.info(f"Common timestamps across all pairs: {len(common_timestamps)}")
        
        if len(common_timestamps) == 0:
            logger.error("No common timestamps found!")
            return {}
        
        # Filter each pair to common timestamps
        aligned_data = {}
        for pair, df in data_dict.items():
            df_aligned = df[df['timestamp'].isin(common_timestamps)].copy()
            df_aligned = df_aligned.sort_values('timestamp').reset_index(drop=True)
            aligned_data[pair] = df_aligned
            logger.info(f"  {pair} aligned: {len(df_aligned)} bars")
        
        return aligned_data


class DataSplitter:
    """Split data into train/validation/test sets."""
    
    def __init__(self, config: DataConfig):
        self.config = config
    
    def _exclude_corrupted_periods(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Exclude corrupted data periods from DataFrame.

        Args:
            df: DataFrame with timestamp column

        Returns:
            DataFrame with corrupted periods removed
        """
        if not self.config.exclude_periods:
            return df

        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        for start_date, end_date in self.config.exclude_periods:
            exclude_start = pd.to_datetime(start_date)
            exclude_end = pd.to_datetime(end_date)

            exclude_mask = (df['timestamp'] >= exclude_start) & (df['timestamp'] <= exclude_end)
            n_excluded = exclude_mask.sum()

            if n_excluded > 0:
                logger.info(f"  Excluding corrupted period {start_date} to {end_date}: {n_excluded} rows")
                df = df[~exclude_mask]

        return df

    def split_data(self, data_dict: Dict[str, pd.DataFrame]) -> Tuple[Dict, Dict, Dict]:
        """
        Split data according to configured date ranges.

        Train: 2012-01-01 → 2023-12-31
        Validation: 2019-01-01 → 2023-12-31 (intentional overlap)
        Test: 2024-01-01 → 2025-01-31 (OOS, never touched)

        Corrupted periods are automatically excluded.
        First 1000 rows of each split are removed to avoid warmup artifacts.

        Args:
            data_dict: Dictionary of aligned DataFrames

        Returns:
            Tuple of (train_dict, val_dict, test_dict)
        """
        train_data = {}
        val_data = {}
        test_data = {}

        train_start = pd.to_datetime(self.config.train_start)
        train_end = pd.to_datetime(self.config.train_end)
        val_start = pd.to_datetime(self.config.val_start)
        val_end = pd.to_datetime(self.config.val_end)
        test_start = pd.to_datetime(self.config.test_start)
        test_end = pd.to_datetime(self.config.test_end)

        for pair, df in data_dict.items():
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Exclude corrupted periods first
            df = self._exclude_corrupted_periods(df)

            # Train split - remove first 1000 rows to avoid warmup period
            train_mask = (df['timestamp'] >= train_start) & (df['timestamp'] <= train_end)
            train_df = df[train_mask].copy()
            if len(train_df) > 1000:
                train_data[pair] = train_df.iloc[1000:].reset_index(drop=True)
                logger.info(f"{pair} train: removed first 1000 rows (warmup)")
            else:
                train_data[pair] = train_df

            # Validation split - remove first 1000 rows to avoid warmup period
            val_mask = (df['timestamp'] >= val_start) & (df['timestamp'] <= val_end)
            val_df = df[val_mask].copy()
            if len(val_df) > 1000:
                val_data[pair] = val_df.iloc[1000:].reset_index(drop=True)
                logger.info(f"{pair} val: removed first 1000 rows (warmup)")
            else:
                val_data[pair] = val_df

            # Test split - remove first 1000 rows to avoid warmup period
            test_mask = (df['timestamp'] >= test_start) & (df['timestamp'] <= test_end)
            test_df = df[test_mask].copy()
            if len(test_df) > 1000:
                test_data[pair] = test_df.iloc[1000:].reset_index(drop=True)
                logger.info(f"{pair} test: removed first 1000 rows (warmup)")
            else:
                test_data[pair] = test_df

            logger.info(f"{pair} split: train={len(train_data[pair])}, "
                       f"val={len(val_data[pair])}, test={len(test_data[pair])}")

        return train_data, val_data, test_data


class DataPersistence:
    """Save and load processed data using HDF5."""
    
    def __init__(self, config: DataConfig):
        self.config = config
    
    def save_to_hdf5(
        self,
        train_data: Dict[str, pd.DataFrame],
        val_data: Dict[str, pd.DataFrame],
        test_data: Dict[str, pd.DataFrame],
        filename: str = "processed_data.h5"
    ) -> Path:
        """
        Save all processed data to HDF5 file.

        Structure:
        /train/<pair>/
        /val/<pair>/
        /test/<pair>/
        /metadata/

        IMPORTANT: Includes raw_close (non-normalized price) for precise PnL calculations

        Args:
            train_data: Training data dictionary
            val_data: Validation data dictionary
            test_data: Test data dictionary
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = self.config.processed_data_dir / filename

        logger.info(f"Saving processed data to {output_path}")

        with h5py.File(output_path, 'w') as f:
            # Save training data
            train_grp = f.create_group('train')
            for pair, df in train_data.items():
                pair_grp = train_grp.create_group(pair)
                pair_grp.create_dataset('timestamp', data=df['timestamp'].astype('int64') // 10**9)
                pair_grp.create_dataset('open', data=df['open'].values, dtype='float64')
                pair_grp.create_dataset('high', data=df['high'].values, dtype='float64')
                pair_grp.create_dataset('low', data=df['low'].values, dtype='float64')
                pair_grp.create_dataset('close', data=df['close'].values, dtype='float64')
                # Hidden columns for precise calculations (not features)
                pair_grp.create_dataset('raw_close', data=df['close'].values, dtype='float64')  # Non-normalized price

            # Save validation data
            val_grp = f.create_group('val')
            for pair, df in val_data.items():
                pair_grp = val_grp.create_group(pair)
                pair_grp.create_dataset('timestamp', data=df['timestamp'].astype('int64') // 10**9)
                pair_grp.create_dataset('open', data=df['open'].values, dtype='float64')
                pair_grp.create_dataset('high', data=df['high'].values, dtype='float64')
                pair_grp.create_dataset('low', data=df['low'].values, dtype='float64')
                pair_grp.create_dataset('close', data=df['close'].values, dtype='float64')
                # Hidden columns for precise calculations (not features)
                pair_grp.create_dataset('raw_close', data=df['close'].values, dtype='float64')  # Non-normalized price

            # Save test data
            test_grp = f.create_group('test')
            for pair, df in test_data.items():
                pair_grp = test_grp.create_group(pair)
                pair_grp.create_dataset('timestamp', data=df['timestamp'].astype('int64') // 10**9)
                pair_grp.create_dataset('open', data=df['open'].values, dtype='float64')
                pair_grp.create_dataset('high', data=df['high'].values, dtype='float64')
                pair_grp.create_dataset('low', data=df['low'].values, dtype='float64')
                pair_grp.create_dataset('close', data=df['close'].values, dtype='float64')
                # Hidden columns for precise calculations (not features)
                pair_grp.create_dataset('raw_close', data=df['close'].values, dtype='float64')  # Non-normalized price
            
            # Save metadata
            meta_grp = f.create_group('metadata')
            meta_grp.attrs['pairs'] = list(train_data.keys())
            meta_grp.attrs['train_start'] = self.config.train_start
            meta_grp.attrs['train_end'] = self.config.train_end
            meta_grp.attrs['val_start'] = self.config.val_start
            meta_grp.attrs['val_end'] = self.config.val_end
            meta_grp.attrs['test_start'] = self.config.test_start
            meta_grp.attrs['test_end'] = self.config.test_end
            meta_grp.attrs['created_at'] = datetime.now().isoformat()
        
        logger.info(f"Successfully saved data to {output_path}")
        return output_path
    
    def load_from_hdf5(self, filename: str = "processed_data.h5") -> Tuple[Dict, Dict, Dict]:
        """
        Load processed data from HDF5 file.

        IMPORTANT: Loads raw_close (non-normalized price) for precise PnL calculations

        Args:
            filename: Input filename

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        input_path = self.config.processed_data_dir / filename

        if not input_path.exists():
            raise FileNotFoundError(f"Data file not found: {input_path}")

        logger.info(f"Loading processed data from {input_path}")

        train_data = {}
        val_data = {}
        test_data = {}

        with h5py.File(input_path, 'r') as f:
            # Load training data
            for pair in f['train'].keys():
                pair_grp = f['train'][pair]
                df_dict = {
                    'timestamp': pd.to_datetime(pair_grp['timestamp'][:], unit='s'),
                    'open': pair_grp['open'][:],
                    'high': pair_grp['high'][:],
                    'low': pair_grp['low'][:],
                    'close': pair_grp['close'][:]
                }
                # Load raw_close if available (for backward compatibility)
                if 'raw_close' in pair_grp:
                    df_dict['raw_close'] = pair_grp['raw_close'][:]
                else:
                    df_dict['raw_close'] = pair_grp['close'][:]  # Fallback
                train_data[pair] = pd.DataFrame(df_dict)

            # Load validation data
            for pair in f['val'].keys():
                pair_grp = f['val'][pair]
                df_dict = {
                    'timestamp': pd.to_datetime(pair_grp['timestamp'][:], unit='s'),
                    'open': pair_grp['open'][:],
                    'high': pair_grp['high'][:],
                    'low': pair_grp['low'][:],
                    'close': pair_grp['close'][:]
                }
                # Load raw_close if available (for backward compatibility)
                if 'raw_close' in pair_grp:
                    df_dict['raw_close'] = pair_grp['raw_close'][:]
                else:
                    df_dict['raw_close'] = pair_grp['close'][:]  # Fallback
                val_data[pair] = pd.DataFrame(df_dict)

            # Load test data
            for pair in f['test'].keys():
                pair_grp = f['test'][pair]
                df_dict = {
                    'timestamp': pd.to_datetime(pair_grp['timestamp'][:], unit='s'),
                    'open': pair_grp['open'][:],
                    'high': pair_grp['high'][:],
                    'low': pair_grp['low'][:],
                    'close': pair_grp['close'][:]
                }
                # Load raw_close if available (for backward compatibility)
                if 'raw_close' in pair_grp:
                    df_dict['raw_close'] = pair_grp['raw_close'][:]
                else:
                    df_dict['raw_close'] = pair_grp['close'][:]  # Fallback
                test_data[pair] = pd.DataFrame(df_dict)

            # Log metadata
            meta = f['metadata'].attrs
            logger.info(f"Data created at: {meta['created_at']}")
            logger.info(f"Pairs: {list(meta['pairs'])}")

        logger.info("Successfully loaded data")
        return train_data, val_data, test_data


class DataPipeline:
    """Main data pipeline orchestrator."""
    
    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()
        self.downloader = DataDownloader(self.config)
        self.aggregator = DataAggregator()
        self.cleaner = DataCleaner(self.config)
        self.aligner = DataAligner()
        self.splitter = DataSplitter(self.config)
        self.persistence = DataPersistence(self.config)
    
    def run_full_pipeline(self, force_download: bool = False) -> Tuple[Dict, Dict, Dict]:
        """
        Run the complete data pipeline.
        
        Steps:
        1. Download data (or load from cache)
        2. Aggregate to 5-minute bars
        3. Clean and validate
        4. Align temporally
        5. Split into train/val/test
        6. Save to HDF5
        
        Args:
            force_download: If True, re-download even if cache exists
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        logger.info("="*80)
        logger.info("Starting SAC EUR/USD Data Pipeline")
        logger.info("="*80)
        
        # Check if processed data already exists
        processed_file = self.config.processed_data_dir / "processed_data.h5"
        if processed_file.exists() and not force_download:
            logger.info("Processed data file found, loading from cache...")
            return self.persistence.load_from_hdf5()
        
        # Step 1: Download data
        logger.info("\n[1/6] Downloading data...")
        raw_data = self.downloader.download_all_pairs(
            start_year=2011,
            end_year=2025
        )
        
        if not raw_data:
            raise ValueError("No data downloaded!")
        
        # Step 2: Aggregate to 5-minute bars
        logger.info("\n[2/6] Aggregating to 5-minute bars...")
        aggregated_data = {}
        for pair, df in tqdm(raw_data.items(), desc="Aggregating"):
            aggregated_data[pair] = self.aggregator.aggregate_to_5min(df)
            logger.info(f"  {pair}: {len(df)} → {len(aggregated_data[pair])} bars")
        
        # Step 3: Clean and validate
        logger.info("\n[3/6] Cleaning and validating data...")
        cleaned_data = {}
        for pair, df in tqdm(aggregated_data.items(), desc="Cleaning"):
            cleaned_data[pair] = self.cleaner.clean_pair(df, pair)
        
        # Step 4: Align temporally
        logger.info("\n[4/6] Aligning pairs temporally...")
        aligned_data = self.aligner.align_pairs(cleaned_data)
        
        # Step 5: Split data
        logger.info("\n[5/6] Splitting into train/val/test...")
        train_data, val_data, test_data = self.splitter.split_data(aligned_data)
        
        # Step 6: Save to HDF5
        logger.info("\n[6/6] Saving to HDF5...")
        self.persistence.save_to_hdf5(train_data, val_data, test_data)
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("Data Pipeline Complete!")
        logger.info("="*80)
        logger.info(f"Training samples: {len(next(iter(train_data.values())))}")
        logger.info(f"Validation samples: {len(next(iter(val_data.values())))}")
        logger.info(f"Test samples: {len(next(iter(test_data.values())))}")
        logger.info(f"Total pairs: {len(train_data)}")
        logger.info("="*80)
        
        return train_data, val_data, test_data
    
    def get_processed_data(self) -> Tuple[Dict, Dict, Dict]:
        """
        Get processed data (load from cache if exists, otherwise run pipeline).
        
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        processed_file = self.config.processed_data_dir / "processed_data.h5"
        
        if processed_file.exists():
            logger.info("Loading processed data from cache...")
            return self.persistence.load_from_hdf5()
        else:
            logger.info("No cached data found, running full pipeline...")
            return self.run_full_pipeline()


def main():
    """Example usage of the data pipeline."""
    
    # Create configuration
    config = DataConfig()
    
    # Initialize pipeline
    pipeline = DataPipeline(config)
    
    # Run pipeline
    train_data, val_data, test_data = pipeline.run_full_pipeline(force_download=False)
    
    # Print some statistics
    print("\n" + "="*80)
    print("Data Statistics")
    print("="*80)
    
    for pair in train_data.keys():
        train_df = train_data[pair]
        print(f"\n{pair}:")
        print(f"  Training period: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
        print(f"  Training bars: {len(train_df)}")
        print(f"  Price range: {train_df['close'].min():.5f} - {train_df['close'].max():.5f}")
        print(f"  Mean price: {train_df['close'].mean():.5f}")
        print(f"  Volatility (std): {train_df['close'].std():.5f}")


if __name__ == "__main__":
    main()
