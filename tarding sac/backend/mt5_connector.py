"""
SAC EUR/USD Trading System - MetaTrader 5 Connector
===================================================

This module handles connection and trading operations with MetaTrader 5 platform.

Features:
- MT5 connection management with automatic reconnection
- Real-time price data retrieval (bid/ask)
- Account information monitoring (balance, equity, margin)
- Order execution (market, limit, stop orders)
- Position management (open, close, modify)
- Stop-Loss and Take-Profit management
- Retry logic for failed operations
- Comprehensive logging of all operations
- Market status checking

Author: SAC EUR/USD Project
Version: 2.0
"""

import logging
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MT5Config:
    """Configuration for MT5 connection."""
    
    # Connection parameters
    account: int = 0  # MT5 account number
    password: str = ""  # MT5 password
    server: str = ""  # Broker server name
    
    # Trading parameters
    symbol: str = "EURUSD"
    magic_number: int = 234000  # Unique identifier for this EA
    deviation: int = 20  # Maximum price deviation in points
    
    # Retry parameters
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds
    
    # Reconnection parameters
    reconnect_attempts: int = 5
    reconnect_delay: float = 5.0  # seconds
    
    # Logging
    log_dir: Path = Path("logs/mt5")
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        self.log_dir.mkdir(parents=True, exist_ok=True)


class MT5Error(Exception):
    """Custom exception for MT5 errors."""
    pass


class MT5Connector:
    """
    MetaTrader 5 connection manager.
    
    Handles connection, reconnection, and basic MT5 operations.
    """
    
    def __init__(self, config: MT5Config):
        self.config = config
        self.connected = False
        self.symbol_info = None
        
        logger.info("MT5 Connector initialized")
    
    def connect(self) -> bool:
        """
        Connect to MetaTrader 5.
        
        Returns:
            True if connection successful, False otherwise
        """
        logger.info("Attempting to connect to MT5...")
        
        # Initialize MT5
        if not mt5.initialize():
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False
        
        logger.info("MT5 initialized successfully")
        
        # Login if credentials provided
        if self.config.account > 0:
            logger.info(f"Logging in to account {self.config.account}...")
            
            authorized = mt5.login(
                login=self.config.account,
                password=self.config.password,
                server=self.config.server
            )
            
            if not authorized:
                logger.error(f"Login failed: {mt5.last_error()}")
                mt5.shutdown()
                return False
            
            logger.info(f"Logged in successfully to account {self.config.account}")
        
        # Get symbol info
        self.symbol_info = mt5.symbol_info(self.config.symbol)
        
        if self.symbol_info is None:
            logger.error(f"Symbol {self.config.symbol} not found")
            mt5.shutdown()
            return False
        
        # Enable symbol if needed
        if not self.symbol_info.visible:
            logger.info(f"Enabling symbol {self.config.symbol}...")
            if not mt5.symbol_select(self.config.symbol, True):
                logger.error(f"Failed to enable symbol {self.config.symbol}")
                mt5.shutdown()
                return False
        
        self.connected = True
        logger.info("MT5 connection established successfully")
        
        # Log connection info
        account_info = self.get_account_info()
        if account_info:
            logger.info(f"Account Balance: {account_info['balance']:.2f}")
            logger.info(f"Account Equity: {account_info['equity']:.2f}")
            logger.info(f"Account Leverage: 1:{account_info['leverage']}")
        
        return True
    
    def disconnect(self):
        """Disconnect from MT5."""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("Disconnected from MT5")
    
    def reconnect(self) -> bool:
        """
        Attempt to reconnect to MT5.
        
        Returns:
            True if reconnection successful, False otherwise
        """
        logger.warning("Attempting to reconnect to MT5...")
        
        for attempt in range(self.config.reconnect_attempts):
            logger.info(f"Reconnection attempt {attempt + 1}/{self.config.reconnect_attempts}")
            
            # Disconnect first
            self.disconnect()
            
            # Wait before reconnecting
            time.sleep(self.config.reconnect_delay)
            
            # Try to connect
            if self.connect():
                logger.info("Reconnection successful!")
                return True
        
        logger.error("Failed to reconnect after all attempts")
        return False
    
    def ensure_connection(self) -> bool:
        """
        Ensure connection is active, reconnect if necessary.
        
        Returns:
            True if connected, False otherwise
        """
        if not self.connected:
            return self.reconnect()
        
        # Test connection
        try:
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                logger.warning("Terminal info is None, connection lost")
                return self.reconnect()
            return True
        except Exception as e:
            logger.error(f"Error checking connection: {e}")
            return self.reconnect()
    
    def get_account_info(self) -> Optional[Dict]:
        """
        Get account information.
        
        Returns:
            Dictionary with account info or None if failed
        """
        if not self.ensure_connection():
            return None
        
        try:
            account_info = mt5.account_info()
            
            if account_info is None:
                logger.error(f"Failed to get account info: {mt5.last_error()}")
                return None
            
            return {
                'balance': account_info.balance,
                'equity': account_info.equity,
                'margin': account_info.margin,
                'free_margin': account_info.margin_free,
                'margin_level': account_info.margin_level,
                'profit': account_info.profit,
                'leverage': account_info.leverage,
                'currency': account_info.currency
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None
    
    def get_current_price(self) -> Optional[Dict]:
        """
        Get current bid/ask prices.
        
        Returns:
            Dictionary with price info or None if failed
        """
        if not self.ensure_connection():
            return None
        
        try:
            tick = mt5.symbol_info_tick(self.config.symbol)
            
            if tick is None:
                logger.error(f"Failed to get tick: {mt5.last_error()}")
                return None
            
            return {
                'bid': tick.bid,
                'ask': tick.ask,
                'spread': tick.ask - tick.bid,
                'time': datetime.fromtimestamp(tick.time),
                'volume': tick.volume
            }
        except Exception as e:
            logger.error(f"Error getting current price: {e}")
            return None
    
    def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Get open positions.
        
        Args:
            symbol: Filter by symbol (None = all positions)
            
        Returns:
            List of position dictionaries
        """
        if not self.ensure_connection():
            return []
        
        try:
            if symbol:
                positions = mt5.positions_get(symbol=symbol)
            else:
                positions = mt5.positions_get()
            
            if positions is None:
                logger.error(f"Failed to get positions: {mt5.last_error()}")
                return []
            
            position_list = []
            for pos in positions:
                position_list.append({
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'type': 'buy' if pos.type == mt5.ORDER_TYPE_BUY else 'sell',
                    'volume': pos.volume,
                    'price_open': pos.price_open,
                    'price_current': pos.price_current,
                    'sl': pos.sl,
                    'tp': pos.tp,
                    'profit': pos.profit,
                    'time': datetime.fromtimestamp(pos.time),
                    'comment': pos.comment,
                    'magic': pos.magic
                })
            
            return position_list
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def get_bars(
        self,
        timeframe: int = mt5.TIMEFRAME_M5,
        count: int = 100
    ) -> Optional[pd.DataFrame]:
        """
        Get historical bars.
        
        Args:
            timeframe: MT5 timeframe constant
            count: Number of bars
            
        Returns:
            DataFrame with OHLC data or None if failed
        """
        if not self.ensure_connection():
            return None
        
        try:
            rates = mt5.copy_rates_from_pos(
                self.config.symbol,
                timeframe,
                0,
                count
            )
            
            if rates is None or len(rates) == 0:
                logger.error(f"Failed to get bars: {mt5.last_error()}")
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            return df
        except Exception as e:
            logger.error(f"Error getting bars: {e}")
            return None
    
    def is_market_open(self) -> bool:
        """
        Check if market is open for trading.
        
        Returns:
            True if market is open, False otherwise
        """
        if not self.ensure_connection():
            return False
        
        try:
            symbol_info = mt5.symbol_info(self.config.symbol)
            
            if symbol_info is None:
                return False
            
            # Check trading mode
            if symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
                return False
            
            # Check if market is open (session)
            tick = mt5.symbol_info_tick(self.config.symbol)
            if tick is None:
                return False
            
            # Check if price is changing
            current_time = datetime.now()
            tick_time = datetime.fromtimestamp(tick.time)
            time_diff = (current_time - tick_time).total_seconds()
            
            # If last tick is older than 1 minute, market might be closed
            if time_diff > 60:
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            return False


class OrderManager:
    """
    Order manager for executing and managing trades.
    
    Converts trading signals to MT5 orders with proper position sizing,
    SL/TP, and error handling.
    """
    
    def __init__(self, connector: MT5Connector, config: MT5Config):
        self.connector = connector
        self.config = config
        
        logger.info("Order Manager initialized")
    
    def calculate_position_size(
        self,
        equity: float,
        risk_per_trade: float,
        sl_distance_pips: float,
        max_lots: float = 100.0,
        min_lots: float = 0.01
    ) -> float:
        """
        Calculate position size based on risk management.
        
        Args:
            equity: Account equity
            risk_per_trade: Risk as fraction of equity (e.g., 0.02 for 2%)
            sl_distance_pips: Stop-loss distance in pips
            max_lots: Maximum position size
            min_lots: Minimum position size
            
        Returns:
            Position size in lots
        """
        # Risk amount in account currency
        risk_amount = equity * risk_per_trade
        
        # Get symbol info
        symbol_info = mt5.symbol_info(self.config.symbol)
        if symbol_info is None:
            logger.error("Failed to get symbol info for position sizing")
            return min_lots
        
        # Calculate pip value
        # For EUR/USD: 1 pip = 0.0001
        # 1 lot = 100,000 EUR
        # Pip value = 1 lot × 0.0001 = 10 USD per pip
        point = symbol_info.point
        pip_value = symbol_info.trade_contract_size * point * 10  # 10 points = 1 pip for 5-digit quotes
        
        # Calculate position size
        if sl_distance_pips > 0:
            position_size = risk_amount / (sl_distance_pips * pip_value)
        else:
            position_size = min_lots
        
        # Apply constraints
        position_size = max(min_lots, min(position_size, max_lots))
        
        # Round to symbol's volume step
        volume_step = symbol_info.volume_step
        position_size = round(position_size / volume_step) * volume_step
        
        logger.info(f"Calculated position size: {position_size:.2f} lots "
                   f"(Risk: {risk_per_trade*100:.1f}%, SL: {sl_distance_pips:.1f} pips)")
        
        return position_size
    
    def open_position(
        self,
        action: float,
        volume: float,
        sl_price: Optional[float] = None,
        tp_price: Optional[float] = None,
        comment: str = "SAC_EUR/USD"
    ) -> Optional[Dict]:
        """
        Open a position.
        
        Args:
            action: Action value (-1 to 1, negative = sell, positive = buy)
            volume: Position size in lots
            sl_price: Stop-loss price (None = no SL)
            tp_price: Take-profit price (None = no TP)
            comment: Order comment
            
        Returns:
            Dictionary with order result or None if failed
        """
        if not self.connector.ensure_connection():
            logger.error("Cannot open position: not connected")
            return None
        
        # Determine order type
        if action > 0:
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(self.config.symbol).ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(self.config.symbol).bid
        
        # Prepare request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.config.symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "deviation": self.config.deviation,
            "magic": self.config.magic_number,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Add SL/TP if provided
        if sl_price is not None:
            request["sl"] = sl_price
        if tp_price is not None:
            request["tp"] = tp_price
        
        # Execute with retry logic
        for attempt in range(self.config.max_retries):
            try:
                logger.info(f"Opening position (attempt {attempt + 1}/{self.config.max_retries})...")
                logger.info(f"  Type: {'BUY' if order_type == mt5.ORDER_TYPE_BUY else 'SELL'}")
                logger.info(f"  Volume: {volume:.2f} lots")
                logger.info(f"  Price: {price:.5f}")
                if sl_price:
                    logger.info(f"  Stop-Loss: {sl_price:.5f}")
                if tp_price:
                    logger.info(f"  Take-Profit: {tp_price:.5f}")
                
                result = mt5.order_send(request)
                
                if result is None:
                    logger.error(f"Order send failed: {mt5.last_error()}")
                    time.sleep(self.config.retry_delay)
                    continue
                
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    logger.error(f"Order failed with retcode {result.retcode}: {result.comment}")
                    time.sleep(self.config.retry_delay)
                    continue
                
                # Success
                logger.info(f"Position opened successfully! Ticket: {result.order}")
                
                return {
                    'ticket': result.order,
                    'volume': result.volume,
                    'price': result.price,
                    'bid': result.bid,
                    'ask': result.ask,
                    'comment': result.comment,
                    'retcode': result.retcode
                }
            
            except Exception as e:
                logger.error(f"Exception opening position: {e}")
                time.sleep(self.config.retry_delay)
        
        logger.error("Failed to open position after all retries")
        return None
    
    def close_position(
        self,
        ticket: int,
        volume: Optional[float] = None,
        comment: str = "Close by SAC"
    ) -> bool:
        """
        Close a position.
        
        Args:
            ticket: Position ticket
            volume: Volume to close (None = close all)
            comment: Close comment
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connector.ensure_connection():
            logger.error("Cannot close position: not connected")
            return False
        
        # Get position info
        position = mt5.positions_get(ticket=ticket)
        
        if position is None or len(position) == 0:
            logger.error(f"Position {ticket} not found")
            return False
        
        position = position[0]
        
        # Determine close parameters
        if volume is None:
            volume = position.volume
        
        # Determine close type (opposite of opening)
        if position.type == mt5.ORDER_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(self.config.symbol).bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(self.config.symbol).ask
        
        # Prepare close request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.config.symbol,
            "volume": volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": self.config.deviation,
            "magic": self.config.magic_number,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Execute with retry logic
        for attempt in range(self.config.max_retries):
            try:
                logger.info(f"Closing position {ticket} (attempt {attempt + 1}/{self.config.max_retries})...")
                logger.info(f"  Volume: {volume:.2f} lots")
                logger.info(f"  Price: {price:.5f}")
                
                result = mt5.order_send(request)
                
                if result is None:
                    logger.error(f"Close order send failed: {mt5.last_error()}")
                    time.sleep(self.config.retry_delay)
                    continue
                
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    logger.error(f"Close failed with retcode {result.retcode}: {result.comment}")
                    time.sleep(self.config.retry_delay)
                    continue
                
                # Success
                logger.info(f"Position {ticket} closed successfully!")
                return True
            
            except Exception as e:
                logger.error(f"Exception closing position: {e}")
                time.sleep(self.config.retry_delay)
        
        logger.error(f"Failed to close position {ticket} after all retries")
        return False
    
    def modify_position(
        self,
        ticket: int,
        sl_price: Optional[float] = None,
        tp_price: Optional[float] = None
    ) -> bool:
        """
        Modify position's SL/TP.
        
        Args:
            ticket: Position ticket
            sl_price: New stop-loss price (None = no change)
            tp_price: New take-profit price (None = no change)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connector.ensure_connection():
            logger.error("Cannot modify position: not connected")
            return False
        
        # Get position info
        position = mt5.positions_get(ticket=ticket)
        
        if position is None or len(position) == 0:
            logger.error(f"Position {ticket} not found")
            return False
        
        position = position[0]
        
        # Prepare modify request
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": self.config.symbol,
            "position": ticket,
            "sl": sl_price if sl_price is not None else position.sl,
            "tp": tp_price if tp_price is not None else position.tp,
            "magic": self.config.magic_number
        }
        
        # Execute with retry logic
        for attempt in range(self.config.max_retries):
            try:
                logger.info(f"Modifying position {ticket} (attempt {attempt + 1}/{self.config.max_retries})...")
                if sl_price is not None:
                    logger.info(f"  New SL: {sl_price:.5f}")
                if tp_price is not None:
                    logger.info(f"  New TP: {tp_price:.5f}")
                
                result = mt5.order_send(request)
                
                if result is None:
                    logger.error(f"Modify order send failed: {mt5.last_error()}")
                    time.sleep(self.config.retry_delay)
                    continue
                
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    logger.error(f"Modify failed with retcode {result.retcode}: {result.comment}")
                    time.sleep(self.config.retry_delay)
                    continue
                
                # Success
                logger.info(f"Position {ticket} modified successfully!")
                return True
            
            except Exception as e:
                logger.error(f"Exception modifying position: {e}")
                time.sleep(self.config.retry_delay)
        
        logger.error(f"Failed to modify position {ticket} after all retries")
        return False
    
    def close_all_positions(self, symbol: Optional[str] = None) -> int:
        """
        Close all open positions.
        
        Args:
            symbol: Close positions for specific symbol (None = all symbols)
            
        Returns:
            Number of positions closed
        """
        logger.info("Closing all positions...")
        
        positions = self.connector.get_positions(symbol=symbol)
        closed_count = 0
        
        for position in positions:
            # Only close positions with our magic number
            if position['magic'] == self.config.magic_number:
                if self.close_position(position['ticket']):
                    closed_count += 1
        
        logger.info(f"Closed {closed_count}/{len(positions)} positions")
        return closed_count
    
    def execute_signal(
        self,
        action: float,
        current_price: float,
        atr: float,
        equity: float,
        risk_per_trade: float = 0.02,
        sl_atr_mult: float = 2.0,
        tp_atr_mult: float = 4.0
    ) -> Optional[Dict]:
        """
        Execute trading signal with full risk management.
        
        Args:
            action: Action value from agent (-1 to 1)
            current_price: Current market price
            atr: Current ATR value
            equity: Account equity
            risk_per_trade: Risk as fraction of equity
            sl_atr_mult: Stop-loss ATR multiplier
            tp_atr_mult: Take-profit ATR multiplier
            
        Returns:
            Dictionary with execution result or None if failed
        """
        logger.info("="*80)
        logger.info("Executing Trading Signal")
        logger.info("="*80)
        logger.info(f"Action: {action:.4f}")
        logger.info(f"Current Price: {current_price:.5f}")
        logger.info(f"ATR: {atr:.5f}")
        logger.info(f"Equity: {equity:.2f}")
        
        # Calculate position parameters
        sl_distance = sl_atr_mult * atr
        
        # Convert to pips (for 5-digit quotes, 1 pip = 10 points)
        symbol_info = mt5.symbol_info(self.config.symbol)
        if symbol_info is None:
            logger.error("Failed to get symbol info")
            return None
        
        point = symbol_info.point
        sl_distance_pips = sl_distance / (point * 10)
        
        # Calculate position size
        volume = self.calculate_position_size(
            equity=equity,
            risk_per_trade=risk_per_trade,
            sl_distance_pips=sl_distance_pips
        )
        
        # Calculate SL/TP prices
        if action > 0:  # Buy
            sl_price = current_price - sl_distance
            tp_price = current_price + (tp_atr_mult * atr)
        else:  # Sell
            sl_price = current_price + sl_distance
            tp_price = current_price - (tp_atr_mult * atr)
        
        logger.info(f"Position Size: {volume:.2f} lots")
        logger.info(f"Stop-Loss: {sl_price:.5f}")
        logger.info(f"Take-Profit: {tp_price:.5f}")
        
        # Execute order
        result = self.open_position(
            action=action,
            volume=volume,
            sl_price=sl_price,
            tp_price=tp_price,
            comment="SAC_EUR/USD"
        )
        
        if result:
            logger.info("Signal executed successfully!")
            logger.info("="*80)
        else:
            logger.error("Signal execution failed!")
            logger.info("="*80)
        
        return result


class TradeLogger:
    """Logger for all trading activities."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create trade log file
        self.trade_log_file = self.log_dir / f"trades_{datetime.now().strftime('%Y%m%d')}.csv"
        
        # Initialize trade log if it doesn't exist
        if not self.trade_log_file.exists():
            with open(self.trade_log_file, 'w') as f:
                f.write("timestamp,ticket,action,volume,price,sl,tp,result,comment\n")
    
    def log_trade(
        self,
        ticket: int,
        action: str,
        volume: float,
        price: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        result: str = "executed",
        comment: str = ""
    ):
        """Log a trade to CSV file."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(self.trade_log_file, 'a') as f:
            f.write(f"{timestamp},{ticket},{action},{volume:.2f},{price:.5f},"
                   f"{sl:.5f if sl else 'None'},{tp:.5f if tp else 'None'},"
                   f"{result},{comment}\n")
        
        logger.info(f"Trade logged: {action} {volume:.2f} @ {price:.5f}")
    
    def log_close(
        self,
        ticket: int,
        close_price: float,
        profit: float,
        comment: str = ""
    ):
        """Log a position close."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(self.trade_log_file, 'a') as f:
            f.write(f"{timestamp},{ticket},CLOSE,0,{close_price:.5f},"
                   f"None,None,profit={profit:.2f},{comment}\n")
        
        logger.info(f"Position {ticket} closed: Profit = {profit:.2f}")


def main():
    """Example usage of MT5 connector."""
    
    # Create configuration
    # IMPORTANT: Fill in your MT5 credentials
    config = MT5Config(
        account=0,  # Your MT5 account number
        password="",  # Your MT5 password
        server="",  # Your broker server
        symbol="EURUSD",
        magic_number=234000
    )
    
    # Create connector
    connector = MT5Connector(config)
    
    # Connect
    if not connector.connect():
        logger.error("Failed to connect to MT5")
        return
    
    try:
        # Get account info
        logger.info("\n" + "="*80)
        logger.info("Account Information")
        logger.info("="*80)
        
        account_info = connector.get_account_info()
        if account_info:
            for key, value in account_info.items():
                logger.info(f"{key}: {value}")
        
        # Get current price
        logger.info("\n" + "="*80)
        logger.info("Current Price")
        logger.info("="*80)
        
        price_info = connector.get_current_price()
        if price_info:
            logger.info(f"Bid: {price_info['bid']:.5f}")
            logger.info(f"Ask: {price_info['ask']:.5f}")
            logger.info(f"Spread: {price_info['spread']:.5f}")
        
        # Check market status
        logger.info("\n" + "="*80)
        logger.info("Market Status")
        logger.info("="*80)
        
        is_open = connector.is_market_open()
        logger.info(f"Market Open: {is_open}")
        
        # Get open positions
        logger.info("\n" + "="*80)
        logger.info("Open Positions")
        logger.info("="*80)
        
        positions = connector.get_positions(symbol=config.symbol)
        logger.info(f"Number of open positions: {len(positions)}")
        
        for pos in positions:
            logger.info(f"\nPosition {pos['ticket']}:")
            logger.info(f"  Type: {pos['type']}")
            logger.info(f"  Volume: {pos['volume']:.2f}")
            logger.info(f"  Price: {pos['price_open']:.5f}")
            logger.info(f"  Current: {pos['price_current']:.5f}")
            logger.info(f"  Profit: {pos['profit']:.2f}")
        
        # Get historical bars
        logger.info("\n" + "="*80)
        logger.info("Historical Bars (Last 10)")
        logger.info("="*80)
        
        bars = connector.get_bars(timeframe=mt5.TIMEFRAME_M5, count=10)
        if bars is not None:
            logger.info(f"\n{bars[['time', 'open', 'high', 'low', 'close']].to_string()}")
        
        # Example: Create order manager and execute signal
        logger.info("\n" + "="*80)
        logger.info("Order Manager Demo (NOT EXECUTING)")
        logger.info("="*80)
        
        order_manager = OrderManager(connector, config)
        
        # Simulate signal execution (not actually executing)
        if account_info and price_info:
            logger.info("Simulating signal execution parameters:")
            logger.info(f"  Action: 0.5 (buy signal)")
            logger.info(f"  Current Price: {price_info['bid']:.5f}")
            logger.info(f"  ATR: 0.0010 (example)")
            logger.info(f"  Equity: {account_info['equity']:.2f}")
            
            # Calculate what would happen (not executing)
            volume = order_manager.calculate_position_size(
                equity=account_info['equity'],
                risk_per_trade=0.02,
                sl_distance_pips=20.0
            )
            logger.info(f"  Calculated Volume: {volume:.2f} lots")
        
    finally:
        # Disconnect
        connector.disconnect()
        logger.info("\n" + "="*80)
        logger.info("MT5 Connector Demo Complete")
        logger.info("="*80)


if __name__ == "__main__":
    main()
