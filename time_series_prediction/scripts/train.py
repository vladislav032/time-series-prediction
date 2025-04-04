import os
import time
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from colorama import Fore, init, Style

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error

from time_series_prediction.common.column_parser import ColumnParser
from time_series_prediction.seq2seq.attention_model import ImprovedModel

init(autoreset=True)

# ==================== PATTERNS IMPLEMENTATION ====================

class INormalizationStrategy(ABC):
    """Strategy interface for normalization techniques."""
    
    @abstractmethod
    def normalize(self, data: np.ndarray) -> Tuple[np.ndarray, Any, Any]:
        pass
    
    @abstractmethod
    def denormalize(self, data: np.ndarray, min_val: Any, max_val: Any) -> np.ndarray:
        pass


class MinMaxNormalization(INormalizationStrategy):
    """Concrete strategy for Min-Max normalization."""
    
    def normalize(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        normalized = 2 * (data - min_val) / (max_val - min_val) - 1
        return normalized, min_val, max_val
    
    def denormalize(self, data: np.ndarray, min_val: np.ndarray, max_val: np.ndarray) -> np.ndarray:
        return (data + 1) * (max_val - min_val) / 2 + min_val


class DataLoader:
    """Responsible for loading and preparing data."""
    
    def __init__(self, normalization_strategy: INormalizationStrategy):
        self.normalization_strategy = normalization_strategy
    
    def load_and_prepare_data(self, df: pd.DataFrame, column_names: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load data and apply normalization."""
        df = df[column_names].dropna()
        data = df.values
        normalized_data, min_val, max_val = self.normalization_strategy.normalize(data)
        return data, normalized_data, min_val, max_val


class IModelFactory(ABC):
    """Abstract factory for creating models."""
    
    @abstractmethod
    def create_model(self, input_size: int, output_size: int) -> nn.Module:
        pass


class ImprovedModelFactory(IModelFactory):
    """Concrete factory for creating ImprovedModel instances."""
    
    def create_model(self, input_size: int, output_size: int) -> nn.Module:
        return ImprovedModel(input_size, output_size)


class ILogger(ABC):
    """Abstract observer for logging."""
    
    @abstractmethod
    def log(self, message: str, color: str = Fore.GREEN) -> None:
        pass


class ConsoleLogger(ILogger):
    """Concrete logger that outputs to console."""
    
    def log(self, message: str, color: str = Fore.GREEN) -> None:
        print(color + message + Style.RESET_ALL)


class IModelTrainer(ABC):
    """Abstract strategy for model training."""
    
    @abstractmethod
    def train_model(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        target_accuracy: float,
        max_epochs: int,
        device: torch.device
    ) -> None:
        pass


class DefaultModelTrainer(IModelTrainer):
    """Concrete strategy for default model training."""
    
    def __init__(self, logger: ILogger):
        self.logger = logger
    
    def train_model(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        target_accuracy: float,
        max_epochs: int,
        device: torch.device
    ) -> None:
        """Train the model with progress tracking."""
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        with tqdm(total=target_accuracy, desc="Training Accuracy", ncols=70, leave=True, unit='%', unit_scale=True) as pbar:
            for epoch in range(max_epochs):
                optimizer.zero_grad()
                output = model(X.to(device))
                loss = criterion(output, y.to(device))
                loss.backward()
                optimizer.step()
                
                mse = mean_squared_error(y.cpu().detach().numpy(), output.cpu().detach().numpy())
                accuracy = 1 - np.sqrt(mse)
                
                pbar.n = accuracy * 100
                pbar.refresh()
                
                if accuracy >= target_accuracy:
                    self.logger.log(f"Reached target accuracy: {accuracy:.2%}")
                    break


class ModelRepository:
    """Repository pattern for model persistence."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def save_model(self, model: nn.Module, model_id: int) -> None:
        """Save model state to disk."""
        torch.save(model.state_dict(), os.path.join(self.output_dir, f'model_{model_id}.pth'))
    
    def load_model(self, model: nn.Module, model_id: int) -> nn.Module:
        """Load model state from disk."""
        model.load_state_dict(torch.load(os.path.join(self.output_dir, f'model_{model_id}.pth')))
        return model


# ==================== MAIN CLASS ====================

class TimeSeriesModelTrainer:
    def __init__(
        self,
        df: pd.DataFrame,
        column_names: List[str],
        window_size: int,
        n_models: int,
        output_dir: str = 'models',
        normalization_strategy: INormalizationStrategy = MinMaxNormalization(),
        model_factory: IModelFactory = ImprovedModelFactory(),
        logger: ILogger = ConsoleLogger(),
        model_trainer: IModelTrainer = None
    ):
        """
        Initialize the TimeSeriesModelTrainer with configurable components.
        
        Args:
            df: Input DataFrame containing the time series data
            column_names: List of column names to use from the DataFrame
            window_size: Size of the sliding window for time series
            n_models: Number of models to train
            output_dir: Directory to save trained models
            normalization_strategy: Strategy for data normalization
            model_factory: Factory for creating models
            logger: Logger for progress reporting
            model_trainer: Strategy for model training
        """
        self.df = df
        self.column_names = self._parse_columns(column_names)
        self.window_size = window_size
        self.n_models = n_models
        self.output_dir = output_dir
        
        # Dependency injection
        self.normalization_strategy = normalization_strategy
        self.model_factory = model_factory
        self.logger = logger
        self.model_trainer = model_trainer or DefaultModelTrainer(logger)
        self.model_repository = ModelRepository(output_dir)
        
        # Data attributes
        self.time_series: Optional[np.ndarray] = None
        self.normalized_time_series: Optional[np.ndarray] = None
        self.min_val: Optional[np.ndarray] = None
        self.max_val: Optional[np.ndarray] = None
        self.X: Optional[torch.Tensor] = None
        self.y: Optional[torch.Tensor] = None
        
        self._initialize_data()
    
    def _parse_columns(self, columns: List[str]) -> List[str]:
        """Parse and validate column names."""
        parser = ColumnParser(self.df)
        return parser.parse_columns(columns)
    
    def _initialize_data(self) -> None:
        """Load and prepare the initial dataset."""
        data_loader = DataLoader(self.normalization_strategy)
        self.time_series, self.normalized_time_series, self.min_val, self.max_val = \
            data_loader.load_and_prepare_data(self.df, self.column_names)
        self._create_datasets()
    
    def _create_datasets(self) -> None:
        """Create training datasets from time series data."""
        X, y = [], []
        n = self.time_series.shape[0]
        
        for i in range(n - self.window_size):
            X.append(self.normalized_time_series[i:i + self.window_size])
            y.append(self.normalized_time_series[i + self.window_size])
        
        self.X = torch.tensor(np.array(X), dtype=torch.float32)
        self.y = torch.tensor(np.array(y), dtype=torch.float32)
    
    def _build_model(self) -> nn.Module:
        """Create a new model instance."""
        input_size = len(self.column_names)
        output_size = len(self.column_names)
        return self.model_factory.create_model(input_size, output_size)
    
    def train_and_save_models(self, target_accuracy: float = 0.80, max_epochs: int = 1000) -> None:
        """Train multiple models and save them to disk."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.log(f'Using device: {device}')
        
        for i in range(self.n_models):
            model = self._build_model().to(device)
            
            start_time = time.time()
            self.logger.log(f'Training model {i+1}')
            
            self.model_trainer.train_model(
                model=model,
                X=self.X,
                y=self.y,
                target_accuracy=target_accuracy,
                max_epochs=max_epochs,
                device=device
            )
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            self.model_repository.save_model(model, i+1)
            self.logger.log(f'Model {i+1} saved in {elapsed_time:.2f} s', Fore.YELLOW)
    
    def continue_training(self, new_df: pd.DataFrame, target_accuracy: float = 0.80, max_epochs: int = 1000) -> None:
        """
        Continue training with additional data from a new DataFrame.
        
        Args:
            new_df: New DataFrame containing additional time series data
            target_accuracy: Target accuracy to reach
            max_epochs: Maximum number of training epochs
        """
        # Normalize new data using existing parameters
        new_time_series = new_df[self.column_names].values
        normalized_new_time_series = self.normalization_strategy.normalize(
            new_time_series, self.min_val, self.max_val
        )
        
        # Combine with existing data
        self.normalized_time_series = np.concatenate(
            (self.normalized_time_series, normalized_new_time_series),
            axis=0
        )
        
        # Recreate datasets
        self._create_datasets()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.log(f'Using device: {device}')
        
        for i in range(self.n_models):
            model = self._build_model().to(device)
            model = self.model_repository.load_model(model, i+1)
            
            start_time = time.time()
            self.logger.log(f'Continuing training for model {i+1}')
            
            self.model_trainer.train_model(
                model=model,
                X=self.X,
                y=self.y,
                target_accuracy=target_accuracy,
                max_epochs=max_epochs,
                device=device
            )
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            self.model_repository.save_model(model, i+1)
            self.logger.log(f'Model {i+1} re-saved in {elapsed_time:.2f} s', Fore.YELLOW)