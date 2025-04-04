import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from colorama import Fore, init, Style

from time_series_prediction.common.column_parser import ColumnParser
from time_series_prediction.seq2seq.attention_model import ImprovedModel

# Initialize Colorama
init(autoreset=True)


# ==================== PATTERNS IMPLEMENTATION ====================

class IDataLoader(ABC):
    """Interface for data loading strategies."""
    
    @abstractmethod
    def load_data(self, dataframe: pd.DataFrame, column_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        pass


class SingletonDataLoader(IDataLoader):
    """Singleton implementation of data loader."""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SingletonDataLoader, cls).__new__(cls)
        return cls._instance

    def load_data(self, dataframe: pd.DataFrame, column_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Load and prepare time series data with dates."""
        df = dataframe.dropna(subset=column_names + ['Date'])
        return df[column_names].values, pd.to_datetime(df['Date'], errors='coerce').values


class INormalizer(ABC):
    """Interface for normalization strategies."""
    
    @abstractmethod
    def normalize(self, data: np.ndarray) -> Tuple[np.ndarray, Any, Any]:
        pass
    
    @abstractmethod
    def denormalize(self, normalized_data: np.ndarray, original_data: np.ndarray) -> np.ndarray:
        pass


class MinMaxNormalizer(INormalizer):
    """Min-Max normalization strategy."""
    
    def normalize(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        normalized = 2 * (data - min_val) / (max_val - min_val) - 1
        return normalized, min_val, max_val
    
    def denormalize(self, normalized_data: np.ndarray, original_data: np.ndarray) -> np.ndarray:
        data_min = np.min(original_data, axis=0)
        data_max = np.max(original_data, axis=0)
        return ((normalized_data + 1) / 2) * (data_max - data_min) + data_min


class IModelFactory(ABC):
    """Interface for model factories."""
    
    @abstractmethod
    def create_model(self, input_size: int, output_size: int) -> torch.nn.Module:
        pass


class ImprovedModelFactory(IModelFactory):
    """Factory for creating ImprovedModel instances."""
    
    def create_model(self, input_size: int, output_size: int) -> torch.nn.Module:
        return ImprovedModel(input_size, output_size)


class ILogger(ABC):
    """Interface for logging."""
    
    @abstractmethod
    def log(self, message: str, color: str = Fore.GREEN) -> None:
        pass


class ConsoleLogger(ILogger):
    """Console logging implementation."""
    
    def log(self, message: str, color: str = Fore.GREEN) -> None:
        print(color + message + Style.RESET_ALL)


class IModelRepository(ABC):
    """Interface for model persistence."""
    
    @abstractmethod
    def load_models(self, model_paths: List[str], model_factory: IModelFactory) -> Dict[str, torch.nn.Module]:
        pass


class TorchModelRepository(IModelRepository):
    """Repository for PyTorch model persistence."""
    
    def __init__(self, column_names: List[str]):
        self.column_names = column_names  # Сохраняем названия столбцов
    
    def load_models(self, model_paths: List[str], model_factory: IModelFactory) -> Dict[str, torch.nn.Module]:
        """Load multiple models from disk."""
        models = {}
        for model_path in model_paths:
            model = model_factory.create_model(
                input_size=len(self.column_names), 
                output_size=len(self.column_names)
            )
            model.load_state_dict(torch.load(model_path, weights_only=True))
            model.eval()
            models[model_path] = model
        return models


class IWeightCalculator(ABC):
    """Interface for weight calculation strategies."""
    
    @abstractmethod
    def calculate_weights(self, model_weights: List[str], model_paths: List[str]) -> Dict[str, float]:
        pass


class ProportionalWeightCalculator(IWeightCalculator):
    """Calculates model weights proportionally."""
    
    def calculate_weights(self, model_weights: List[str], model_paths: List[str]) -> Dict[str, float]:
        """Calculate weights for model predictions."""
        model_weights_dict = {}
        total_specified_weight = 0.0
        unspecified_weight_count = 0

        # Parse specified weights
        for model_weight in model_weights:
            if ':' in model_weight:
                model_dir, weight = model_weight.split(':')
                weight = float(weight)
                model_weights_dict[model_dir] = weight
                total_specified_weight += weight
            else:
                model_weights_dict[model_weight] = None
                unspecified_weight_count += 1

        # Distribute remaining weight equally among unspecified models
        if unspecified_weight_count > 0:
            remaining_weight = 1.0 - total_specified_weight
            equal_weight = remaining_weight / unspecified_weight_count

            for model_dir in model_weights_dict:
                if model_weights_dict[model_dir] is None:
                    model_weights_dict[model_dir] = equal_weight

        # Validate total weight
        total_weight = sum(model_weights_dict.values())
        if not np.isclose(total_weight, 1.0):
            raise ValueError("Total weight must be equal to 1")

        # Distribute weights to individual model files
        weights = {}
        for model_dir, weight in model_weights_dict.items():
            dir_model_paths = [p for p in model_paths if p.startswith(model_dir)]
            for model_file in dir_model_paths:
                weights[model_file] = weight / len(dir_model_paths)
        
        return weights


class IPredictionAdjuster(ABC):
    """Interface for prediction adjustment strategies."""
    
    @abstractmethod
    def adjust_predictions(self, predictions: np.ndarray, last_actual: np.ndarray) -> np.ndarray:
        pass


class SimpleShiftAdjuster(IPredictionAdjuster):
    """Simple shift adjustment strategy."""
    
    def adjust_predictions(self, predictions: np.ndarray, last_actual: np.ndarray) -> np.ndarray:
        """Adjust predictions by shifting to match the last actual point."""
        correction = last_actual - predictions[0]
        return predictions + correction


class IResultVisualizer(ABC):
    """Interface for result visualization."""
    
    @abstractmethod
    def visualize(self, 
                actual_times: np.ndarray,
                actual_values: np.ndarray,
                predicted_times: np.ndarray,
                predicted_values: np.ndarray,
                column_names: List[str],
                columns_to_show: List[str]) -> None:
        pass


class MatplotlibVisualizer(IResultVisualizer):
    """Matplotlib-based visualization implementation."""
    
    def visualize(self, 
                actual_times: np.ndarray,
                actual_values: np.ndarray,
                predicted_times: np.ndarray,
                predicted_values: np.ndarray,
                column_names: List[str],
                columns_to_show: List[str]) -> None:
        """Plot actual vs predicted values."""
        plt.figure(figsize=(12, 6))
        for i, column_name in enumerate(column_names):
            if column_name in columns_to_show:
                plt.plot(actual_times, actual_values[:, i], label=f'Actual: {column_name}')
                plt.plot(predicted_times, predicted_values[:, i], label=f'Predicted: {column_name}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.xticks(rotation=45)
        plt.show()


class IResultSaver(ABC):
    """Interface for saving prediction results."""
    
    @abstractmethod
    def save(self, 
            predicted_times: np.ndarray,
            predicted_values: np.ndarray,
            column_names: List[str],
            columns_to_save: List[str],
            output_path: str) -> None:
        pass


class CSVResultSaver(IResultSaver):
    """CSV-based result saver."""
    
    def save(self, 
            predicted_times: np.ndarray,
            predicted_values: np.ndarray,
            column_names: List[str],
            columns_to_save: List[str],
            output_path: str) -> None:
        """Save predictions to CSV file."""
        prediction_data = {'time': predicted_times}
        for i, column_name in enumerate(column_names):
            if column_name in columns_to_save:
                prediction_data[f'predicted_value_{column_name}'] = predicted_values[:, i]

        prediction_df = pd.DataFrame(prediction_data)
        prediction_df.to_csv(output_path, index=False)


# ==================== MAIN CLASS ====================

class TimeSeriesPredictor:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        column_names: List[str],
        model_weights: List[str],
        future_values: int = 50,
        data_loader: IDataLoader = SingletonDataLoader(),
        normalizer: INormalizer = MinMaxNormalizer(),
        model_factory: IModelFactory = ImprovedModelFactory(),
        logger: ILogger = ConsoleLogger(),
        model_repository: IModelRepository = None,
        weight_calculator: IWeightCalculator = ProportionalWeightCalculator(),
        prediction_adjuster: IPredictionAdjuster = SimpleShiftAdjuster(),
        visualizer: IResultVisualizer = MatplotlibVisualizer(),
        result_saver: IResultSaver = CSVResultSaver()
    ):
        """
        Initialize TimeSeriesPredictor with configurable components.
        
        Args:
            dataframe: Input DataFrame containing time series data
            column_names: List of column names to use
            model_weights: List of model weights in format 'path:weight' or 'path'
            future_values: Number of future values to predict
            data_loader: Strategy for loading data
            normalizer: Strategy for data normalization
            model_factory: Factory for creating models
            logger: Strategy for logging
            model_repository: Strategy for model persistence
            weight_calculator: Strategy for weight calculation
            prediction_adjuster: Strategy for prediction adjustment
            visualizer: Strategy for result visualization
            result_saver: Strategy for saving results
        """
        self.dataframe = dataframe
        self.column_names = self._parse_columns(column_names)
        self.model_weights = model_weights
        self.future_values = future_values
        self.window_size = future_values
        
        # Dependency injection
        self.data_loader = data_loader
        self.normalizer = normalizer
        self.model_factory = model_factory
        self.logger = logger
        self.model_repository = model_repository or TorchModelRepository(self.column_names)
        self.weight_calculator = weight_calculator
        self.prediction_adjuster = prediction_adjuster
        self.visualizer = visualizer
        self.result_saver = result_saver
        
        # Data attributes
        self.time_series: Optional[np.ndarray] = None
        self.time_data: Optional[np.ndarray] = None
        self.normalized_time_series: Optional[np.ndarray] = None
        self.min_val: Optional[np.ndarray] = None
        self.max_val: Optional[np.ndarray] = None
        self.X: Optional[torch.Tensor] = None
        self.models: Optional[Dict[str, torch.nn.Module]] = None
        self.weights: Optional[Dict[str, float]] = None
        
        # Initialize data and models
        self._initialize()

    def _parse_columns(self, columns: List[str]) -> List[str]:
        """Parse and validate column names."""
        parser = ColumnParser(self.dataframe)
        return parser.parse_columns(columns)

    def _initialize(self) -> None:
        """Initialize data and models."""
        # Load and normalize data
        self.time_series, self.time_data = self.data_loader.load_data(self.dataframe, self.column_names)
        self.normalized_time_series, self.min_val, self.max_val = self.normalizer.normalize(self.time_series)
        self.X = self._create_windowed_data(self.normalized_time_series, self.window_size)
        
        # Load models and calculate weights
        model_paths = self._get_model_paths()
        self.models = self.model_repository.load_models(model_paths, self.model_factory)
        self.weights = self.weight_calculator.calculate_weights(self.model_weights, list(self.models.keys()))

    def _get_model_paths(self) -> List[str]:
        """Get paths to all model files."""
        model_paths = []
        for model_weight in self.model_weights:
            model_dir = model_weight.split(':')[0]
            paths = [os.path.join(model_dir, f) 
                    for f in os.listdir(model_dir) 
                    if f.startswith('model_') and f.endswith('.pth')]
            model_paths.extend(paths)
        return model_paths

    def _create_windowed_data(self, data: np.ndarray, window_size: int) -> torch.Tensor:
        """Create windowed dataset for time series prediction."""
        X = np.lib.stride_tricks.sliding_window_view(data, window_shape=(window_size, data.shape[1]))
        return torch.tensor(X.reshape(-1, window_size, data.shape[1]), dtype=torch.float32)

    def _predict_with_model(self, model: torch.nn.Module, X_new: torch.Tensor) -> np.ndarray:
        """Make predictions with a single model."""
        predicted = []
        with torch.no_grad():
            for _ in range(self.future_values):
                X_new = X_new.reshape(1, self.window_size, X_new.shape[2])
                pred = model(X_new).cpu().numpy()
                predicted.append(pred[0])
                X_new = torch.cat([X_new[:, 1:, :], torch.tensor(pred).reshape(1, 1, -1)], axis=1)
        return np.array(predicted)

    def make_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        """Make weighted predictions using all models."""
        weighted_predictions = np.zeros((self.future_values, len(self.column_names)))
        X_last = self.X[-1].reshape(1, self.window_size, self.X.shape[2])

        # Parallel prediction using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
            futures = {
                executor.submit(self._predict_with_model, model, X_last): model_path 
                for model_path, model in self.models.items()
            }

            for future in futures:
                model_path = futures[future]
                model_pred = future.result()
                weighted_predictions += model_pred * self.weights[model_path]

        # Denormalize and adjust predictions
        denormalized = self.normalizer.denormalize(weighted_predictions, self.time_series)
        adjusted = self.prediction_adjuster.adjust_predictions(denormalized, self.time_series[-1])

        # Calculate future timestamps
        time_diff = self.time_data[1] - self.time_data[0]
        future_times = np.array([self.time_data[-1] + time_diff * i for i in range(1, self.future_values + 1)])

        return future_times, adjusted

    def visualize_predictions(self, future_times: np.ndarray, predictions: Tuple[np.ndarray, np.ndarray], columns_to_show: List[str]) -> None:
        """Visualize predictions using the configured visualizer."""
        if isinstance(predictions, tuple):
            predicted_times, predicted_values = predictions
        else:
            predicted_times = future_times
            predicted_values = predictions

        self.visualizer.visualize(
            actual_times=self.time_data,
            actual_values=self.time_series,
            predicted_times=predicted_times,
            predicted_values=predicted_values,
            column_names=self.column_names,
            columns_to_show=columns_to_show
        )

    def save_predictions(self, future_times: np.ndarray, predictions: np.ndarray, output_path: str, columns_to_save: List[str]) -> None:
        """Save predictions using the configured saver."""
        self.result_saver.save(
            predicted_times=future_times,
            predicted_values=predictions,
            column_names=self.column_names,
            columns_to_save=columns_to_save,
            output_path=output_path
        )
        self.logger.log(f'Predictions saved to {output_path}', Fore.GREEN)