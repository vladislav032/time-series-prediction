#!/usr/bin/env python3
"""
Time Series Model Continue Training CLI

A sophisticated tool for continuing training of time series ensemble models based on additional data.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from time_series_prediction.scripts.train import TimeSeriesModelTrainer
from colorama import Fore, Style, init

init(autoreset=True)

class ConsoleLogger:
    """Basic console logger with color support."""
    
    def log(self, message: str, color: Optional[str] = Fore.RESET) -> None:
        print(f"{color}{message}{Style.RESET_ALL}")


class ContinueTrainingApp:
    """Main application class encapsulating logic for continuing model training."""
    
    def __init__(self):
        self.parser = self._configure_parser()
        self.args = self._parse_arguments()
        self.logger = ConsoleLogger()
        
    def _configure_parser(self) -> argparse.ArgumentParser:
        """Setup and return argument parser with all configurations."""
        parser = argparse.ArgumentParser(
            description='Time Series Model Continue Training Tool',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        parser.add_argument(
            'model_dir',
            type=str,
            help='Path to the directory containing the saved models for continuation'
        )
        parser.add_argument(
            'data_file',
            type=str,
            help='Path to the CSV file containing the new data for continuing training'
        )
        
        parser.add_argument(
            '-c', '--columns',
            type=str,
            required=True,
            help='Columns to use (@all[exclusions] or comma-separated list)'
        )
        parser.add_argument(
            '-w', '--window',
            type=int,
            default=50,
            help='Sliding window size for time series'
        )
        parser.add_argument(
            '-n', '--num-models',
            type=int,
            help='Number of models in the ensemble. If not provided, it is inferred from the model directory'
        )
        
        parser.add_argument(
            '-e', '--epochs',
            type=int,
            default=1000,
            help='Maximum training epochs per model'
        )
        parser.add_argument(
            '-a', '--accuracy',
            type=float,
            default=0.85,
            help='Target training accuracy (0-1)'
        )
        
        return parser
    
    def _parse_arguments(self) -> argparse.Namespace:
        """Parse and validate command line arguments."""
        args = self.parser.parse_args()
        
        # Validate numerical arguments
        if args.window < 1:
            self.parser.error("Window size must be positive")
        if args.accuracy <= 0 or args.accuracy >= 1:
            self.parser.error("Accuracy must be between 0 and 1")
            
        return args
    
    def _infer_num_models(self, model_dir: Path) -> int:
        """Infer the number of models based on files in the model directory."""
        model_files = list(model_dir.glob("model_*.pth"))
        return len(model_files)
    
    def run(self) -> int:
        """Main application entry point."""
        try:
            self._execute_training_pipeline()
            return 0
        except Exception as e:
            self._handle_error(e)
            return 1
    
    def _execute_training_pipeline(self) -> None:
        """Execute complete training workflow."""
        model_dir = Path(self.args.model_dir)
        data_file = Path(self.args.data_file)
        
        if not model_dir.exists() or not model_dir.is_dir():
            raise ValueError(f"Model directory not found: {model_dir}")
        if not data_file.exists() or not data_file.is_file():
            raise ValueError(f"Data file not found: {data_file}")
        
        data = pd.read_csv(data_file)
        if data.empty:
            raise ValueError("Data file contains no data")
        
        num_models = self.args.num_models or self._infer_num_models(model_dir)
        if num_models == 0:
            raise ValueError(f"No models found in the directory: {model_dir}")
        
        trainer = TimeSeriesModelTrainer(
            df=data,
            column_names=self.args.columns,
            window_size=self.args.window,
            n_models=num_models,
            output_dir=str(model_dir)
        )
        
        self._print_configuration_summary(model_dir, data_file, num_models)
        self.logger.log("🚀 Starting model continuation training...", Fore.GREEN)
        trainer.continue_training(
            new_df=data,
            target_accuracy=self.args.accuracy,
            max_epochs=self.args.epochs
        )
        self.logger.log("✅ Model continuation training completed successfully!", Fore.GREEN)
    
    def _print_configuration_summary(self, model_dir: Path, data_file: Path, num_models: int) -> None:
        """Print training configuration summary."""
        print("\n" + "="*50)
        print(Fore.BLUE + "⚙️ Continue Training Configuration")
        print("="*50)
        print(f"📂 Model directory: {model_dir}")
        print(f"📂 Data file: {data_file}")
        print(f"📊 Columns: {self.args.columns}")
        print(f"🖼️ Window size: {self.args.window}")
        print(f"🤖 Number of models: {num_models}")
        print(f"🎯 Target accuracy: {self.args.accuracy:.0%}")
        print(f"⏳ Max epochs: {self.args.epochs}")
        print("="*50 + "\n")
    
    def _handle_error(self, error: Exception) -> None:
        """Handle and display errors appropriately."""
        self.logger.log(f"❌ Error: {str(error)}", Fore.RED)
        print("\nUsage example:")
        print("  python continue_training_app.py models_dir data.csv -c @all[0,1] -w 30 -e 500 -a 0.9")
        print("\nFor detailed help:")
        print("  python continue_training_app.py -h")


if __name__ == "__main__":
    app = ContinueTrainingApp()
    sys.exit(app.run())