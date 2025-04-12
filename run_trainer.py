#!/usr/bin/env python3
"""
Time Series Model Trainer CLI

A sophisticated tool for training ensemble time series prediction models.
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


class TrainingApp:
    """Main application class encapsulating training logic."""
    
    def __init__(self):
        self.parser = self._configure_parser()
        self.args = self._parse_arguments()
        self.logger = ConsoleLogger()
        
    def _configure_parser(self) -> argparse.ArgumentParser:
        """Setup and return argument parser with all configurations."""
        parser = argparse.ArgumentParser(
            description='Time Series Model Training Tool',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        parser.add_argument(
            'input',
            type=str,
            help='Path to input CSV file containing time series data'
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
            default=5,
            help='Number of models to train in ensemble'
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
        
        parser.add_argument(
            '-o', '--output',
            type=str,
            default='models',
            help='Output directory for trained models'
        )
        
        return parser
    
    def _parse_arguments(self) -> argparse.Namespace:
        """Parse and validate command line arguments."""
        args = self.parser.parse_args()
        
        if args.window < 1:
            self.parser.error("Window size must be positive")
        if args.num_models < 1:
            self.parser.error("Number of models must be positive")
        if not (0 < args.accuracy < 1):
            self.parser.error("Accuracy must be between 0 and 1")
            
        return args
    
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
        data = self._load_input_data()
        trainer = self._initialize_trainer(data)
        
        self._print_configuration_summary()
        self._train_models(trainer)
        self._print_completion_message()
    
    def _load_input_data(self) -> pd.DataFrame:
        """Load and validate input data file."""
        try:
            data = pd.read_csv(self.args.input)
            if data.empty:
                raise ValueError("Input file contains no data")
            return data
        except FileNotFoundError:
            raise ValueError(f"Input file not found: {self.args.input}")
        except pd.errors.EmptyDataError:
            raise ValueError("Input file appears to be corrupt")
        except Exception as e:
            raise ValueError(f"Error loading input file: {str(e)}")
    
    def _initialize_trainer(self, data: pd.DataFrame) -> TimeSeriesModelTrainer:
        """Initialize and configure the model trainer."""
        return TimeSeriesModelTrainer(
            df=data,
            column_names=self.args.columns,
            window_size=self.args.window,
            n_models=self.args.num_models,
            output_dir=self.args.output
        )
    
    def _print_configuration_summary(self) -> None:
        """Print training configuration summary."""
        print("\n" + "="*50)
        print(Fore.BLUE + "⚙️ Training Configuration")
        print("="*50)
        print(f"📂 Input file: {self.args.input}")
        print(f"📊 Columns: {self.args.columns}")
        print(f"🖼️ Window size: {self.args.window}")
        print(f"🤖 Number of models: {self.args.num_models}")
        print(f"🎯 Target accuracy: {self.args.accuracy:.0%}")
        print(f"⏳ Max epochs: {self.args.epochs}")
        print(f"📁 Output directory: {self.args.output}")
        print("="*50 + "\n")
    
    def _train_models(self, trainer: TimeSeriesModelTrainer) -> None:
        """Execute the model training process."""
        self.logger.log("🚀 Starting model training...", Fore.GREEN)
        trainer.train_and_save_models(
            target_accuracy=self.args.accuracy,
            max_epochs=self.args.epochs
        )
    
    def _print_completion_message(self) -> None:
        """Print training completion summary."""
        print("\n" + "="*50)
        print(Fore.GREEN + "✅ Training completed successfully!")
        print("="*50)
        print(f"📁 Models saved to: {self.args.output}")
        print(f"🤖 Ensemble size: {self.args.num_models}")
        print("="*50)
    
    def _handle_error(self, error: Exception) -> None:
        """Handle and display errors appropriately."""
        self.logger.log(f"❌ Error: {str(error)}", Fore.RED)
        print("\nUsage example:")
        print("  python train.py data.csv -c @all[0,1] -w 30 -n 10 -a 0.9 -o my_models")
        print("\nFor detailed help:")
        print("  python train.py -h")

if __name__ == "__main__":
    app = TrainingApp()
    sys.exit(app.run())