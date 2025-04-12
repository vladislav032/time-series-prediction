#!/usr/bin/env python3
"""
Time Series Prediction CLI Tool

A sophisticated tool for making time series predictions using ensemble models.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from time_series_prediction.scripts.predict import TimeSeriesPredictor


class PredictionApp:
    """Main application class encapsulating prediction logic."""
    
    def __init__(self):
        self.parser = self._setup_arg_parser()
        self.args = self._parse_arguments()
        
    @staticmethod
    def _setup_arg_parser() -> argparse.ArgumentParser:
        """Configure and return argument parser."""
        parser = argparse.ArgumentParser(
            description='Time Series Prediction Tool',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        parser.add_argument(
            'input',
            type=str,
            help='Path to input CSV file containing time series data'
        )
        parser.add_argument(
            'models',
            type=str, 
            nargs='+',
            help='Model paths with optional weights (format: path[:weight])'
        )
        
        parser.add_argument(
            '-c', '--columns',
            type=str,
            required=True,
            help='Columns to use (@all[exclusions] or comma-separated list)'
        )
        parser.add_argument(
            '-f', '--future',
            type=int,
            default=50,
            help='Number of future values to predict'
        )
        
        parser.add_argument(
            '-s', '--show',
            type=str,
            help='Columns to visualize (default: all predicted columns)'
        )
        parser.add_argument(
            '-o', '--output',
            type=str,
            help='Output file path (predictions only saved if specified)'
        )
        
        return parser
    
    def _parse_arguments(self) -> argparse.Namespace:
        """Parse and validate command line arguments."""
        args = self.parser.parse_args()
        
        if not Path(args.input).exists():
            self.parser.error(f"Input file not found: {args.input}")
            
        return args
    
    def run(self) -> int:
        """Main application entry point."""
        try:
            self._execute_prediction_pipeline()
            return 0
        except Exception as e:
            self._handle_error(e)
            return 1
    
    def _execute_prediction_pipeline(self) -> None:
        """Execute the complete prediction workflow."""
        data = self._load_data()
        predictor = self._initialize_predictor(data)
        future_times, predictions = predictor.make_predictions()
        
        self._visualize_results(predictor, future_times, predictions)
        self._save_results_if_requested(predictor, future_times, predictions)
        self._print_success_message()
    
    def _load_data(self) -> pd.DataFrame:
        """Load and validate input data."""
        try:
            data = pd.read_csv(self.args.input)
            if data.empty:
                raise ValueError("Input file is empty")
            return data
        except pd.errors.EmptyDataError:
            raise ValueError("Input file appears to be corrupt or empty")
    
    def _initialize_predictor(self, data: pd.DataFrame) -> TimeSeriesPredictor:
        """Initialize and configure the time series predictor."""
        return TimeSeriesPredictor(
            dataframe=data,
            column_names=self.args.columns,
            model_weights=self.args.models,
            future_values=self.args.future
        )
    
    def _visualize_results(
        self,
        predictor: TimeSeriesPredictor,
        future_times: np.ndarray,
        predictions: np.ndarray
    ) -> None:
        """Generate visualization of prediction results."""
        columns_to_show = self.args.show if self.args.show else self.args.columns
        predictor.visualize_predictions(future_times, predictions, columns_to_show)
    
    def _save_results_if_requested(
        self,
        predictor: TimeSeriesPredictor,
        future_times: np.ndarray,
        predictions: np.ndarray
    ) -> None:
        """Save results to file if output path was specified."""
        if self.args.output:
            predictor.save_predictions(
                future_times=future_times,
                predictions=predictions,
                output_path=self.args.output,
                columns_to_save=self.args.columns
            )
            print(f"💾 Results saved to: {self.args.output}")
    
    def _print_success_message(self) -> None:
        """Print completion status and summary."""
        print("\n" + "="*50)
        print("✅ Prediction completed successfully!")
        print(f"📊 Visualized columns: {self.args.show or 'all'}")
        print(f"🔮 Predicted steps: {self.args.future}")
        print("="*50)
    
    def _handle_error(self, error: Exception) -> None:
        """Handle and display errors appropriately."""
        print("\n❌ Error:", str(error), file=sys.stderr)
        print("\nUsage example:")
        print("  python run.py data.csv model1:0.7 model2:0.3 \\")
        print("    -c @all[0,1] -f 30 -s Close -o predictions.csv")
        print("\nFor full help:")
        print("  python run.py -h")


if __name__ == "__main__":
    app = PredictionApp()
    sys.exit(app.run())