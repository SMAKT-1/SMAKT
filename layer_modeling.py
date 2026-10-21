# -*- coding: utf-8 -*-
"""
SMAKT Layer 2: Modeling and Anomaly Detection
===============================================

This module implements the second layer of the SMAKT framework. It is responsible for
aggregating time-series data, performing real-time emotion inference using an EGC model,
and detecting patterns that indicate emerging risks (e.g., turnover, quality issues).

This layer transforms raw, privacy-safe sensor data into actionable affective signals
and anomaly flags, ready for business KPI mapping in Layer 3.


"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Any
from datetime import datetime, timedelta

# Scikit-learn for standard, well-vetted ML algorithms
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Assuming previous scripts are accessible
from config import Config, Layer2_Modeling, EGCModelSpecs
from layer_sensing import SensingLayer

class EGCModel:
    """
    A simulated Emotion Generating Calculations (EGC) model.
    This class acts as a wrapper for the inference process, measuring latency and accuracy.
    """
    def __init__(self, config: Config):
        self.config = config
        self.model_specs = config.egc_model
        # In a real implementation, you would load a pre-trained model here:
        # self.model = torch.load(self.model_specs.model_path)
        # self.model.eval()
        
        # For simulation, we create a simple function to simulate EGC behavior
        # This function maps sensor data to an affective vector [stress, engagement, frustration]
        # It is designed to be imperfect to allow for accuracy measurement.
        self._create_simulated_egc_function()

    def _create_simulated_egc_function(self):
        """Creates a plausible but imperfect function to simulate EGC inference."""
        # This function has some logic but also noise to simulate real-world inaccuracy
        def egc_inference(sensor_row: pd.Series) -> np.ndarray:
            # Base affect derived from simulated sensor data
            stress = max(0, min(1, sensor_row.get('error_corrections', 0) * 0.05 + sensor_row.get('speech_negative_word_count', 0) * 0.02))
            engagement = max(0, min(1, 1.0 - sensor_row.get('task_completion_variance', 0) * 2))
            frustration = max(0, min(1, sensor_row.get('help_button_presses', 0) * 0.2 + sensor_row.get('speech_negative_word_count', 0) * 0.05))
            
            # Add noise to simulate model imperfection
            noise = np.random.normal(0, 0.1, 3)
            
            return np.clip(np.array([stress, engagement, frustration]) + noise, 0, 1)
        
        self.inference_func = egc_inference

    def predict(self, sensor_data: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
        """
        Performs EGC inference on a batch of sensor data.
        
        Returns:
            A DataFrame with predicted affective vectors and the total inference time.
        """
        start_time = time.perf_counter()
        
        # Apply the simulated inference function to each row
        affective_vectors = sensor_data.apply(self.inference_func, axis=1, result_type='expand')
        affective_vectors.columns = ['predicted_stress', 'predicted_engagement', 'predicted_frustration']
        
        # Combine with original pseudonyms for tracking
        predictions_df = pd.concat([sensor_data[['worker_pseudonym', 'timestamp']].reset_index(drop=True), affective_vectors], axis=1)
        
        inference_time = (time.perf_counter() - start_time) * 1000  # in ms
        
        return predictions_df, inference_time

class AnomalyDetector:
    """
    Detects anomalous patterns in affective time-series data.
    """
    def __init__(self, config: Config):
        self.config = config
        self.settings = config.layer2
        # Use Isolation Forest as a robust, unsupervised anomaly detection algorithm
        self.model = IsolationForest(n_estimators=100, contamination='auto', random_state=config.simulation.random_seed)
        self.is_trained = False

    def train(self, historical_data: pd.DataFrame):
        """
        Trains the anomaly detector on a batch of historical data.
        This simulates the baseline establishment phase.
        """
        # We train on the core affective features
        features = historical_data[['predicted_stress', 'predicted_engagement', 'predicted_frustration']]
        self.model.fit(features)
        self.is_trained = True
        print(f"AnomalyDetector trained on {len(historical_data)} data points.")

    def detect(self, current_data: pd.DataFrame) -> pd.DataFrame:
        """
        Detects anomalies in the current data batch.
        Returns the original data with an 'anomaly_score' and 'is_anomaly' flag.
        """
        if not self.is_trained:
            raise RuntimeError("AnomalyDetector must be trained before detection.")
            
        features = current_data[['predicted_stress', 'predicted_engagement', 'predicted_frustration']]
        # The model returns -1 for anomalies and 1 for inliers
        anomaly_labels = self.model.predict(features)
        anomaly_scores = self.model.decision_function(features)
        
        current_data['anomaly_score'] = anomaly_scores
        current_data['is_anomaly'] = anomaly_labels == -1
        
        return current_data

class DriftDetector:
    """
    Monitors the performance of the EGC model to detect concept drift.
    """
    def __init__(self, config: Config):
        self.config = config
        self.sensitivity = config.layer2.drift_detection_sensitivity
        self.performance_history = []
        self.drift_threshold = 0.15 # A 15% drop in accuracy triggers a drift alert

    def update_performance(self, accuracy: float):
        """Adds the latest performance metric to the history."""
        self.performance_history.append(accuracy)
        # Keep only the last N measurements for rolling comparison
        if len(self.performance_history) > 50:
            self.performance_history.pop(0)

    def check_for_drift(self) -> bool:
        """
        Checks if the recent performance has dropped significantly compared to the historical average.
        """
        if len(self.performance_history) < 20: # Need enough data to make a judgment
            return False
        
        recent_avg = np.mean(self.performance_history[-10:])
        historical_avg = np.mean(self.performance_history[:-10])
        
        if historical_avg > 0 and (historical_avg - recent_avg) / historical_avg > self.drift_threshold:
            print(f"DRIFT DETECTED! Performance dropped from {historical_avg:.2f} to {recent_avg:.2f}")
            return True
        return False

class ModelingLayer:
    """
    The main class for SMAKT Layer 2. It orchestrates EGC inference, aggregation,
    anomaly detection, and drift monitoring.
    """
    def __init__(self, config: Config):
        self.config = config
        self.egc_model = EGCModel(config)
        self.anomaly_detector = AnomalyDetector(config)
        self.drift_detector = DriftDetector(config)
        
        # Store data for aggregation and training
        self.historical_affective_data = pd.DataFrame()

    def process_data(self, sensor_data: pd.DataFrame, ground_truth_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        The main pipeline for the modeling layer.
        """
        if sensor_data.empty:
            return {"status": "no_data"}

        # 1. EGC Inference (L_edge part)
        predictions_df, egc_latency = self.egc_model.predict(sensor_data)
        
        # 2. Aggregate data into historical store
        self.historical_affective_data = pd.concat([self.historical_affective_data, predictions_df], ignore_index=True)

        # 3. Train Anomaly Detector if not done yet
        if not self.anomaly_detector.is_trained and len(self.historical_affective_data) > 100:
            self.anomaly_detector.train(self.historical_affective_data)

        # 4. Anomaly Detection
        anomaly_flags_df = pd.DataFrame() # Default empty
        if self.anomaly_detector.is_trained:
            anomaly_flags_df = self.anomaly_detector.detect(predictions_df)

        # 5. Accuracy and Drift Monitoring (if ground truth is available)
        accuracy_metrics = {}
        if ground_truth_data is not None:
            accuracy_metrics = self._calculate_accuracy(predictions_df, ground_truth_data)
            self.drift_detector.update_performance(accuracy_metrics['accuracy'])
            self.drift_detector.check_for_drift()

        # 6. Aggregate into team-level patterns
        team_patterns = self._aggregate_team_patterns(anomaly_flags_df)
        
        return {
            "status": "success",
            "egc_predictions": predictions_df,
            "anomaly_flags": anomaly_flags_df,
            "team_patterns": team_patterns,
            "performance": {
                "egc_latency_ms": egc_latency,
                "accuracy_metrics": accuracy_metrics
            }
        }

    def _calculate_accuracy(self, predictions: pd.DataFrame, ground_truth: pd.DataFrame) -> Dict[str, float]:
        """
        Calculates model accuracy by comparing predictions to ground truth from the simulation.
        This is a critical step for validating the EGC model's performance.
        """
        # Merge predictions with ground truth on worker pseudonym and timestamp
        merged_df = pd.merge(predictions, ground_truth, on=['worker_pseudonym', 'timestamp'], how='inner')
        
        if merged_df.empty:
            return {"accuracy": 0.0, "message": "No matching ground truth data found."}

        # For simplicity, we'll classify stress as high (1) if > 0.7, and low (0) otherwise
        y_true_stress = (merged_df['stress'] > 0.7).astype(int)
        y_pred_stress = (merged_df['predicted_stress'] > 0.7).astype(int)
        
        accuracy = accuracy_score(y_true_stress, y_pred_stress)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true_stress, y_pred_stress, average='binary', zero_division=0)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    def _aggregate_team_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Aggregates individual data into team-level insights."""
        if data.empty:
            return {}
        
        # Example aggregation: average stress and number of anomalies in the last period
        avg_stress = data['predicted_stress'].mean()
        num_anomalies = data['is_anomaly'].sum()
        
        return {
            "avg_team_stress": avg_stress,
            "anomaly_count": num_anomalies,
            "data_points_analyzed": len(data)
        }

# --- Main execution for testing ---
if __name__ == '__main__':
    from config import get_scenario
    from agent_model import Environment

    # 1. Setup
    sim_config = get_scenario("small_factory")
    factory = Environment(config=sim_config)
    sensing_layer = SensingLayer(config=sim_config)
    modeling_layer = ModelingLayer(config=sim_config)

    print("SMAKT Layer 2: Modeling and Anomaly Detection Test")
    print("-" * 50)

    # 2. Run simulation for a while to gather data and train models
    print("Phase 1: Gathering data and training models...")
    for i in range(150): # Simulate for 150 hours to generate enough data
        factory.step()
        if i % 10 == 0:
            sensor_data = sensing_layer.collect_and_process_data(factory)
            if sensor_data is not None:
                # Initial processing without ground truth to train the anomaly detector
                modeling_layer.process_data(sensor_data)

    # 3. Run a new batch and evaluate performance with ground truth
    print("\nPhase 2: Running evaluation with ground truth...")
    factory.step() # One more step to get new data
    latest_sensor_data = sensing_layer.collect_and_process_data(factory)
    
    # Get ground truth from the environment for the latest data
    # We need to match the pseudonyms back to worker IDs to get ground truth
    ground_truth_list = []
    for worker in factory.workers:
        if worker.history:
            last_state = worker.history[-1]
            pseudonym = sensing_layer.pseudonymizer.pseudonymize(worker.id)
            ground_truth_list.append({
                'worker_pseudonym': pseudonym,
                'timestamp': last_state['timestamp'],
                'stress': last_state['stress'],
                'engagement': last_state['engagement'],
                'frustration': last_state['frustration']
            })
    ground_truth_df = pd.DataFrame(ground_truth_list)

    # Process the latest data with ground truth for accuracy calculation
    results = modeling_layer.process_data(latest_sensor_data, ground_truth_df)

    # 4. Display results
    if results["status"] == "success":
        print("\n--- EGC Predictions (Last 5) ---")
        print(results["egc_predictions"].head())
        
        print("\n--- Anomaly Flags ---")
        print(results["anomaly_flags"][['worker_pseudonym', 'anomaly_score', 'is_anomaly']].head())
        
        print("\n--- Team Patterns ---")
        print(results["team_patterns"])

        print("\n--- Performance Metrics ---")
        perf = results["performance"]
        print(f"  EGC Inference Latency: {perf['egc_latency_ms']:.2f} ms")
        if perf['accuracy_metrics']:
            print(f"  Accuracy (High Stress Detection): {perf['accuracy_metrics']['accuracy']:.2%}")
            print(f"  Precision: {perf['accuracy_metrics']['precision']:.2f}")
            print(f"  Recall: {perf['accuracy_metrics']['recall']:.2f}")
            print(f"  F1-Score: {perf['accuracy_metrics']['f1_score']:.2f}")
