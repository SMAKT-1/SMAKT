# -*- coding: utf-8 -*-
"""
SMAKT Layer 1: Sensing and Privacy-Preserving Collection
========================================================

This module implements the first layer of the SMAKT framework. It is responsible for
collecting multi-modal affective data from the environment, applying privacy-preserving
measures at the point of collection, and managing data access based on roles.

This layer operationalizes the 'Privacy-by-Design' principle, ensuring that governance
is an architectural property, not a policy overlay.

"""

import time
import hashlib
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

# Assuming agent_model.py and config.py are in the same directory or accessible
from agent_model import Environment
from config import Config, Layer1_Sensing, ComplianceSettings

class Pseudonymizer:
    """
    Implements non-reversible pseudonymization to protect worker identity.
    This is a core component of the privacy-by-design architecture.
    """
    def __init__(self, salt: str):
        """
        Initializes the pseudonymizer with a cryptographic salt.
        A salt ensures that even if two workers have the same ID, their hashes will be different.
        """
        self.salt = salt.encode('utf-8')

    def pseudonymize(self, worker_id: int) -> str:
        """
        Generates a non-reversible hash for a given worker ID.
        """
        # Combine worker ID with the salt
        salted_id = f"{worker_id}{self.salt}".encode('utf-8')
        # Generate a SHA-256 hash
        return hashlib.sha256(salted_id).hexdigest()

class PrivacyManager:
    """
    Manages data access control and consent, enforcing privacy policies.
    """
    def __init__(self, config: Config):
        self.config = config
        self.consent_records = {} # In a real system, this would be a secure database
        self.audit_log = []

    def grant_consent(self, worker_pseudonym: str):
        """Simulates a worker granting explicit consent."""
        self.consent_records[worker_pseudonym] = True
        self._log_action(f"Consent GRANTED for {worker_pseudonym}")

    def revoke_consent(self, worker_pseudonym: str):
        """Simulates a worker revoking consent."""
        self.consent_records[worker_pseudonym] = False
        self._log_action(f"Consent REVOKED for {worker_pseudonym}")

    def has_consent(self, worker_pseudonym: str) -> bool:
        """Checks if a worker has an active consent record."""
        # In a real system, this would also check for expiry.
        return self.consent_records.get(worker_pseudonym, False)

    def check_access(self, requesting_role: str, data: Dict[str, Any]) -> bool:
        """
        Implements Role-Based Access Control (RBAC).
        """
        # Define access rules based on the paper's descriptions
        access_rules = {
            'worker': lambda d: d.get('worker_pseudonym') == d.get('requester_pseudonym'),
            'supervisor': lambda d: d.get('team_id') is not None, # Can see team aggregates
            'hr_manager': lambda d: True, # Can see all non-identifiable data
            'ethics_committee': lambda d: True, # Has veto power and full access
            'plant_manager': lambda d: 'worker_pseudonym' not in d # Only sees aggregates
        }
        
        has_permission = access_rules.get(requesting_role, lambda d: False)(data)
        self._log_action(f"Access CHECK by {requesting_role}: {'GRANTED' if has_permission else 'DENIED'}")
        return has_permission

    def _log_action(self, message: str):
        """Logs all privacy-related actions for auditability."""
        self.audit_log.append({
            'timestamp': datetime.now(),
            'message': message
        })

class SensingLayer:
    """
    The main class for SMAKT Layer 1. It orchestrates data collection, privacy application,
    and measures performance metrics like latency.
    """
    def __init__(self, config: Config):
        self.config = config
        self.settings = config.layer1
        self.privacy_manager = PrivacyManager(config)
        # Use a unique salt for this simulation run
        self.pseudonymizer = Pseudonymizer(salt=config.simulation.random_seed)
        
        # Pre-grant consent for all workers in the simulation for demonstration
        # In a real-world scenario, this would be an interactive process.
        for i in range(config.simulation.num_workers):
            self.privacy_manager.grant_consent(self.pseudonymizer.pseudonymize(i))

    def collect_and_process_data(self, environment: Environment) -> Optional[pd.DataFrame]:
        """
        The main pipeline for the sensing layer.
        It collects raw data, applies privacy measures, and measures latency.
        """
        start_time = time.perf_counter()

        # 1. Data Acquisition (L_sense)
        raw_sensor_df = environment.get_all_sensor_data()
        acquisition_time = time.perf_counter() - start_time

        if raw_sensor_df.empty:
            return None

        # 2. Privacy Preservation at the Source
        processed_df = self._apply_privacy_measures(raw_sensor_df)

        # 3. Edge Processing Simulation (L_edge)
        # In a real system, EGC inference would happen here.
        # We simulate its time cost based on config.
        edge_processing_time = self.config.egc_model.target_inference_time_ms / 1000.0
        
        # 4. Network Transmission Simulation (L_network)
        network_time = self.config.edge_device.network_latency_ms / 1000.0

        # 5. Total Latency Calculation
        total_latency = (time.perf_counter() - start_time) + edge_processing_time + network_time
        
        # Log performance metrics
        self._log_performance_metrics(acquisition_time, edge_processing_time, network_time, total_latency)
        
        # Return the processed, privacy-safe data
        return processed_df

    def _apply_privacy_measures(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies pseudonymization and filters data based on consent.
        """
        # 1. Pseudonymize worker IDs
        df['worker_pseudonym'] = df['worker_id'].apply(self.pseudonymizer.pseudonymize)
        df.drop(columns=['worker_id'], inplace=True)

        # 2. Filter out workers who have not given consent
        # This is a critical step for GDPR/CCPA compliance.
        initial_count = len(df)
        df = df[df['worker_pseudonym'].isin(self.privacy_manager.consent_records.keys())]
        df = df[df['worker_pseudonym'].apply(self.privacy_manager.has_consent)]
        final_count = len(df)
        
        if initial_count != final_count:
            print(f"Privacy Filter: {initial_count - final_count} records removed due to lack of consent.")

        # 3. Add simulated EMI noise to sensor readings
        emi_noise_level = self.config.environment.emi_strength
        if emi_noise_level > 0:
            # Add noise to numerical sensor columns
            for col in ['task_completion_variance', 'hrv_rmssd', 'error_corrections']:
                if col in df.columns:
                    noise = np.random.normal(0, emi_noise_level * 0.1, df[col].shape)
                    df[col] += noise
        
        return df

    def _log_performance_metrics(self, l_sense: float, l_edge: float, l_network: float, l_total: float):
        """Logs the latency breakdown for performance analysis."""
        metrics = {
            'timestamp': datetime.now(),
            'latency_acquisition_ms': l_sense * 1000,
            'latency_edge_processing_ms': l_edge * 1000,
            'latency_network_ms': l_network * 1000,
            'latency_total_ms': l_total * 1000
        }
        # In a real system, this would be sent to a monitoring system.
        print(f"Performance: Acquisition={metrics['latency_acquisition_ms']:.2f}ms, "
              f"Edge={metrics['latency_edge_processing_ms']:.2f}ms, "
              f"Network={metrics['latency_network_ms']:.2f}ms, "
              f"Total={metrics['latency_total_ms']:.2f}ms")

    def get_audit_log(self) -> List[Dict]:
        """Returns the privacy audit log for review."""
        return self.privacy_manager.audit_log

# --- Main execution for testing ---
if __name__ == '__main__':
    from config import get_scenario

    # 1. Setup
    sim_config = get_scenario("small_factory")
    factory = Environment(config=sim_config)
    sensing_layer = SensingLayer(config=sim_config)

    print("SMAKT Layer 1: Sensing and Privacy Test")
    print("-" * 40)

    # 2. Run a few simulation steps to generate data
    for _ in range(5):
        factory.step()

    # 3. Collect and process data through the sensing layer
    print("\nCollecting and processing data...")
    processed_data = sensing_layer.collect_and_process_data(factory)

    # 4. Display results
    if processed_data is not None:
        print("\n--- Processed Data (Privacy-Safe) ---")
        print(processed_data.head())
        
        print("\n--- Privacy Audit Log ---")
        for log_entry in sensing_layer.get_audit_log():
            print(f"  - {log_entry['timestamp']}: {log_entry['message']}")

    # 5. Demonstrate access control
    print("\n--- Access Control Demonstration ---")
    sample_data = {'worker_pseudonym': sensing_layer.pseudonymizer.pseudonymize(0)}
    
    print(f"Can a 'worker' access their own data? {sensing_layer.privacy_manager.check_access('worker', {**sample_data, 'requester_pseudonym': sample_data['worker_pseudonym']})}")
    print(f"Can a 'worker' access another's data? {sensing_layer.privacy_manager.check_access('worker', {**sample_data, 'requester_pseudonym': 'fake_hash'})}")
    print(f"Can 'hr_manager' access the data? {sensing_layer.privacy_manager.check_access('hr_manager', sample_data)}")
