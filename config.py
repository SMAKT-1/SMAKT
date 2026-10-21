# -*- coding: utf-8 -*-
"""
SMAKT Simulation Configuration File
===================================

This module centralizes all configuration parameters for the SMAKT framework simulation.
It is designed for flexibility, allowing researchers to easily model different industrial
scenarios, hardware constraints, and cultural contexts without modifying the core simulation logic.

Companion to the manuscript "SMAKT: A Five-Layer Architecture for Integrating
Affect-Aware Computing into Industrial Decision Support". All values used by the
reported experiments (E1-E10) are set in run_experiments.py; defaults here mirror
those values where the two must agree.
Version: 2.0
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any

# --- Core Simulation Parameters ---

@dataclass
class SimulationSettings:
    """Controls the fundamental aspects of the simulation run."""
    # Total number of discrete time steps to simulate (e.g., days)
    total_duration_days: int = 365
    # Number of workers (agents) in the simulated factory
    num_workers: int = 150
    # Random seed for reproducibility of results
    random_seed: int = 42
    # Directory to save raw simulation outputs
    output_dir: str = "./simulation_output"
    # Directory to save final analysis plots and reports
    results_dir: str = "./results"

# --- Agent and Environment Modeling ---

@dataclass
class AgentProfile:
    """Defines the baseline characteristics and behavioral parameters of a worker agent."""
    # Baseline affective state [stress, engagement, frustration]
    baseline_affect: List[float] = field(default_factory=lambda: [0.3, 0.7, 0.2])
    # Skill level, impacting task completion time and error rate (0.0 to 1.0)
    skill_level: float = 0.7
    # Intrinsic resilience to stress (higher is more resilient)
    stress_resilience: float = 0.6
    # Rate of fatigue accumulation per hour of work
    fatigue_rate_per_hour: float = 0.02

@dataclass
class CulturalProfile:
    """Models cultural variations in emotional expression and perception."""
    # Multiplier for EGC accuracy. 1.0 = baseline (e.g., Western), <1.0 = degradation.
    # Based on Murad et al. (2019), which suggests 12-18% degradation.
    # 1.0 - 0.15 = 0.85 represents a 15% penalty.
    egc_accuracy_modifier: float = 0.85
    # Sensitivity to hierarchical communication (affects stress from management messages)
    hierarchical_sensitivity: float = 1.2
    # Tendency to express frustration verbally (affects speech analysis modality)
    verbal_frustration_tendency: float = 0.8

@dataclass
class EnvironmentSettings:
    """Defines the physical and operational characteristics of the industrial environment."""
    # Base ambient noise level (0.0 to 1.0), affecting sensor data quality
    base_noise_level: float = 0.1
    # Electromagnetic interference (EMI) strength (0.0 to 1.0)
    emi_strength: float = 0.2
    # Probability of a random disruptive event per day (e.g., machine breakdown)
    daily_disturbance_probability: float = 0.05
    # Shift schedule in hours
    shift_duration_hours: int = 8

# --- Hardware and Performance Constraints ---

@dataclass
class EdgeDeviceSpecs:
    """Simulates the computational constraints of an edge device (e.g., Industrial PC or PLC)."""
    # CPU performance in GFLOPS (e.g., a modern industrial PLC)
    cpu_gflops: float = 50.0
    # Available RAM in MB
    available_ram_mb: int = 512
    # Network latency to central server/cloud in milliseconds
    network_latency_ms: float = 5.0

@dataclass
class EGCModelSpecs:
    """Defines the parameters for the Emotion Generating Calculations (EGC) model."""
    # Model size in MB, impacting memory and inference time
    model_size_mb: float = 1.2
    # Base accuracy in ideal lab conditions (e.g., 88% from literature)
    baseline_accuracy: float = 0.88
    # Target inference time on edge device in milliseconds
    target_inference_time_ms: float = 3.0
    # File path to the pre-trained EGC model (for real implementation)
    model_path: str = "./models/egc_model_v1.pt"

# --- SMAKT Framework Layer Parameters ---

@dataclass
class Layer1_Sensing:
    """Configuration for Layer 1: Sensing and Privacy."""
    # List of active data modalities for each agent
    active_modalities: List[str] = field(default_factory=lambda: ["mes_logs", "speech_analysis", "wearable_hrv"])
    # Time interval for data collection in minutes
    collection_interval_min: int = 15
    # Anonymization technique: 'pseudonymization', 'differential_privacy', etc.
    anonymization_method: str = "pseudonymization"
    # Data retention policy in days for raw, identifiable data
    data_retention_days_raw: int = 90

@dataclass
class Layer2_Modeling:
    """Configuration for Layer 2: Modeling and Anomaly Detection."""
    # Algorithm for anomaly detection: 'isolation_forest', 'lstm_ad', etc.
    anomaly_detection_algorithm: str = "isolation_forest"
    # Window size in hours for time-series aggregation
    aggregation_window_hours: int = 24
    # Threshold for flagging an anomaly (e.g., 2 standard deviations)
    anomaly_threshold_sigma: float = 2.0
    # Sensitivity for model drift detection (lower is more sensitive)
    drift_detection_sensitivity: float = 0.1

@dataclass
class Layer3_4_Knowledge:
    """Configuration for Layer 3 (Affect Mapping) and Layer 4 (Knowledge Synthesis)."""
    # File path for the affect-to-business KPI mapping rules
    kpi_mapping_rules_path: str = "./config/kpi_mapping_rules.json"
    # Interpretability method for decision support: 'shap', 'lime', 'rule_based'
    interpretability_method: str = "shap"
    # Threshold for triggering a predictive insight (e.g., turnover risk)
    predictive_insight_threshold: float = 0.75

@dataclass
class Layer5_Transformation:
    """Configuration for Layer 5: Decision and Action Orchestration."""
    # Time in hours before a high-risk prediction requires human validation
    human_validation_window_hr: int = 48
    # Simulated probability of manager accepting a recommendation.
    # NOTE: the comparative experiment (E10) fixes this at 0.55; keep in sync.
    recommendation_acceptance_rate: float = 0.55
    # Time in hours to simulate the effect of an intervention (e.g., career conversation)
    intervention_effect_duration_hr: int = 72

# --- Economic and Regulatory Parameters ---

@dataclass
class EconomicModel:
    """Parameters for calculating Return on Investment (ROI) and costs."""
    # Average annual cost of employee turnover (including recruitment, training, lost productivity).
    # Conservative end of benchmark ranges (50% of a ~$52k annual salary); used in E10.
    cost_per_turnover_usd: float = 26000
    # Hourly wage for an average worker
    average_hourly_wage_usd: float = 25.0
    # Annual cost per edge device (hardware + maintenance)
    edge_device_annual_cost_usd: float = 1500
    # Annual cost for cloud services and software licenses
    software_license_annual_cost_usd: float = 10000
    # Estimated annual salary for personnel managing the system (e.g., data analyst)
    personnel_annual_cost_usd: float = 80000

@dataclass
class ComplianceSettings:
    """Configuration for legal and ethical compliance."""
    # Data protection regulation standard: 'GDPR', 'CCPA', 'custom'
    regulation_standard: str = "GDPR"
    # Flag to require explicit worker consent before data collection
    require_explicit_consent: bool = True
    # Path to the configuration for the simulated ethics committee
    ethics_committee_config_path: str = "./config/ethics_rules.json"

# --- Main Configuration Class ---
@dataclass
class Config:
    """A master class to hold all configuration objects."""
    simulation: SimulationSettings = field(default_factory=SimulationSettings)
    agent_profile: AgentProfile = field(default_factory=AgentProfile)
    cultural_profile: CulturalProfile = field(default_factory=CulturalProfile)
    environment: EnvironmentSettings = field(default_factory=EnvironmentSettings)
    edge_device: EdgeDeviceSpecs = field(default_factory=EdgeDeviceSpecs)
    egc_model: EGCModelSpecs = field(default_factory=EGCModelSpecs)
    layer1: Layer1_Sensing = field(default_factory=Layer1_Sensing)
    layer2: Layer2_Modeling = field(default_factory=Layer2_Modeling)
    layer3_4: Layer3_4_Knowledge = field(default_factory=Layer3_4_Knowledge)
    layer5: Layer5_Transformation = field(default_factory=Layer5_Transformation)
    economic: EconomicModel = field(default_factory=EconomicModel)
    compliance: ComplianceSettings = field(default_factory=ComplianceSettings)

    def __post_init__(self):
        """Post-initialization checks and setup."""
        # Create output directories if they don't exist
        os.makedirs(self.simulation.output_dir, exist_ok=True)
        os.makedirs(self.simulation.results_dir, exist_ok=True)
        print(f"SMAKT Configuration Initialized.")
        print(f"  - Output Directory: {self.simulation.output_dir}")
        print(f"  - Results Directory: {self.simulation.results_dir}")
        print(f"  - Simulating {self.simulation.num_workers} workers for {self.simulation.total_duration_days} days.")

# --- Predefined Scenarios for Easy Demonstration ---

def get_scenario(scenario_name: str) -> Config:
    """
    Returns a pre-configured Config object based on a named scenario.
    This is extremely useful for demonstrating the framework's flexibility.
    """
    if scenario_name == "small_factory":
        config = Config()
        config.simulation.num_workers = 30
        config.edge_device.cpu_gflops = 25.0 # Less powerful hardware
        config.environment.base_noise_level = 0.15 # Noisier environment
        return config
    elif scenario_name == "large_multinational":
        config = Config()
        config.simulation.num_workers = 500
        config.cultural_profile.egc_accuracy_modifier = 0.80 # More cultural diversity
        config.compliance.regulation_standard = "GDPR" # Stricter compliance
        config.economic.cost_per_turnover_usd = 75000 # Higher turnover cost in skilled roles
        return config
    elif scenario_name == "high_tech_precision":
        config = Config()
        config.agent_profile.skill_level = 0.9
        config.environment.base_noise_level = 0.05 # Clean room environment
        config.egc_model.baseline_accuracy = 0.90 # Better model for skilled tasks
        return config
    else:
        # Default scenario
        return Config()

# Example of how to use it in the main script:
# if __name__ == '__main__':
#     # Load default configuration
#     default_cfg = Config()
#
#     # Load a specific scenario for demonstration
#     large_factory_cfg = get_scenario("large_multinational")
#
#     print("\n--- Default Config ---")
#     print(default_cfg)
#
#     print("\n--- Large Multinational Scenario Config ---")
#     print(large_factory_cfg)
