# -*- coding: utf-8 -*-
"""
SMAKT Layer 3 & 4: Affect Mapping and Knowledge Synthesis
==========================================================

This module implements Layers 3 (Affect Mapping) and 4 (Knowledge) of the SMAKT framework.
It translates raw affective signals and anomaly flags into context-aware business KPIs,
generates diagnostic, predictive, and prescriptive insights, and provides explanations
for its recommendations.

This layer is the bridge between technical signals and strategic business intelligence.


"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime

# For interpretability (SHAP is a standard, powerful choice)
try:
    import shap
    import sklearn
    SHAP_AVAILABLE = True
except ImportError:
    print("Warning: 'shap' library not found. Interpretability features will be simulated.")
    SHAP_AVAILABLE = False

# Assuming previous scripts are accessible
from config import Config, Layer3_4_Knowledge
from layer_modeling import ModelingLayer

class AffectToKPIMapper:
    """
    Implements the context-aware translation from affective signals to business KPIs.
    This is the core of Layer 3.
    """
    def __init__(self, config: Config):
        self.config = config
        self.mapping_rules = self._load_mapping_rules()

    def _load_mapping_rules(self) -> Dict:
        """
        Loads business rules for mapping affect to KPIs.
        In a real system, this would be loaded from a file (e.g., JSON) or a rules engine.
        """
        # For this example, we define them directly. This makes the logic transparent.
        return {
            "turnover_risk": {
                "conditions": [
                    {"signal": "sustained_high_stress", "duration_weeks": 4, "threshold": 0.7},
                    {"signal": "declining_engagement", "duration_weeks": 4, "threshold": 0.4}
                ],
                "kpi_impact": {"turnover_probability": 0.75, "time_to_exit_weeks": 8}
            },
            "quality_risk": {
                "conditions": [
                    {"signal": "sharp_frustration_spike", "threshold": 0.8},
                    {"signal": "task_complexity", "threshold": 0.6} # Context modifier
                ],
                "kpi_impact": {"defect_rate_increase": 0.15, "time_to_defect_days": 3}
            },
            "change_resistance": {
                "conditions": [
                    {"signal": "collective_anxiety", "context": "post_announcement"}
                ],
                "kpi_impact": {"adoption_delay_weeks": 2, "resistance_level": "high"}
            }
        }

    def map(self, modeling_output: Dict, context: Dict) -> List[Dict]:
        """
        Maps modeling outputs to business signals based on defined rules.
        """
        business_signals = []
        predictions_df = modeling_output.get("egc_predictions", pd.DataFrame())
        anomaly_flags_df = modeling_output.get("anomaly_flags", pd.DataFrame())

        if predictions_df.empty:
            return []

        # Check each rule against the current data
        for rule_name, rule_details in self.mapping_rules.items():
            if self._evaluate_rule(rule_details, predictions_df, anomaly_flags_df, context):
                signal = {
                    "signal_type": rule_name,
                    "timestamp": datetime.now(),
                    "kpi_impact": rule_details["kpi_impact"],
                    "affected_workers": self._get_affected_workers(rule_details, predictions_df, anomaly_flags_df)
                }
                business_signals.append(signal)
        
        return business_signals

    def _evaluate_rule(self, rule: Dict, predictions: pd.DataFrame, anomalies: pd.DataFrame, context: Dict) -> bool:
        """Evaluates if a specific business rule is triggered."""
        # This is a simplified evaluation. A real system would be more robust.
        if rule["conditions"][0]["signal"] == "sustained_high_stress":
            # Simplified check: if any worker currently has high stress
            return predictions['predicted_stress'].max() > rule["conditions"][0]["threshold"]
        
        if rule["conditions"][0]["signal"] == "sharp_frustration_spike":
            # Check if any anomaly flag is true (which could be a frustration spike)
            return anomalies['is_anomaly'].any()

        if rule["conditions"][0]["signal"] == "collective_anxiety":
            # Check context
            return context.get("recent_announcement", False)
        
        return False

    def _get_affected_workers(self, rule: Dict, predictions: pd.DataFrame, anomalies: pd.DataFrame) -> List[str]:
        """Identifies workers who trigger the rule."""
        # Simplified: returns all workers with stress above a threshold
        if "stress" in rule["conditions"][0]["signal"]:
            return predictions[predictions['predicted_stress'] > 0.7]['worker_pseudonym'].tolist()
        return []

class InsightEngine:
    """
    Generates diagnostic, predictive, and prescriptive insights from business signals.
    This is the core of Layer 4.
    """
    def __init__(self, config: Config):
        self.config = config
        self.interpretability_method = config.layer3_4.interpretability_method

    def generate_insights(self, business_signals: List[Dict], modeling_output: Dict) -> List[Dict]:
        """
        Translates business signals into actionable insights.
        """
        insights = []
        for signal in business_signals:
            insight = {
                "type": "prescriptive", # Can be diagnostic, predictive, or prescriptive
                "signal_type": signal["signal_type"],
                "timestamp": signal["timestamp"],
                "recommendation": self._generate_recommendation(signal),
                "confidence": np.random.uniform(0.8, 0.95), # Simulated confidence
                "affected_pseudonyms": signal["affected_workers"]
            }
            insights.append(insight)
        
        return insights

    def _generate_recommendation(self, signal: Dict) -> str:
        """Creates a specific, actionable recommendation."""
        if signal["signal_type"] == "turnover_risk":
            return "Initiate a career conversation and discuss potential role adjustments for affected workers."
        if signal["signal_type"] == "quality_risk":
            return "Review the user interface for the specific tool and provide targeted retraining."
        if signal["signal_type"] == "change_resistance":
            return "Hold a Q&A session with leadership to clarify the goals and impact of the recent announcement."
        return "Monitor the situation closely."

    def explain_recommendation(self, insight: Dict, modeling_output: Dict) -> Dict:
        """
        Provides an explanation for a given recommendation using SHAP.
        This directly addresses the interpretability requirement.
        """
        if not SHAP_AVAILABLE or self.interpretability_method != 'shap':
            return {"method": "rule_based", "explanation": "Recommendation based on pre-defined business rules linking high stress to turnover risk."}

        # In a real system, you would use the trained model from ModelingLayer
        # For this simulation, we'll create a dummy model to demonstrate SHAP
        # This part is for demonstration of the *capability* of interpretability.
        
        # Create a simple dummy model that predicts turnover risk
        # This is a placeholder for the actual EGC model
        dummy_model = sklearn.linear_model.LogisticRegression()
        features = modeling_output['egc_predictions'][['predicted_stress', 'predicted_engagement', 'predicted_frustration']]
        # Create a dummy target variable (high stress = high risk)
        target = (features['predicted_stress'] > 0.7).astype(int)
        dummy_model.fit(features, target)

        # Use SHAP to explain the prediction for the first affected worker
        explainer = shap.Explainer(dummy_model, features)
        shap_values = explainer(features.iloc[[0]]) # Explain the first instance

        # Convert SHAP output to a human-readable format
        feature_names = features.columns
        shap_contributions = shap_values.values[0]
        
        explanation = {
            "method": "shap",
            "worker_pseudonym": insight['affected_pseudonyms'][0],
            "feature_contributions": {
                feature_names[i]: float(shap_contributions[i]) for i in range(len(feature_names))
            },
            "summary": f"The recommendation is primarily driven by the worker's high stress level (SHAP value: {shap_contributions[0]:.2f})."
        }
        return explanation

class KnowledgeLayer:
    """
    The main class for SMAKT Layers 3 & 4. It orchestrates the mapping and insight generation.
    """
    def __init__(self, config: Config):
        self.config = config
        self.mapper = AffectToKPIMapper(config)
        self.insight_engine = InsightEngine(config)

    def process(self, modeling_output: Dict, context: Dict) -> Dict[str, Any]:
        """
        The main pipeline for the knowledge layer.
        """
        # 1. Map affective signals to business KPIs
        business_signals = self.mapper.map(modeling_output, context)

        # 2. Generate actionable insights from business signals
        insights = self.insight_engine.generate_insights(business_signals, modeling_output)

        # 3. Generate explanations for the insights
        explanations = []
        for insight in insights:
            explanation = self.insight_engine.explain_recommendation(insight, modeling_output)
            explanations.append(explanation)

        return {
            "status": "success",
            "business_signals": business_signals,
            "insights": insights,
            "explanations": explanations
        }

# --- Main execution for testing ---
if __name__ == '__main__':
    from config import get_scenario
    from agent_model import Environment
    from layer_sensing import SensingLayer
    from layer_modeling import ModelingLayer

    # 1. Setup
    sim_config = get_scenario("small_factory")
    factory = Environment(config=sim_config)
    sensing_layer = SensingLayer(config=sim_config)
    modeling_layer = ModelingLayer(config=sim_config)
    knowledge_layer = KnowledgeLayer(config=sim_config)

    print("SMAKT Layer 3 & 4: Affect Mapping and Knowledge Test")
    print("-" * 50)

    # 2. Run simulation to generate data
    # We need enough data to train the anomaly detector in the modeling layer
    print("Phase 1: Gathering data and training models...")
    for i in range(150):
        factory.step()
        if i % 10 == 0:
            sensor_data = sensing_layer.collect_and_process_data(factory)
            if sensor_data is not None:
                modeling_layer.process_data(sensor_data)

    # 3. Create a scenario that should trigger a signal
    print("\nPhase 2: Triggering a high-stress scenario...")
    # Let's manually increase stress for a few workers to simulate a real event
    for worker in factory.workers[:5]:
        worker.affective_state[0] = 0.9 # Set stress to high
        worker._log_state(datetime.now()) # Log this new state

    # 4. Process the new data through all layers
    factory.step()
    latest_sensor_data = sensing_layer.collect_and_process_data(factory)
    modeling_results = modeling_layer.process_data(latest_sensor_data)

    # Define context for the mapping layer
    context = {"recent_announcement": False} # Let's say no announcement was made

    # 5. Process through the knowledge layer
    knowledge_results = knowledge_layer.process(modeling_results, context)

    # 6. Display results
    if knowledge_results["status"] == "success":
        print("\n--- Business Signals Detected ---")
        for signal in knowledge_results["business_signals"]:
            print(f"  - Signal: {signal['signal_type']}, Impact: {signal['kpi_impact']}")

        print("\n--- Generated Insights ---")
        for insight in knowledge_results["insights"]:
            print(f"  - Type: {insight['type']}, Recommendation: {insight['recommendation']}")
            print(f"    Confidence: {insight['confidence']:.2%}, Affected: {len(insight['affected_pseudonyms'])} workers")

        print("\n--- Explanations ---")
        for explanation in knowledge_results["explanations"]:
            print(f"  - Method: {explanation['method']}")
            if explanation['method'] == 'shap':
                print(f"    Summary: {explanation['summary']}")
                print(f"    Feature Contributions: {explanation['feature_contributions']}")
            else:
                print(f"    Explanation: {explanation['explanation']}")