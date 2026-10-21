# -*- coding: utf-8 -*-
"""
SMAKT Layer 5: Transformation and Decision Orchestration
=========================================================

This module implements the fifth and final layer of the SMAKT framework. It is responsible
for orchestrating decisions based on insights, managing the feedback loop, and simulating
the impact of interventions on the agent population.

This layer closes the loop, turning insights into actions and learning from the outcomes
to continuously improve the system.


"""

import time
import random
import pandas as pd
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import numpy as np


# Assuming previous scripts are accessible
from config import Config, Layer5_Transformation, EconomicModel
from agent_model import Environment, Worker
from layer_knowledge import KnowledgeLayer

class InterventionSimulator:
    """
    Simulates the effect of a managerial intervention on a worker's state.
    This is crucial for closing the feedback loop and measuring ROI.
    """
    def __init__(self, config: Config):
        self.config = config
        self.intervention_effects = {
            "career_conversation": {"stress": -0.2, "engagement": 0.15, "duration_hr": 72},
            "retraining": {"frustration": -0.3, "engagement": 0.1, "duration_hr": 48},
            "short_break": {"stress": -0.1, "fatigue": -0.3, "duration_hr": 1},
            "clarification_meeting": {"stress": -0.15, "engagement": 0.2, "duration_hr": 24}
        }

    def apply(self, intervention: Dict, worker: Worker):
        """Applies the effects of an intervention to a worker."""
        action_type = intervention.get("action_type")
        if action_type in self.intervention_effects:
            effects = self.intervention_effects[action_type]
            worker.affective_state[0] += effects.get("stress", 0) # Stress
            worker.affective_state[1] += effects.get("engagement", 0) # Engagement
            worker.affective_state[2] += effects.get("frustration", 0) # Frustration
            worker.fatigue += effects.get("fatigue", 0)

            # Clamp values to be within [0, 1]
            worker.affective_state = np.clip(worker.affective_state, 0.0, 1.0)
            worker.fatigue = np.clip(worker.fatigue, 0.0, 1.0)
            
            print(f"[INTERVENTION] Applied '{action_type}' to Worker {worker.id}. New state: S={worker.affective_state[0]:.2f}, E={worker.affective_state[1]:.2f}")
        else:
            print(f"[INTERVENTION] Unknown action type: {action_type}")

class DecisionEngine:
    """
    Manages the decision-making process, including authorization and execution.
    """
    def __init__(self, config: Config, pseudonymizer=None):
        self.config = config
        self.settings = config.layer5
        self.intervention_simulator = InterventionSimulator(config)
        # The pseudonymizer is injected so actions can be resolved inside the
        # trusted intervention boundary without exposing identity to Layer 3/4.
        self.pseudonymizer = pseudonymizer
        self.decision_log = []

    def orchestrate(self, insights: List[Dict], environment: Environment) -> List[Dict]:
        """
        Reviews insights, authorizes actions, and executes them.
        """
        executed_actions = []
        for insight in insights:
            action = self._create_action(insight)
            if self._authorize_action(action):
                self._execute_action(action, environment)
                executed_actions.append(action)
        return executed_actions

    def _create_action(self, insight: Dict) -> Dict:
        """Creates a specific action from a prescriptive insight."""
        action_type = "unknown"
        if "career conversation" in insight.get("recommendation", "").lower():
            action_type = "career_conversation"
        elif "retraining" in insight.get("recommendation", "").lower():
            action_type = "retraining"
        
        return {
            "insight_id": id(insight),
            "action_type": action_type,
            "affected_pseudonyms": insight.get("affected_pseudonyms", []),
            "priority": "high" if insight.get("signal_type") == "turnover_risk" else "medium",
            "status": "pending"
        }

    def _authorize_action(self, action: Dict) -> bool:
        """
        Simulates the authorization process based on action priority and human validation.
        """
        # High-priority actions require human validation
        if action["priority"] == "high":
            # Simulate a manager's decision
            if random.random() < self.settings.recommendation_acceptance_rate:
                action["status"] = "authorized"
                return True
            else:
                action["status"] = "rejected_by_manager"
                return False
        else:
            # Lower priority actions can be automated
            action["status"] = "authorized"
            return True

    def _execute_action(self, action: Dict, environment: Environment):
        """Finds the affected workers and applies the intervention."""
        if self.pseudonymizer is None:
            raise RuntimeError("DecisionEngine requires a pseudonymizer to resolve interventions.")
        action["status"] = "executed"
        action["execution_timestamp"] = datetime.now()

        # Pseudonyms are resolved back to workers only here, inside the trusted
        # intervention boundary (Layer 5); Layers 1-4 never see raw identities.
        worker_by_pseudonym = {self.pseudonymizer.pseudonymize(w.id): w for w in environment.workers}

        for pseudonym in action["affected_pseudonyms"]:
            worker = worker_by_pseudonym.get(pseudonym)
            if worker is not None:
                self.intervention_simulator.apply(action, worker)
        
        self._log_decision(action)

    def _log_decision(self, action: Dict):
        """Logs the decision for audit and analysis."""
        self.decision_log.append(action)

class BaselineSimulator:
    """
    Simulates a "business-as-usual" scenario without SMAKT interventions.
    This is essential for comparative analysis and ROI calculation (Irregularity #29).
    """
    def __init__(self, config: Config):
        self.config = config

    def simulate_step(self, environment: Environment):
        """In a baseline scenario, no proactive interventions are made."""
        # We can simulate that problems are only addressed reactively, e.g., after a worker leaves.
        # For simplicity, we just let the environment run its course without intervention.
        pass

class TransformationLayer:
    """
    The main class for SMAKT Layer 5. It orchestrates the entire decision loop.
    """
    def __init__(self, config: Config, pseudonymizer=None):
        self.config = config
        self.decision_engine = DecisionEngine(config, pseudonymizer=pseudonymizer)
        self.baseline_simulator = BaselineSimulator(config)
        self.smaKT_results_log = []
        self.baseline_results_log = []

    def run_smaKT_cycle(self, environment: Environment, knowledge_layer: KnowledgeLayer, modeling_output: Dict):
        """Runs one full cycle of the SMAKT framework."""
        context = {"recent_announcement": False} # Can be dynamically updated
        knowledge_results = knowledge_layer.process(modeling_output, context)
        
        if knowledge_results["status"] == "success":
            insights = knowledge_results.get("insights", [])
            executed_actions = self.decision_engine.orchestrate(insights, environment)
            
            # Log results for this cycle
            cycle_log = {
                "timestamp": datetime.now(),
                "insights_count": len(insights),
                "actions_executed": len(executed_actions),
                "avg_team_stress": modeling_output.get("team_patterns", {}).get("avg_team_stress", 0)
            }
            self.smaKT_results_log.append(cycle_log)
            return cycle_log
        return {"status": "no_insights"}

    def run_baseline_cycle(self, environment: Environment):
        """Runs one cycle of the baseline scenario."""
        self.baseline_simulator.simulate_step(environment)
        
        # Calculate baseline metrics
        avg_stress = np.mean([w.affective_state[0] for w in environment.workers])
        cycle_log = {
            "timestamp": datetime.now(),
            "avg_team_stress": avg_stress
        }
        self.baseline_results_log.append(cycle_log)
        return cycle_log

    def generate_final_report(self) -> Dict:
        """Generates a comparative report between SMAKT and baseline scenarios."""
        if not self.smaKT_results_log or not self.baseline_results_log:
            return {"error": "Not enough data to generate report."}

        smaKT_stress = np.mean([log["avg_team_stress"] for log in self.smaKT_results_log])
        baseline_stress = np.mean([log["avg_team_stress"] for log in self.baseline_results_log])
        
        stress_reduction = baseline_stress - smaKT_stress
        
        # Simple ROI estimation based on stress reduction
        # Assume a 10% stress reduction leads to a 5% reduction in turnover-related costs
        cost_saving = (stress_reduction / baseline_stress) * 0.5 * self.config.economic.cost_per_turnover_usd * self.config.simulation.num_workers

        return {
            "smaKT_avg_stress": smaKT_stress,
            "baseline_avg_stress": baseline_stress,
            "stress_reduction_percentage": (stress_reduction / baseline_stress) * 100,
            "estimated_annual_cost_saving_usd": cost_saving,
            "total_smaKT_interventions": sum(log["actions_executed"] for log in self.smaKT_results_log)
        }

# --- Main execution for testing ---
if __name__ == '__main__':
    from config import get_scenario
    from agent_model import Environment
    from layer_sensing import SensingLayer
    from layer_modeling import ModelingLayer

    # 1. Setup
    sim_config = get_scenario("small_factory")
    
    # Create two separate environments for a fair comparison
    smaKT_factory = Environment(config=sim_config)
    baseline_factory = Environment(config=sim_config)
    
    sensing_layer = SensingLayer(config=sim_config)
    modeling_layer = ModelingLayer(config=sim_config)
    knowledge_layer = KnowledgeLayer(config=sim_config)
    transformation_layer = TransformationLayer(config=sim_config)

    print("SMAKT Layer 5: Transformation and Decision Orchestration Test")
    print("-" * 60)

    # 2. Initial data gathering and training for the SMAKT environment
    print("Phase 1: Training SMAKT models...")
    for _ in range(150):
        smaKT_factory.step()
        sensor_data = sensing_layer.collect_and_process_data(smaKT_factory)
        if sensor_data is not None:
            modeling_layer.process_data(sensor_data)

    # 3. Run comparative simulation
    print("\nPhase 2: Running comparative simulation (SMAKT vs Baseline)...")
    simulation_cycles = 50
    for i in range(simulation_cycles):
        # --- SMAKT Cycle ---
        smaKT_factory.step()
        latest_sensor_data = sensing_layer.collect_and_process_data(smaKT_factory)
        modeling_results = modeling_layer.process_data(latest_sensor_data)
        transformation_layer.run_smaKT_cycle(smaKT_factory, knowledge_layer, modeling_results)

        # --- Baseline Cycle ---
        baseline_factory.step()
        transformation_layer.run_baseline_cycle(baseline_factory)
        
        if (i+1) % 10 == 0:
            print(f"  Completed {i+1}/{simulation_cycles} cycles...")

    # 4. Generate and display the final report
    print("\nPhase 3: Generating final report...")
    final_report = transformation_layer.generate_final_report()

    print("\n--- Comparative Analysis Report ---")
    print(f"  Average Team Stress (Baseline): {final_report['baseline_avg_stress']:.3f}")
    print(f"  Average Team Stress (SMAKT):    {final_report['smaKT_avg_stress']:.3f}")
    print(f"  Stress Reduction:               {final_report['stress_reduction_percentage']:.2f}%")
    print(f"  Total SMAKT Interventions:      {final_report['total_smaKT_interventions']}")
    print(f"  Estimated Annual Cost Saving:   ${final_report['estimated_annual_cost_saving_usd']:,.2f}")

    print("\n--- Last few decisions made by SMAKT ---")
    for log in transformation_layer.decision_engine.decision_log[-3:]:
        print(f"  - Action: {log['action_type']}, Status: {log['status']}, Priority: {log['priority']}")
