# -*- coding: utf-8 -*-
"""
SMAKT Agent and Environment Model
=================================

This module defines the core components of the simulation: the Worker agents and the
industrial Environment they operate in. It brings the theoretical concepts of the
SMAKT paper to life by simulating the dynamic interplay between human affect,
task performance, and external events.

Companion to the manuscript "SMAKT: A Five-Layer Architecture for Integrating
Affect-Aware Computing into Industrial Decision Support". Note: the disruptive-event
machinery (Environment._introduce_disturbance, Worker.react_to_event) is part of the
artifact but is NOT activated by the experiment pipeline (run_experiments.py); the
reported experiments use an event-free data-generating process.
Version: 2.0
"""

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Any
from config import Config, AgentProfile, CulturalProfile, EnvironmentSettings

class Worker:
    """
    Represents a single worker agent in the simulation.
    This class models the dynamic affective state and behavior of a worker,
    grounding the simulation in principles of bounded rationality and individual variability.
    """
    def __init__(self, worker_id: int, config: Config):
        self.id = worker_id
        self.config = config
        self.profile = config.agent_profile
        self.cultural_profile = config.cultural_profile

        # --- Dynamic State Variables ---
        # Affective state vector: [stress, engagement, frustration]
        # Workers genuinely differ: individual baselines drawn from +-0.15
        # around the population default, giving the equilibrium distribution
        # meaningful between-worker spread and realistic tails (upper/lower
        # terciles are stable and largely between-worker).
        self.baseline_state = np.array(self.profile.baseline_affect) + np.random.uniform(-0.15, 0.15, 3)
        np.clip(self.baseline_state, 0.0, 1.0, out=self.baseline_state)
        self.affective_state = self.baseline_state.copy()
        # Cultural display-rule factor: suppresses/attenuates the *observable expression*
        # of affect (sensor channels), NOT the latent affective state itself.
        self.expression_factor = config.cultural_profile.verbal_frustration_tendency

        self.fatigue = 0.0
        self.task_complexity = 0.5 # Current task complexity
        self.current_task_start_time = None

        # --- Performance Metrics ---
        self.quality_error_rate = 0.05 * (1.5 - self.profile.skill_level) # Inversely proportional to skill
        self.task_completion_variance = 0.1

        # --- Data Generation History ---
        self.history = []

    def work(self, current_time: datetime, task_complexity: float, duration_hours: float = 1.0):
        """
        Simulates the worker performing a task for a given duration.
        This is the core method where affective state is updated based on workload.
        """
        self.task_complexity = task_complexity
        self.current_task_start_time = current_time

        # 1. Update Fatigue (operationalizes a key human factor)
        self.fatigue += self.profile.fatigue_rate_per_hour * duration_hours
        self.fatigue = min(self.fatigue, 1.0)

        # 2. Update Affective State (the heart of the model)
        # Homeostatic relaxation toward the individual baseline: a 30% hourly
        # pull keeps every state at a stable, worker-specific equilibrium
        # (net forcing per hour is ~0.06, so equilibrium = baseline + forcing/k;
        # with k=0.10 the equilibrium saturates near 1.0 and the population
        # drifts for weeks — a genuine non-stationarity we fixed).
        self.affective_state += (self.baseline_state - self.affective_state) * 0.30

        # Stress increases with task complexity and fatigue, mitigated by resilience
        stress_increase = (task_complexity * 0.15) + (self.fatigue * 0.2) - (self.profile.stress_resilience * 0.05)
        self.affective_state[0] += stress_increase

        # Engagement is affected by task match and fatigue
        # High skill workers are more engaged with complex tasks (STS principle)
        if self.profile.skill_level > 0.7 and task_complexity > 0.7:
            self.affective_state[1] += 0.10 # "Flow state"
        else:
            self.affective_state[1] -= 0.02 + self.fatigue * 0.2

        # Frustration increases with errors and fatigue
        frustration_increase = (self.quality_error_rate * 0.4) + (self.fatigue * 0.2)
        self.affective_state[2] += frustration_increase

        # Bounded state space
        np.clip(self.affective_state, 0.0, 1.0, out=self.affective_state)

        # 3. Update Performance Metrics (linking affect to business KPIs)
        # Stress and frustration increase error rate (affect-to-KPI pathway)
        self.quality_error_rate = 0.05 * (1.5 - self.profile.skill_level) * (1 + self.affective_state[0] * 1.2 + self.affective_state[2] * 0.8)
        self.quality_error_rate = np.clip(self.quality_error_rate, 0.0, 0.5) # Cap at 50%

        # Engagement decreases task completion variance (calibrated coupling)
        self.task_completion_variance = 0.5 * (1.5 - self.affective_state[1])
        
        # 4. Log state for this time step
        self._log_state(current_time)

    def react_to_event(self, event_type: str, event_data: Dict[str, Any], current_time: datetime):
        """
        Allows the agent to react to external events (e.g., new automation announcement, machine breakdown).
        This models the impact of organizational decisions on worker affect.
        """
        if event_type == "management_announcement":
            # Example: Announcement of a new automation project
            # Impact is modulated by cultural sensitivity to hierarchy
            impact = event_data.get('impact', 0.2) * self.cultural_profile.hierarchical_sensitivity
            self.affective_state[0] += impact # Stress increases
            self.affective_state[1] -= impact * 0.5 # Engagement might decrease
        elif event_type == "machine_breakdown":
            # Frustration spikes due to process friction
            self.affective_state[2] += 0.3
        
        self._log_state(current_time, event=event_type)

    def take_break(self, duration_minutes: int):
        """Simulates the effect of taking a break."""
        self.fatigue = max(0, self.fatigue - duration_minutes * 0.01)
        self.affective_state[0] = max(0, self.affective_state[0] - 0.1) # Stress reduces

    def _log_state(self, current_time: datetime, event: str = None):
        """Logs the current state of the agent for data generation."""
        log_entry = {
            'timestamp': current_time,
            'worker_id': self.id,
            'stress': self.affective_state[0],
            'engagement': self.affective_state[1],
            'frustration': self.affective_state[2],
            'fatigue': self.fatigue,
            'task_complexity': self.task_complexity,
            'error_rate': self.quality_error_rate,
            'event': event
        }
        self.history.append(log_entry)

    def get_sensor_data(self, timestamp: datetime) -> Dict[str, Any]:
        """
        Generates multi-modal sensor data for the SMAKT Layer 1.

        Signal model (calibrated partial observability): each affective state
        is expressed, weakly and differently, in EACH modality group, so that
        no single modality is sufficient and fusion has a genuine basis:
          stress      -> lower HRV (strong), shorter transcripts, more negative
                         words, higher completion variance, more help presses
          engagement  -> longer transcripts (strong), lower completion variance
          frustration -> negative words + help presses (strong), error corrections
        Cultural display rules attenuate the *expressed* (behavioural/verbal)
        components only; latent states and the physiological (HRV) signal are
        unchanged.
        """
        expressed_frustration = self.affective_state[2] * self.expression_factor
        expressed_stress = self.affective_state[0] * self.expression_factor
        expressed_engagement = self.affective_state[1]  # behavioural, not culturally attenuated

        # 1. MES Interaction Logs
        mes_log = {
            'timestamp': timestamp,
            'worker_id': self.id,
            'task_completion_variance': max(0.0, 0.35 * (1.5 - expressed_engagement)
                                            + 0.30 * expressed_stress
                                            + np.random.normal(0, 0.02)),
            'help_button_presses': int(np.random.rand() < max(0.0,
                expressed_frustration - 0.15 + 0.35 * expressed_stress)),
            'error_corrections': np.random.poisson(self.quality_error_rate * 15
                                                   + 2.0 * expressed_frustration),
        }

        # 2. Speech Analysis (simulated)
        negative_words = np.random.poisson(15.0 * expressed_frustration
                                           + 8.0 * expressed_stress)
        speech_data = {
            'timestamp': timestamp,
            'worker_id': self.id,
            'transcript_length': int(max(10, np.random.randint(30, 80)
                                         + expressed_engagement * 120
                                         - 50.0 * expressed_stress)),
            'negative_word_count': negative_words
        }

        # 3. Wearable HRV (simulated): stress and fatigue decrease HRV; mild
        # frustration-related arousal decreases it slightly further
        base_hrv = 60
        hrv_reduction = (self.affective_state[0] * 30) + (self.fatigue * 25) \
            + (8.0 * self.affective_state[2] * self.expression_factor)
        wearable_data = {
            'timestamp': timestamp,
            'worker_id': self.id,
            'hrv_rmssd': max(20, base_hrv - hrv_reduction + np.random.normal(0, 4))
        }

        return {
            'mes_logs': mes_log,
            'speech_analysis': speech_data,
            'wearable_hrv': wearable_data
        }

class Environment:
    """
    Represents the industrial environment (the factory floor).
    It orchestrates tasks, introduces disturbances, and manages the overall context.
    """
    def __init__(self, config: Config):
        self.config = config
        self.settings = config.environment
        self.current_time = datetime.now()
        self.workers: List[Worker] = []
        self.events_log = []

        # Initialize workers
        for i in range(config.simulation.num_workers):
            self.workers.append(Worker(worker_id=i, config=config))

    def step(self, duration_hours: float = 1.0):
        """
        Advances the simulation by one time step (e.g., one hour).
        This is the main simulation loop driver.
        """
        self.current_time += timedelta(hours=duration_hours)

        # 1. Assign tasks to workers
        for worker in self.workers:
            # Task complexity can vary based on time of day, worker skill, etc.
            task_complexity = np.random.beta(2, 5) # Skewed towards simpler tasks
            if worker.profile.skill_level > 0.8:
                task_complexity += 0.2 # Give complex tasks to skilled workers
            
            task_complexity = np.clip(task_complexity, 0.1, 1.0)
            worker.work(self.current_time, task_complexity, duration_hours)

        # 2. Introduce random disturbances
        if random.random() < self.settings.daily_disturbance_probability:
            self._introduce_disturbance()

        # 3. Simulate shift changes (fatigue reset)
        if self.current_time.hour % self.settings.shift_duration_hours == 0:
            for worker in self.workers:
                worker.fatigue = max(0, worker.fatigue - 0.5) # Partial recovery

    def _introduce_disturbance(self):
        """Randomly introduces a disruptive event into the environment."""
        event_type = random.choice(['machine_breakdown', 'quality_alert', 'management_announcement'])
        
        affected_workers = random.sample(self.workers, k=random.randint(5, 20))
        
        event_log = {
            'timestamp': self.current_time,
            'event_type': event_type,
            'affected_workers': [w.id for w in affected_workers]
        }
        self.events_log.append(event_log)

        event_data = {'impact': random.uniform(0.1, 0.3)}
        for worker in affected_workers:
            worker.react_to_event(event_type, event_data, self.current_time)
        
        print(f"[{self.current_time}] DISTURBANCE: {event_type} affecting {len(affected_workers)} workers.")

    def get_all_sensor_data(self) -> pd.DataFrame:
        """
        Collects all sensor data from all workers at the current time step.
        This is the primary input for the SMAKT sensing layer.
        """
        all_data = []
        for worker in self.workers:
            sensor_data = worker.get_sensor_data(self.current_time)
            # Flatten the nested dictionary for easier processing
            flat_data = {
                'timestamp': self.current_time,
                'worker_id': worker.id,
                **sensor_data['mes_logs'],
                **{f"speech_{k}": v for k, v in sensor_data['speech_analysis'].items()},
                **{f"wearable_{k}": v for k, v in sensor_data['wearable_hrv'].items()}
            }
            all_data.append(flat_data)
        
        return pd.DataFrame(all_data)

# --- Main execution for testing ---
if __name__ == '__main__':
    # Load configuration
    from config import get_scenario
    sim_config = get_scenario("small_factory") # Use a scenario for testing

    # Initialize environment
    factory = Environment(config=sim_config)
    
    print(f"Starting simulation for {sim_config.simulation.num_workers} workers...")
    print("-" * 30)

    # Run simulation for a few steps
    for _ in range(24 * 3): # Simulate 3 days (24 steps per day)
        factory.step(duration_hours=1.0)

    # --- Analysis and Demonstration ---
    print("\nSimulation finished. Generating sample outputs...")

    # 1. Show final state of a single worker
    sample_worker = factory.workers[0]
    print(f"\nFinal state of Worker {sample_worker.id}:")
    print(f"  Stress: {sample_worker.affective_state[0]:.2f}")
    print(f"  Engagement: {sample_worker.affective_state[1]:.2f}")
    print(f"  Frustration: {sample_worker.affective_state[2]:.2f}")
    print(f"  Fatigue: {sample_worker.fatigue:.2f}")
    print(f"  Error Rate: {sample_worker.quality_error_rate:.2f}")

    # 2. Show sample sensor data for that worker
    sample_sensor_data = sample_worker.get_sensor_data(factory.current_time)
    print(f"\nSample sensor data for Worker {sample_worker.id}:")
    for modality, data in sample_sensor_data.items():
        print(f"  {modality}: {data}")

    # 3. Show the aggregated sensor data for all workers (input for Layer 1)
    all_sensor_df = factory.get_all_sensor_data()
    print(f"\nAggregated sensor data for all workers (last step):")
    print(all_sensor_df.head())

    # 4. Show event log
    print(f"\nEvent Log:")
    for event in factory.events_log:
        print(f"  - {event}")
