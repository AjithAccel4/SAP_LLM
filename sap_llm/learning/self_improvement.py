"""
Self-Improvement Pipeline

Orchestrates automatic model improvement through nightly retraining,
synthetic data generation, curriculum learning, ensemble learning,
and meta-learning. Operates without human intervention.
"""

import json
import os
import pickle
import random
import shutil
from collections import defaultdict
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score

from sap_llm.learning.adaptive_learning import AdaptiveLearningEngine
from sap_llm.learning.feedback_loop import FeedbackLoopSystem, ModelVersion
from sap_llm.learning.knowledge_augmentation import KnowledgeAugmentationEngine
from sap_llm.learning.online_learning import OnlineLearningEngine
from sap_llm.pmg.graph_client import ProcessMemoryGraph
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class TrainingStrategy(Enum):
    """Training strategy types."""

    FULL_RETRAIN = "full_retrain"
    INCREMENTAL = "incremental"
    TRANSFER = "transfer"
    FEW_SHOT = "few_shot"
    ENSEMBLE = "ensemble"
    META = "meta"


class TrainingPriority(Enum):
    """Training priority levels."""

    CRITICAL = "critical"  # Immediate
    HIGH = "high"  # Next scheduled run
    MEDIUM = "medium"  # Nightly
    LOW = "low"  # Weekly


class TrainingJob:
    """Training job specification."""

    def __init__(
        self,
        job_id: str,
        doc_type: str,
        strategy: TrainingStrategy,
        priority: TrainingPriority,
        training_data: List[Dict[str, Any]],
        validation_data: Optional[List[Dict[str, Any]]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ):
        self.job_id = job_id
        self.doc_type = doc_type
        self.strategy = strategy
        self.priority = priority
        self.training_data = training_data
        self.validation_data = validation_data or []
        self.hyperparameters = hyperparameters or {}
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.status = "pending"
        self.metrics: Dict[str, float] = {}
        self.error: Optional[str] = None


class CurriculumLearner:
    """Curriculum learning for complex document types."""

    def __init__(self):
        self.curricula: Dict[str, List[Dict[str, Any]]] = {}

    def create_curriculum(
        self,
        doc_type: str,
        training_samples: List[Dict[str, Any]],
        num_stages: int = 3,
    ) -> List[List[Dict[str, Any]]]:
        """
        Create curriculum from easy to hard examples.

        Args:
            doc_type: Document type
            training_samples: All training samples
            num_stages: Number of curriculum stages

        Returns:
            List of training batches (easy -> hard)
        """
        logger.info(f"Creating curriculum for {doc_type} ({num_stages} stages)")

        # Score difficulty of each sample
        scored_samples = []
        for sample in training_samples:
            difficulty = self._assess_difficulty(sample)
            scored_samples.append((difficulty, sample))

        # Sort by difficulty
        scored_samples.sort(key=lambda x: x[0])

        # Split into stages
        samples_per_stage = len(scored_samples) // num_stages
        curriculum = []

        for stage in range(num_stages):
            start_idx = stage * samples_per_stage
            end_idx = start_idx + samples_per_stage if stage < num_stages - 1 else len(scored_samples)

            stage_samples = [s for _, s in scored_samples[start_idx:end_idx]]
            curriculum.append(stage_samples)

            logger.info(
                f"Stage {stage + 1}: {len(stage_samples)} samples "
                f"(difficulty: {scored_samples[start_idx][0]:.2f} - "
                f"{scored_samples[end_idx - 1][0]:.2f})"
            )

        self.curricula[doc_type] = curriculum
        return curriculum

    def _assess_difficulty(self, sample: Dict[str, Any]) -> float:
        """
        Assess difficulty of training sample.

        Factors:
        - Number of fields (more = harder)
        - Field complexity (nested structures)
        - Value variance
        - Label ambiguity
        """
        difficulty = 0.0

        # Field count
        num_fields = len(sample.get('features', {}))
        difficulty += num_fields * 0.1

        # Nested structures
        features = sample.get('features', {})
        for value in features.values():
            if isinstance(value, (dict, list)):
                difficulty += 0.3

        # String length (proxy for complexity)
        total_length = sum(
            len(str(v)) for v in features.values()
        )
        difficulty += total_length / 1000.0

        # Confidence (inverse - low confidence = harder)
        confidence = sample.get('confidence', 0.5)
        difficulty += (1.0 - confidence)

        return min(difficulty, 10.0)  # Cap at 10


class EnsembleLearner:
    """Ensemble learning from multiple model versions."""

    def __init__(self):
        self.ensembles: Dict[str, VotingClassifier] = {}
        self.ensemble_models: Dict[str, List[Any]] = defaultdict(list)

    def create_ensemble(
        self,
        doc_type: str,
        models: List[Any],
        voting: str = 'soft',
    ) -> VotingClassifier:
        """
        Create ensemble from multiple models.

        Args:
            doc_type: Document type
            models: List of trained models
            voting: 'hard' or 'soft' voting

        Returns:
            Ensemble classifier
        """
        logger.info(f"Creating ensemble for {doc_type} with {len(models)} models")

        if len(models) < 2:
            logger.warning("Need at least 2 models for ensemble")
            return None

        # Create ensemble
        estimators = [(f"model_{i}", model) for i, model in enumerate(models)]

        ensemble = VotingClassifier(
            estimators=estimators,
            voting=voting,
        )

        self.ensembles[doc_type] = ensemble
        self.ensemble_models[doc_type] = models

        return ensemble

    def train_ensemble(
        self,
        doc_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Dict[str, float]:
        """Train ensemble on data."""
        if doc_type not in self.ensembles:
            return {"error": "No ensemble created"}

        logger.info(f"Training ensemble for {doc_type}")

        try:
            self.ensembles[doc_type].fit(X_train, y_train)

            # Calculate training accuracy
            y_pred = self.ensembles[doc_type].predict(X_train)
            accuracy = accuracy_score(y_train, y_pred)

            logger.info(f"Ensemble trained: accuracy={accuracy:.3f}")

            return {
                "accuracy": accuracy,
                "num_models": len(self.ensemble_models[doc_type]),
            }

        except Exception as e:
            logger.error(f"Ensemble training failed: {e}")
            return {"error": str(e)}


class MetaLearner:
    """Meta-learning for fast adaptation to new document types."""

    def __init__(self):
        self.meta_model = None
        self.meta_parameters: Dict[str, Any] = {}
        self.adaptation_history: List[Dict[str, Any]] = []

    def learn_meta_parameters(
        self,
        task_distributions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Learn meta-parameters from multiple task distributions.

        Args:
            task_distributions: List of task (doc_type) data

        Returns:
            Learned meta-parameters
        """
        logger.info(f"Learning meta-parameters from {len(task_distributions)} tasks")

        # Simplified meta-learning
        # In production: MAML, Reptile, or similar

        meta_params = {
            "initial_learning_rate": 0.001,
            "adaptation_steps": 5,
            "inner_loop_lr": 0.01,
        }

        # Analyze task distributions
        avg_samples = np.mean([len(task.get('samples', [])) for task in task_distributions])
        avg_complexity = np.mean([task.get('complexity', 1.0) for task in task_distributions])

        # Adjust meta-parameters based on analysis
        if avg_samples < 100:
            meta_params["adaptation_steps"] = 10
        if avg_complexity > 5.0:
            meta_params["initial_learning_rate"] = 0.0001

        self.meta_parameters = meta_params

        logger.info(f"Learned meta-parameters: {meta_params}")
        return meta_params

    def fast_adapt(
        self,
        new_task_data: List[Dict[str, Any]],
        num_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Quickly adapt to new task with few examples.

        Args:
            new_task_data: Small set of examples for new task
            num_steps: Number of adaptation steps

        Returns:
            Adapted model metrics
        """
        steps = num_steps or self.meta_parameters.get("adaptation_steps", 5)

        logger.info(f"Fast adapting to new task with {len(new_task_data)} examples ({steps} steps)")

        # Simulate adaptation
        # In production: perform actual gradient-based adaptation

        self.adaptation_history.append({
            "task_samples": len(new_task_data),
            "adaptation_steps": steps,
            "timestamp": datetime.now().isoformat(),
        })

        return {
            "adapted": True,
            "adaptation_steps": steps,
            "estimated_accuracy": 0.75 + (len(new_task_data) / 100.0),
        }


class SyntheticDataGenerator:
    """Generate synthetic data for data augmentation."""

    def __init__(self):
        self.generation_strategies = [
            "noise_injection",
            "template_based",
            "recombination",
            "edge_case_generation",
        ]

    def generate_synthetic_samples(
        self,
        doc_type: str,
        base_samples: List[Dict[str, Any]],
        num_samples: int = 1000,
        strategies: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate synthetic training samples.

        Args:
            doc_type: Document type
            base_samples: Base samples to augment
            num_samples: Number of synthetic samples
            strategies: Generation strategies to use

        Returns:
            Synthetic samples
        """
        logger.info(
            f"Generating {num_samples} synthetic samples for {doc_type} "
            f"from {len(base_samples)} base samples"
        )

        if not base_samples:
            return []

        strategies = strategies or self.generation_strategies
        synthetic = []

        samples_per_strategy = num_samples // len(strategies)

        for strategy in strategies:
            if strategy == "noise_injection":
                synthetic.extend(
                    self._noise_injection(base_samples, samples_per_strategy)
                )
            elif strategy == "template_based":
                synthetic.extend(
                    self._template_based(base_samples, samples_per_strategy)
                )
            elif strategy == "recombination":
                synthetic.extend(
                    self._recombination(base_samples, samples_per_strategy)
                )
            elif strategy == "edge_case_generation":
                synthetic.extend(
                    self._edge_cases(base_samples, samples_per_strategy)
                )

        logger.info(f"Generated {len(synthetic)} synthetic samples")
        return synthetic[:num_samples]

    def _noise_injection(
        self,
        base_samples: List[Dict[str, Any]],
        num_samples: int,
    ) -> List[Dict[str, Any]]:
        """Generate samples by injecting noise."""
        synthetic = []

        for _ in range(num_samples):
            base = random.choice(base_samples).copy()
            features = base.get('features', {}).copy()

            # Add noise to numeric fields
            for key, value in features.items():
                if isinstance(value, (int, float)):
                    noise = np.random.normal(0, 0.1 * abs(value))
                    features[key] = value + noise

            base['features'] = features
            base['synthetic'] = True
            base['generation_method'] = 'noise_injection'
            synthetic.append(base)

        return synthetic

    def _template_based(
        self,
        base_samples: List[Dict[str, Any]],
        num_samples: int,
    ) -> List[Dict[str, Any]]:
        """Generate samples using templates."""
        synthetic = []

        # Extract template structure from base samples
        template_structures = self._extract_templates(base_samples)

        for _ in range(num_samples):
            template = random.choice(template_structures)

            # Fill template with varied values
            sample = self._fill_template(template, base_samples)
            sample['synthetic'] = True
            sample['generation_method'] = 'template_based'
            synthetic.append(sample)

        return synthetic

    def _recombination(
        self,
        base_samples: List[Dict[str, Any]],
        num_samples: int,
    ) -> List[Dict[str, Any]]:
        """Generate samples by recombining features."""
        synthetic = []

        for _ in range(num_samples):
            # Combine features from multiple samples
            sample1 = random.choice(base_samples)
            sample2 = random.choice(base_samples)

            features = {}
            for key in sample1.get('features', {}).keys():
                # Randomly pick from sample1 or sample2
                source = random.choice([sample1, sample2])
                if key in source.get('features', {}):
                    features[key] = source['features'][key]

            synthetic.append({
                'features': features,
                'label': sample1.get('label'),
                'synthetic': True,
                'generation_method': 'recombination',
            })

        return synthetic

    def _edge_cases(
        self,
        base_samples: List[Dict[str, Any]],
        num_samples: int,
    ) -> List[Dict[str, Any]]:
        """Generate edge case samples."""
        synthetic = []

        for _ in range(num_samples):
            base = random.choice(base_samples).copy()
            features = base.get('features', {}).copy()

            # Create edge cases
            for key, value in features.items():
                if isinstance(value, (int, float)):
                    # Create extreme values
                    if random.random() < 0.5:
                        features[key] = value * 10  # Very large
                    else:
                        features[key] = value / 10  # Very small
                elif isinstance(value, str):
                    # Create edge case strings
                    if random.random() < 0.3:
                        features[key] = ""  # Empty
                    elif random.random() < 0.3:
                        features[key] = value * 3  # Repeated

            base['features'] = features
            base['synthetic'] = True
            base['generation_method'] = 'edge_case'
            synthetic.append(base)

        return synthetic

    def _extract_templates(
        self,
        samples: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Extract template structures from samples."""
        templates = []

        for sample in samples[:10]:  # Use first 10 as templates
            template = {
                'structure': list(sample.get('features', {}).keys()),
                'types': {
                    k: type(v).__name__
                    for k, v in sample.get('features', {}).items()
                },
            }
            templates.append(template)

        return templates

    def _fill_template(
        self,
        template: Dict[str, Any],
        base_samples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Fill template with values."""
        features = {}

        for key in template['structure']:
            # Find values from base samples
            values = []
            for sample in base_samples:
                if key in sample.get('features', {}):
                    values.append(sample['features'][key])

            if values:
                features[key] = random.choice(values)

        return {
            'features': features,
            'label': base_samples[0].get('label') if base_samples else 'UNKNOWN',
        }


class SelfImprovementPipeline:
    """
    Comprehensive self-improvement pipeline.

    Orchestrates:
    - Nightly model retraining
    - Synthetic data generation
    - Curriculum learning
    - Ensemble learning
    - Meta-learning
    - Automatic deployment
    """

    def __init__(
        self,
        pmg: ProcessMemoryGraph,
        online_learner: OnlineLearningEngine,
        feedback_system: FeedbackLoopSystem,
        adaptive_engine: AdaptiveLearningEngine,
        knowledge_augmentation: KnowledgeAugmentationEngine,
        schedule_time: time = time(2, 0),  # 2 AM
    ):
        """
        Initialize self-improvement pipeline.

        Args:
            pmg: Process Memory Graph
            online_learner: Online learning engine
            feedback_system: Feedback loop system
            adaptive_engine: Adaptive learning engine
            knowledge_augmentation: Knowledge augmentation engine
            schedule_time: Scheduled run time (default: 2 AM)
        """
        self.pmg = pmg
        self.online_learner = online_learner
        self.feedback_system = feedback_system
        self.adaptive_engine = adaptive_engine
        self.knowledge_augmentation = knowledge_augmentation
        self.schedule_time = schedule_time

        # Sub-components
        self.curriculum_learner = CurriculumLearner()
        self.ensemble_learner = EnsembleLearner()
        self.meta_learner = MetaLearner()
        self.synthetic_generator = SyntheticDataGenerator()

        # Training queue
        self.training_queue: List[TrainingJob] = []
        self.completed_jobs: List[TrainingJob] = []

        # Statistics
        self.improvement_stats: Dict[str, Any] = defaultdict(dict)
        self.last_run: Optional[datetime] = None

        logger.info("SelfImprovementPipeline initialized")

    def run_nightly_improvement(self) -> Dict[str, Any]:
        """
        Run nightly self-improvement cycle.

        Returns:
            Improvement statistics
        """
        logger.info("=" * 80)
        logger.info("STARTING NIGHTLY SELF-IMPROVEMENT CYCLE")
        logger.info("=" * 80)

        start_time = datetime.now()
        stats = {
            "start_time": start_time.isoformat(),
            "jobs_executed": 0,
            "models_improved": 0,
            "models_deployed": 0,
            "errors": [],
        }

        try:
            # 1. Collect feedback and identify improvement opportunities
            logger.info("Step 1: Analyzing feedback and performance...")
            opportunities = self._identify_improvement_opportunities()
            stats["improvement_opportunities"] = len(opportunities)

            # 2. Generate training data
            logger.info("Step 2: Generating training data...")
            training_data = self._prepare_training_data(opportunities)
            stats["training_samples_generated"] = sum(
                len(data.get('samples', [])) for data in training_data.values()
            )

            # 3. Create training jobs
            logger.info("Step 3: Creating training jobs...")
            jobs = self._create_training_jobs(opportunities, training_data)
            stats["jobs_created"] = len(jobs)

            # 4. Execute training jobs
            logger.info("Step 4: Executing training jobs...")
            results = self._execute_training_jobs(jobs)
            stats["jobs_executed"] = len(results)

            # 5. Evaluate and deploy improved models
            logger.info("Step 5: Evaluating and deploying models...")
            deployments = self._evaluate_and_deploy(results)
            stats["models_deployed"] = len(deployments)
            stats["deployments"] = deployments

            # 6. Update knowledge base
            logger.info("Step 6: Updating knowledge base...")
            kb_stats = self._update_knowledge_base()
            stats["knowledge_base"] = kb_stats

            # 7. Finalize A/B tests
            logger.info("Step 7: Finalizing A/B tests...")
            ab_results = self.feedback_system.finalize_ab_tests()
            stats["ab_tests_finalized"] = ab_results.get("finalized_count", 0)

        except Exception as e:
            logger.error(f"Nightly improvement failed: {e}")
            stats["errors"].append(str(e))

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        stats["end_time"] = end_time.isoformat()
        stats["duration_seconds"] = duration

        self.last_run = end_time
        self._save_improvement_stats(stats)

        logger.info("=" * 80)
        logger.info(f"NIGHTLY IMPROVEMENT COMPLETE (Duration: {duration:.1f}s)")
        logger.info(f"Jobs Executed: {stats['jobs_executed']}")
        logger.info(f"Models Deployed: {stats['models_deployed']}")
        logger.info("=" * 80)

        return stats

    def _identify_improvement_opportunities(self) -> List[Dict[str, Any]]:
        """Identify models that need improvement."""
        opportunities = []

        # Check performance degradation
        performance_summary = self.adaptive_engine.get_performance_summary()

        for doc_type, summary in performance_summary.items():
            if summary.get("needs_refresh"):
                opportunities.append({
                    "doc_type": doc_type,
                    "reason": "performance_degradation",
                    "priority": TrainingPriority.HIGH,
                    "details": summary,
                })

        # Check drift
        for doc_type in performance_summary.keys():
            drift_results = self.adaptive_engine.check_drift(doc_type)
            if drift_results.get("drift_detected"):
                opportunities.append({
                    "doc_type": doc_type,
                    "reason": "drift_detected",
                    "priority": TrainingPriority.HIGH,
                    "details": drift_results,
                })

        # Check feedback triggers
        for doc_type, triggers in self.feedback_system.retrain_triggers.items():
            if triggers:
                opportunities.append({
                    "doc_type": doc_type,
                    "reason": "feedback_threshold",
                    "priority": TrainingPriority.MEDIUM,
                    "details": {"trigger_count": len(triggers)},
                })

        logger.info(f"Identified {len(opportunities)} improvement opportunities")
        return opportunities

    def _prepare_training_data(
        self,
        opportunities: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Prepare training data for each opportunity."""
        training_data = {}

        for opp in opportunities:
            doc_type = opp["doc_type"]

            # Collect high-quality samples from PMG
            production_samples = self.knowledge_augmentation.training_data_generator.generate_from_production(
                doc_type=doc_type,
                days=30,
                min_confidence=0.9,
            )

            # Process feedback
            feedback_samples = self._extract_feedback_samples(doc_type)

            # Generate synthetic samples if needed
            all_samples = production_samples + feedback_samples

            if len(all_samples) < 1000:
                synthetic = self.synthetic_generator.generate_synthetic_samples(
                    doc_type=doc_type,
                    base_samples=all_samples,
                    num_samples=1000 - len(all_samples),
                )
                all_samples.extend(synthetic)

            training_data[doc_type] = {
                "samples": all_samples,
                "production_count": len(production_samples),
                "feedback_count": len(feedback_samples),
                "synthetic_count": len(all_samples) - len(production_samples) - len(feedback_samples),
            }

            logger.info(
                f"Prepared {len(all_samples)} training samples for {doc_type} "
                f"(prod: {len(production_samples)}, feedback: {len(feedback_samples)}, "
                f"synthetic: {len(all_samples) - len(production_samples) - len(feedback_samples)})"
            )

        return training_data

    def _extract_feedback_samples(self, doc_type: str) -> List[Dict[str, Any]]:
        """Extract training samples from feedback."""
        feedbacks = self.feedback_system.feedback_by_doc_type.get(doc_type, [])

        samples = []
        for feedback in feedbacks:
            if feedback.get("feedback_type") == "correction":
                features = feedback.get("metadata", {}).get("features", {})
                label = feedback.get("correct_value")

                if features and label:
                    samples.append({
                        "features": features,
                        "label": label,
                        "source": "user_feedback",
                    })

        return samples

    def _create_training_jobs(
        self,
        opportunities: List[Dict[str, Any]],
        training_data: Dict[str, Dict[str, Any]],
    ) -> List[TrainingJob]:
        """Create training jobs."""
        jobs = []

        for opp in opportunities:
            doc_type = opp["doc_type"]

            if doc_type not in training_data:
                continue

            data = training_data[doc_type]
            samples = data["samples"]

            # Determine training strategy
            if len(samples) < 100:
                strategy = TrainingStrategy.FEW_SHOT
            elif opp["reason"] == "drift_detected":
                strategy = TrainingStrategy.FULL_RETRAIN
            else:
                strategy = TrainingStrategy.INCREMENTAL

            job = TrainingJob(
                job_id=f"{doc_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                doc_type=doc_type,
                strategy=strategy,
                priority=opp["priority"],
                training_data=samples,
            )

            jobs.append(job)
            self.training_queue.append(job)

        # Sort by priority
        jobs.sort(key=lambda j: list(TrainingPriority).index(j.priority))

        logger.info(f"Created {len(jobs)} training jobs")
        return jobs

    def _execute_training_jobs(
        self,
        jobs: List[TrainingJob],
    ) -> List[TrainingJob]:
        """Execute training jobs."""
        results = []

        for job in jobs:
            logger.info(
                f"Executing job: {job.job_id} ({job.doc_type}, {job.strategy.value})"
            )

            job.status = "running"
            job.started_at = datetime.now()

            try:
                if job.strategy == TrainingStrategy.INCREMENTAL:
                    metrics = self._train_incremental(job)
                elif job.strategy == TrainingStrategy.FEW_SHOT:
                    metrics = self._train_few_shot(job)
                elif job.strategy == TrainingStrategy.FULL_RETRAIN:
                    metrics = self._train_full(job)
                elif job.strategy == TrainingStrategy.ENSEMBLE:
                    metrics = self._train_ensemble(job)
                else:
                    metrics = self._train_full(job)

                job.metrics = metrics
                job.status = "completed"
                job.completed_at = datetime.now()

                results.append(job)

            except Exception as e:
                logger.error(f"Job {job.job_id} failed: {e}")
                job.status = "failed"
                job.error = str(e)
                job.completed_at = datetime.now()

            self.completed_jobs.append(job)

        return results

    def _train_incremental(self, job: TrainingJob) -> Dict[str, float]:
        """Train using incremental learning."""
        samples = [(s['features'], s['label']) for s in job.training_data]

        result = self.online_learner.batch_update(
            doc_type=job.doc_type,
            samples=samples,
        )

        return {
            "accuracy": result.get("accuracy", 0.0),
            "samples_processed": result.get("samples_processed", 0),
        }

    def _train_few_shot(self, job: TrainingJob) -> Dict[str, float]:
        """Train using few-shot learning."""
        samples = [(s['features'], s['label']) for s in job.training_data]

        result = self.online_learner.few_shot_learning(
            doc_type=job.doc_type,
            examples=samples,
        )

        return {
            "accuracy": result.get("accuracy", 0.0),
            "num_examples": result.get("num_examples", 0),
        }

    def _train_full(self, job: TrainingJob) -> Dict[str, float]:
        """Full retraining."""
        # Use curriculum learning for complex doc types
        curriculum = self.curriculum_learner.create_curriculum(
            doc_type=job.doc_type,
            training_samples=job.training_data,
            num_stages=3,
        )

        # Train through curriculum stages
        final_accuracy = 0.0

        for stage_idx, stage_samples in enumerate(curriculum):
            logger.info(f"Training stage {stage_idx + 1}/{len(curriculum)}")

            samples = [(s['features'], s['label']) for s in stage_samples]
            result = self.online_learner.batch_update(
                doc_type=job.doc_type,
                samples=samples,
            )

            final_accuracy = result.get("accuracy", 0.0)

        return {
            "accuracy": final_accuracy,
            "curriculum_stages": len(curriculum),
        }

    def _train_ensemble(self, job: TrainingJob) -> Dict[str, float]:
        """Train ensemble model."""
        # Simplified - in production, train multiple diverse models
        return {
            "accuracy": 0.0,
            "ensemble_size": 0,
        }

    def _evaluate_and_deploy(
        self,
        completed_jobs: List[TrainingJob],
    ) -> List[Dict[str, Any]]:
        """Evaluate trained models and deploy if improved."""
        deployments = []

        for job in completed_jobs:
            if job.status != "completed":
                continue

            doc_type = job.doc_type
            new_accuracy = job.metrics.get("accuracy", 0.0)

            # Get baseline performance
            baseline_summary = self.adaptive_engine.get_performance_summary(doc_type)
            baseline_accuracy = baseline_summary.get(doc_type, {}).get("metrics", {}).get("accuracy", 0.0)

            improvement = new_accuracy - baseline_accuracy

            logger.info(
                f"Model evaluation for {doc_type}: "
                f"baseline={baseline_accuracy:.3f}, new={new_accuracy:.3f}, "
                f"improvement={improvement:.3f}"
            )

            # Deploy if improvement
            if improvement > 0.01 or new_accuracy > 0.9:
                # Create new version
                version = self.feedback_system.create_model_version(
                    doc_type=doc_type,
                    metrics=job.metrics,
                    parent_version=self.feedback_system.active_versions.get(doc_type),
                )

                # Start A/B test if enabled
                if self.feedback_system.ab_test_enabled and improvement < 0.05:
                    # Small improvement - A/B test
                    champion_version = self.feedback_system.active_versions.get(doc_type, "v1.0.0")

                    ab_test = self.feedback_system.start_ab_test(
                        test_name=f"{doc_type}_improvement",
                        doc_type=doc_type,
                        champion_version=champion_version,
                        challenger_version=version.version_id,
                        duration_hours=24,
                    )

                    deployments.append({
                        "doc_type": doc_type,
                        "version": version.version_id,
                        "deployment_type": "ab_test",
                        "ab_test_id": ab_test["test_id"],
                        "improvement": improvement,
                    })

                else:
                    # Large improvement - direct deploy
                    self.feedback_system.deploy_version(doc_type, version.version_id)

                    deployments.append({
                        "doc_type": doc_type,
                        "version": version.version_id,
                        "deployment_type": "direct",
                        "improvement": improvement,
                    })

                # Clear refresh triggers
                self.adaptive_engine.clear_refresh_triggers(doc_type)

                logger.info(f"Deployed {doc_type}/{version.version_id}")

        return deployments

    def _update_knowledge_base(self) -> Dict[str, Any]:
        """Update knowledge base from recent data."""
        doc_types = ["PURCHASE_ORDER", "INVOICE", "GOODS_RECEIPT"]

        stats = self.knowledge_augmentation.build_knowledge_base(
            doc_types=doc_types,
            days=7,
        )

        return stats

    def _save_improvement_stats(self, stats: Dict[str, Any]):
        """Save improvement statistics."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.improvement_stats[timestamp] = stats

        # Also save to file
        stats_dir = "/tmp/sap_llm/improvement_stats"
        os.makedirs(stats_dir, exist_ok=True)

        stats_file = f"{stats_dir}/stats_{timestamp}.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Saved improvement stats to {stats_file}")

    def get_improvement_history(
        self,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """Get improvement history."""
        cutoff = datetime.now() - timedelta(days=days)

        history = []
        for timestamp, stats in self.improvement_stats.items():
            run_time = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
            if run_time > cutoff:
                history.append(stats)

        return sorted(history, key=lambda x: x.get("start_time", ""))
