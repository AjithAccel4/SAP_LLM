"""
Model Registry for Centralized Model Management.

Provides:
- Model versioning with semantic versions
- Metadata storage and querying
- Champion/Challenger model management
- Model lifecycle tracking
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from sap_llm.models.registry.model_version import ModelVersion, INITIAL_VERSION
from sap_llm.models.registry.storage_backend import StorageBackend, LocalStorageBackend

logger = logging.getLogger(__name__)


class ModelStatus:
    """Model status constants."""
    REGISTERED = "registered"      # Model registered but not active
    CHALLENGER = "challenger"      # Under A/B testing
    CHAMPION = "champion"          # Current production model
    ARCHIVED = "archived"          # Previous champion, kept for rollback
    DEPRECATED = "deprecated"      # Old model, can be deleted


class ModelRegistry:
    """
    Centralized model registry with versioning and metadata management.

    Features:
    - Model versioning (semantic versioning)
    - Champion/Challenger management
    - Metadata storage in SQLite
    - Model artifact storage via pluggable backend
    - Automated rollback capability
    """

    def __init__(
        self,
        storage_backend: Optional[StorageBackend] = None,
        db_path: str = "./model_registry/registry.db"
    ):
        """
        Initialize model registry.

        Args:
            storage_backend: Storage backend for model artifacts
            db_path: Path to SQLite database for metadata
        """
        self.storage_backend = storage_backend or LocalStorageBackend()

        # Initialize database
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

        logger.info(f"ModelRegistry initialized (db={self.db_path})")

    def _init_database(self):
        """Initialize SQLite database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create models table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                model_type TEXT NOT NULL,
                status TEXT NOT NULL,
                metrics TEXT,
                metadata TEXT,
                created_at TIMESTAMP NOT NULL,
                promoted_at TIMESTAMP,
                demoted_at TIMESTAMP,
                UNIQUE(name, version)
            )
        """)

        # Create promotion history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS promotion_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                from_status TEXT NOT NULL,
                to_status TEXT NOT NULL,
                reason TEXT,
                metrics_before TEXT,
                metrics_after TEXT,
                timestamp TIMESTAMP NOT NULL,
                FOREIGN KEY (model_id) REFERENCES models(id)
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_status ON models(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_type ON models(model_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON models(created_at)")

        conn.commit()
        conn.close()

        logger.info("Database schema initialized")

    def register_model(
        self,
        model: torch.nn.Module,
        name: str,
        model_type: str,
        version: Optional[ModelVersion] = None,
        metrics: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        auto_increment_version: bool = True
    ) -> str:
        """
        Register a new model in the registry.

        Args:
            model: PyTorch model
            name: Model name (e.g., "vision_encoder")
            model_type: Model type for categorization
            version: Model version (auto-generated if None)
            metrics: Performance metrics
            metadata: Additional metadata
            auto_increment_version: Auto-increment from latest version

        Returns:
            Model ID
        """
        # Determine version
        if version is None:
            if auto_increment_version:
                latest = self.get_latest_version(name)
                version = latest.increment_patch() if latest else INITIAL_VERSION
            else:
                version = INITIAL_VERSION

        # Generate model ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"{name}_{version}_{timestamp}"

        logger.info(f"Registering model: {model_id}")

        # Prepare metadata
        full_metadata = {
            "name": name,
            "version": str(version),
            "model_type": model_type,
            "architecture": model.__class__.__name__,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            **(metadata or {})
        }

        # Save model artifacts
        try:
            storage_path = self.storage_backend.save_model(
                model=model,
                model_id=model_id,
                metadata=full_metadata
            )

            # Save metadata to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO models (
                    id, name, version, model_type, status,
                    metrics, metadata, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_id,
                name,
                str(version),
                model_type,
                ModelStatus.REGISTERED,
                json.dumps(metrics or {}),
                json.dumps(full_metadata),
                datetime.now()
            ))

            conn.commit()
            conn.close()

            logger.info(f"Model registered successfully: {model_id}")
            return model_id

        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            # Cleanup on failure
            self.storage_backend.delete_model(model_id)
            raise

    def get_model(self, model_id: str) -> Dict[str, Any]:
        """
        Get model metadata.

        Args:
            model_id: Model identifier

        Returns:
            Model metadata dictionary
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM models WHERE id = ?", (model_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            raise ValueError(f"Model not found: {model_id}")

        return dict(row)

    def get_champion(self, model_type: str) -> Optional[Dict[str, Any]]:
        """
        Get current champion model for a model type.

        Args:
            model_type: Model type (e.g., "vision_encoder")

        Returns:
            Champion model metadata or None
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM models
            WHERE model_type = ? AND status = ?
            ORDER BY promoted_at DESC
            LIMIT 1
        """, (model_type, ModelStatus.CHAMPION))

        row = cursor.fetchone()
        conn.close()

        return dict(row) if row else None

    def promote_to_champion(
        self,
        model_id: str,
        reason: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None
    ):
        """
        Promote model to champion status.

        Automatically demotes current champion to archived.

        Args:
            model_id: Model ID to promote
            reason: Reason for promotion
            metrics: Updated metrics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Get model to be promoted
            cursor.execute("SELECT * FROM models WHERE id = ?", (model_id,))
            model = cursor.fetchone()

            if not model:
                raise ValueError(f"Model not found: {model_id}")

            model_type = model[3]  # model_type column

            # Demote current champion to archived
            cursor.execute("""
                UPDATE models
                SET status = ?, demoted_at = ?
                WHERE model_type = ? AND status = ?
            """, (ModelStatus.ARCHIVED, datetime.now(), model_type, ModelStatus.CHAMPION))

            demoted_count = cursor.rowcount
            if demoted_count > 0:
                logger.info(f"Demoted {demoted_count} champion(s) to archived")

            # Promote new champion
            now = datetime.now()
            cursor.execute("""
                UPDATE models
                SET status = ?, promoted_at = ?, metrics = ?
                WHERE id = ?
            """, (
                ModelStatus.CHAMPION,
                now,
                json.dumps(metrics) if metrics else model[5],  # metrics column
                model_id
            ))

            # Record promotion history
            cursor.execute("""
                INSERT INTO promotion_history (
                    model_id, from_status, to_status, reason,
                    metrics_after, timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                model_id,
                model[4],  # old status
                ModelStatus.CHAMPION,
                reason,
                json.dumps(metrics) if metrics else None,
                now
            ))

            conn.commit()
            logger.info(f"Model promoted to champion: {model_id}")

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to promote model: {e}")
            raise
        finally:
            conn.close()

    def promote_to_challenger(self, model_id: str):
        """
        Promote model to challenger status for A/B testing.

        Args:
            model_id: Model ID to promote
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                UPDATE models
                SET status = ?
                WHERE id = ?
            """, (ModelStatus.CHALLENGER, model_id))

            if cursor.rowcount == 0:
                raise ValueError(f"Model not found: {model_id}")

            # Record in history
            cursor.execute("""
                INSERT INTO promotion_history (
                    model_id, from_status, to_status, timestamp
                )
                VALUES (?, ?, ?, ?)
            """, (model_id, ModelStatus.REGISTERED, ModelStatus.CHALLENGER, datetime.now()))

            conn.commit()
            logger.info(f"Model promoted to challenger: {model_id}")

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to promote to challenger: {e}")
            raise
        finally:
            conn.close()

    def rollback_to_previous_champion(
        self,
        model_type: str,
        reason: str
    ) -> str:
        """
        Rollback to previous archived champion.

        Args:
            model_type: Model type
            reason: Reason for rollback

        Returns:
            Model ID of restored champion
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            # Get current champion
            cursor.execute("""
                SELECT * FROM models
                WHERE model_type = ? AND status = ?
            """, (model_type, ModelStatus.CHAMPION))

            current_champion = cursor.fetchone()

            # Get previous champion (most recently archived)
            cursor.execute("""
                SELECT * FROM models
                WHERE model_type = ? AND status = ?
                ORDER BY demoted_at DESC
                LIMIT 1
            """, (model_type, ModelStatus.ARCHIVED))

            previous_champion = cursor.fetchone()

            if not previous_champion:
                raise ValueError(f"No archived champion found for rollback: {model_type}")

            previous_id = previous_champion[0]  # id column

            # Demote current champion
            if current_champion:
                cursor.execute("""
                    UPDATE models
                    SET status = ?, demoted_at = ?
                    WHERE id = ?
                """, (ModelStatus.DEPRECATED, datetime.now(), current_champion[0]))

            # Restore previous champion
            cursor.execute("""
                UPDATE models
                SET status = ?, promoted_at = ?
                WHERE id = ?
            """, (ModelStatus.CHAMPION, datetime.now(), previous_id))

            # Record rollback
            cursor.execute("""
                INSERT INTO promotion_history (
                    model_id, from_status, to_status, reason, timestamp
                )
                VALUES (?, ?, ?, ?, ?)
            """, (
                previous_id,
                ModelStatus.ARCHIVED,
                ModelStatus.CHAMPION,
                f"ROLLBACK: {reason}",
                datetime.now()
            ))

            conn.commit()
            logger.warning(f"Rolled back to previous champion: {previous_id}. Reason: {reason}")

            return previous_id

        except Exception as e:
            conn.rollback()
            logger.error(f"Rollback failed: {e}")
            raise
        finally:
            conn.close()

    def get_latest_version(self, name: str) -> Optional[ModelVersion]:
        """
        Get latest version for a model name.

        Args:
            name: Model name

        Returns:
            Latest ModelVersion or None
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT version FROM models
            WHERE name = ?
            ORDER BY created_at DESC
            LIMIT 1
        """, (name,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return ModelVersion.from_string(row[0])
        return None

    def list_models(
        self,
        model_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List models with optional filters.

        Args:
            model_type: Filter by model type
            status: Filter by status
            limit: Maximum number of results

        Returns:
            List of model metadata
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM models WHERE 1=1"
        params = []

        if model_type:
            query += " AND model_type = ?"
            params.append(model_type)

        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_promotion_history(
        self,
        model_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get promotion history.

        Args:
            model_id: Optional filter by model ID
            limit: Maximum results

        Returns:
            List of promotion records
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if model_id:
            cursor.execute("""
                SELECT * FROM promotion_history
                WHERE model_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (model_id, limit))
        else:
            cursor.execute("""
                SELECT * FROM promotion_history
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def delete_model(self, model_id: str, force: bool = False):
        """
        Delete model from registry.

        Args:
            model_id: Model ID
            force: Force delete even if champion
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Check if champion
            cursor.execute("SELECT status FROM models WHERE id = ?", (model_id,))
            row = cursor.fetchone()

            if not row:
                raise ValueError(f"Model not found: {model_id}")

            if row[0] == ModelStatus.CHAMPION and not force:
                raise ValueError(
                    f"Cannot delete champion model: {model_id}. "
                    f"Use force=True to override."
                )

            # Delete from database
            cursor.execute("DELETE FROM models WHERE id = ?", (model_id,))
            cursor.execute("DELETE FROM promotion_history WHERE model_id = ?", (model_id,))

            conn.commit()

            # Delete artifacts
            self.storage_backend.delete_model(model_id)

            logger.info(f"Model deleted: {model_id}")

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to delete model: {e}")
            raise
        finally:
            conn.close()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Statistics dictionary
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        # Count by status
        cursor.execute("""
            SELECT status, COUNT(*) FROM models GROUP BY status
        """)
        stats["by_status"] = dict(cursor.fetchall())

        # Count by model type
        cursor.execute("""
            SELECT model_type, COUNT(*) FROM models GROUP BY model_type
        """)
        stats["by_type"] = dict(cursor.fetchall())

        # Total models
        cursor.execute("SELECT COUNT(*) FROM models")
        stats["total_models"] = cursor.fetchone()[0]

        # Total promotions
        cursor.execute("SELECT COUNT(*) FROM promotion_history")
        stats["total_promotions"] = cursor.fetchone()[0]

        conn.close()

        # Storage stats
        stats["storage_size_bytes"] = self.storage_backend.get_storage_size()
        stats["storage_size_mb"] = stats["storage_size_bytes"] / (1024 * 1024)

        return stats
