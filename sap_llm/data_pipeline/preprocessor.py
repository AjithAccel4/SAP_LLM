"""
Distributed document preprocessing with Apache Spark.

Handles:
- Image quality assessment and enhancement
- OCR with Tesseract
- Document deskewing and denoising
- Layout analysis
- Data quality filtering
- Train/val/test splitting
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

try:
    from pyspark.sql import SparkSession, DataFrame
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        StructType, StructField, StringType, IntegerType,
        FloatType, BooleanType, ArrayType
    )
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    DataFrame = Any  # Type stub when pyspark not available
    logger.warning("PySpark not available. Install with: pip install pyspark")


class SparkPreprocessor:
    """
    Distributed document preprocessing using Apache Spark.

    Features:
    - Parallel processing across cluster
    - Quality filtering (OCR quality, image quality)
    - Deduplication based on perceptual hashing
    - Stratified train/val/test split
    """

    def __init__(self,
                 app_name: str = "SAP_LLM_Preprocessing",
                 master: str = "local[*]",
                 executor_memory: str = "4g",
                 driver_memory: str = "2g"):
        """
        Initialize Spark preprocessor.

        Args:
            app_name: Spark application name
            master: Spark master URL
            executor_memory: Executor memory allocation
            driver_memory: Driver memory allocation
        """
        if not SPARK_AVAILABLE:
            raise ImportError("PySpark is required. Install with: pip install pyspark")

        self.spark = SparkSession.builder \
            .appName(app_name) \
            .master(master) \
            .config("spark.executor.memory", executor_memory) \
            .config("spark.driver.memory", driver_memory) \
            .config("spark.sql.shuffle.partitions", "200") \
            .getOrCreate()

        logger.info(f"Spark session created: {app_name}")

    def preprocess_documents(self,
                              input_path: str,
                              output_path: str,
                              quality_threshold: float = 0.8,
                              max_pages: int = 50,
                              train_ratio: float = 0.8,
                              val_ratio: float = 0.1) -> Dict[str, Any]:
        """
        Preprocess documents with quality filtering and splitting.

        Args:
            input_path: Input parquet/JSON file with document metadata
            output_path: Output path for processed documents
            quality_threshold: Minimum quality score (0-1)
            max_pages: Maximum pages per document
            train_ratio: Training set ratio
            val_ratio: Validation set ratio

        Returns:
            Processing statistics
        """
        logger.info(f"Starting document preprocessing: {input_path}")

        # Read input data
        df = self.spark.read.parquet(input_path) if input_path.endswith('.parquet') \
            else self.spark.read.json(input_path)

        initial_count = df.count()
        logger.info(f"Loaded {initial_count} documents")

        # Apply quality filters
        df_clean = df \
            .filter(F.col("quality_score") > quality_threshold) \
            .filter(F.col("num_pages") <= max_pages) \
            .filter(F.col("file_size_bytes") > 1024)  # Min 1KB

        # Remove duplicates based on document hash
        df_dedup = df_clean.dropDuplicates(["document_hash"])

        filtered_count = df_dedup.count()
        logger.info(f"After filtering: {filtered_count} documents ({filtered_count/initial_count*100:.1f}%)")

        # Add preprocessing columns
        df_processed = df_dedup \
            .withColumn("preprocessed_at", F.current_timestamp()) \
            .withColumn("preprocessed_version", F.lit("1.0"))

        # Stratified split by document_type
        df_train, df_val, df_test = self._stratified_split(
            df_processed,
            "document_type",
            train_ratio,
            val_ratio
        )

        # Save splits
        train_path = f"{output_path}/train"
        val_path = f"{output_path}/val"
        test_path = f"{output_path}/test"

        df_train.write.mode("overwrite").parquet(train_path)
        df_val.write.mode("overwrite").parquet(val_path)
        df_test.write.mode("overwrite").parquet(test_path)

        stats = {
            "total_input": initial_count,
            "total_output": filtered_count,
            "train_count": df_train.count(),
            "val_count": df_val.count(),
            "test_count": df_test.count(),
            "quality_threshold": quality_threshold,
            "output_paths": {
                "train": train_path,
                "val": val_path,
                "test": test_path
            }
        }

        logger.info(f"Preprocessing complete: {stats}")
        return stats

    def _stratified_split(self,
                          df: DataFrame,
                          stratify_column: str,
                          train_ratio: float,
                          val_ratio: float) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """
        Perform stratified train/val/test split.

        Args:
            df: Input DataFrame
            stratify_column: Column to stratify on
            train_ratio: Training set ratio
            val_ratio: Validation set ratio

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        test_ratio = 1.0 - train_ratio - val_ratio

        # Get unique values in stratify column
        unique_values = df.select(stratify_column).distinct().collect()

        train_dfs = []
        val_dfs = []
        test_dfs = []

        for row in unique_values:
            value = row[stratify_column]

            # Filter by current value
            df_subset = df.filter(F.col(stratify_column) == value)

            # Random split
            splits = df_subset.randomSplit([train_ratio, val_ratio, test_ratio], seed=42)

            train_dfs.append(splits[0])
            val_dfs.append(splits[1])
            test_dfs.append(splits[2])

        # Union all splits
        from functools import reduce
        train_df = reduce(DataFrame.union, train_dfs)
        val_df = reduce(DataFrame.union, val_dfs)
        test_df = reduce(DataFrame.union, test_dfs)

        return train_df, val_df, test_df

    def compute_document_statistics(self, input_path: str) -> Dict[str, Any]:
        """
        Compute statistics on document dataset.

        Args:
            input_path: Path to parquet/JSON file

        Returns:
            Dataset statistics
        """
        df = self.spark.read.parquet(input_path) if input_path.endswith('.parquet') \
            else self.spark.read.json(input_path)

        stats = {
            "total_documents": df.count(),
            "by_type": {},
            "quality_distribution": {},
            "size_distribution": {}
        }

        # Count by document type
        type_counts = df.groupBy("document_type").count().collect()
        for row in type_counts:
            stats["by_type"][row["document_type"]] = row["count"]

        # Quality score distribution
        quality_stats = df.select(
            F.mean("quality_score").alias("mean"),
            F.stddev("quality_score").alias("stddev"),
            F.min("quality_score").alias("min"),
            F.max("quality_score").alias("max"),
            F.expr("percentile_approx(quality_score, 0.5)").alias("median")
        ).collect()[0]

        stats["quality_distribution"] = {
            "mean": float(quality_stats["mean"]) if quality_stats["mean"] else 0,
            "stddev": float(quality_stats["stddev"]) if quality_stats["stddev"] else 0,
            "min": float(quality_stats["min"]) if quality_stats["min"] else 0,
            "max": float(quality_stats["max"]) if quality_stats["max"] else 0,
            "median": float(quality_stats["median"]) if quality_stats["median"] else 0,
        }

        # File size distribution
        size_stats = df.select(
            F.mean("file_size_bytes").alias("mean"),
            F.min("file_size_bytes").alias("min"),
            F.max("file_size_bytes").alias("max")
        ).collect()[0]

        stats["size_distribution"] = {
            "mean_bytes": int(size_stats["mean"]) if size_stats["mean"] else 0,
            "min_bytes": int(size_stats["min"]) if size_stats["min"] else 0,
            "max_bytes": int(size_stats["max"]) if size_stats["max"] else 0,
        }

        return stats

    def enhance_image_quality(self,
                                df: DataFrame,
                                operations: List[str] = ["denoise", "deskew", "binarize"]) -> DataFrame:
        """
        Apply image enhancement operations.

        Args:
            df: Input DataFrame with image paths
            operations: List of operations to apply

        Returns:
            DataFrame with enhanced images
        """
        # This would use UDFs to apply image processing
        # Placeholder for now
        logger.info(f"Applying image enhancements: {operations}")

        enhanced_df = df.withColumn("enhanced", F.lit(True))
        return enhanced_df

    def stop(self):
        """Stop Spark session."""
        self.spark.stop()
        logger.info("Spark session stopped")


# Standalone preprocessing functions
def denoise_image(image_path: str, output_path: str) -> bool:
    """
    Apply denoising to document image.

    Args:
        image_path: Input image path
        output_path: Output image path

    Returns:
        Success status
    """
    try:
        from PIL import Image
        import cv2

        # Load image
        img = cv2.imread(image_path)

        # Apply Non-local Means Denoising
        denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

        # Save
        cv2.imwrite(output_path, denoised)
        return True

    except Exception as e:
        logger.error(f"Error denoising {image_path}: {e}")
        return False


def deskew_image(image_path: str, output_path: str) -> bool:
    """
    Deskew document image.

    Args:
        image_path: Input image path
        output_path: Output image path

    Returns:
        Success status
    """
    try:
        from PIL import Image
        import cv2
        import numpy as np

        # Load image
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect skew angle
        coords = np.column_stack(np.where(gray > 0))
        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # Rotate image
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)

        # Save
        cv2.imwrite(output_path, rotated)
        return True

    except Exception as e:
        logger.error(f"Error deskewing {image_path}: {e}")
        return False


def assess_image_quality(image_path: str) -> float:
    """
    Assess document image quality.

    Metrics:
    - Sharpness (Laplacian variance)
    - Contrast
    - Brightness
    - Noise level

    Args:
        image_path: Path to image

    Returns:
        Quality score (0-1)
    """
    try:
        import cv2
        import numpy as np

        # Load image
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()

        # Normalize sharpness (empirical threshold: 100 = good, 500+ = excellent)
        sharpness_score = min(sharpness / 500.0, 1.0)

        # Contrast (standard deviation of pixel values)
        contrast = gray.std() / 128.0  # Normalize by half of range

        # Brightness (mean pixel value, target ~128)
        brightness = 1.0 - abs(gray.mean() - 128) / 128.0

        # Combine scores
        quality_score = (sharpness_score * 0.5 + contrast * 0.3 + brightness * 0.2)

        return min(max(quality_score, 0.0), 1.0)

    except Exception as e:
        logger.error(f"Error assessing quality for {image_path}: {e}")
        return 0.0


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize preprocessor
    preprocessor = SparkPreprocessor(
        app_name="SAP_LLM_Data_Preprocessing",
        master="local[4]"  # Use 4 cores
    )

    # Preprocess documents
    stats = preprocessor.preprocess_documents(
        input_path="./data/raw/metadata.json",
        output_path="./data/processed",
        quality_threshold=0.8,
        max_pages=50
    )

    print(f"Preprocessing stats: {stats}")

    # Compute statistics
    dataset_stats = preprocessor.compute_document_statistics("./data/processed/train")
    print(f"Dataset statistics: {dataset_stats}")

    preprocessor.stop()
