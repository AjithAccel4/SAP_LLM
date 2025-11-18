"""
Public Dataset Downloader for SAP_LLM Training Corpus.

Downloads and prepares public document AI datasets:
- RVL-CDIP: 400K document images (16 categories)
- FUNSD: Form Understanding in Noisy Scanned Documents (199 forms)
- CORD: Consolidated Receipt Dataset for post-OCR parsing (11,000 receipts)
- SROIE: Scanned Receipts OCR and Information Extraction (1,000 receipts)

Target: 200K+ documents from public datasets to augment training corpus.
"""

import os
import json
import logging
import urllib.request
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil

logger = logging.getLogger(__name__)


class PublicDatasetDownloader:
    """
    Download and prepare public document AI datasets.

    Handles automated download, extraction, and format conversion
    for training corpus integration.
    """

    def __init__(self, download_dir: str, max_workers: int = 4):
        """
        Initialize public dataset downloader.

        Args:
            download_dir: Directory to store downloaded datasets
            max_workers: Number of parallel download workers
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

        self.max_workers = max_workers

        # Dataset configurations
        self.dataset_configs = self._initialize_dataset_configs()

        # Download statistics
        self.stats = {
            "total_downloaded": 0,
            "by_dataset": {},
            "errors": 0
        }

        logger.info(f"PublicDatasetDownloader initialized: {download_dir}")

    def _initialize_dataset_configs(self) -> Dict[str, Dict]:
        """Initialize configuration for each public dataset."""
        return {
            "rvl-cdip": {
                "name": "RVL-CDIP",
                "description": "400K document images across 16 categories",
                "url": "https://huggingface.co/datasets/aharley/rvl_cdip",
                "expected_documents": 400000,
                "use_hf_datasets": True,
                "hf_dataset_name": "aharley/rvl_cdip",
                "categories": [
                    "letter", "form", "email", "handwritten", "advertisement",
                    "scientific_report", "scientific_publication", "specification",
                    "file_folder", "news_article", "budget", "invoice",
                    "presentation", "questionnaire", "resume", "memo"
                ]
            },
            "funsd": {
                "name": "FUNSD",
                "description": "Form Understanding in Noisy Scanned Documents",
                "url": "https://guillaumejaume.github.io/FUNSD/dataset.zip",
                "expected_documents": 199,
                "use_hf_datasets": True,
                "hf_dataset_name": "nielsr/funsd",
                "download_url": "https://guillaumejaume.github.io/FUNSD/dataset.zip"
            },
            "cord": {
                "name": "CORD",
                "description": "Consolidated Receipt Dataset",
                "url": "https://huggingface.co/datasets/naver-clova-ix/cord-v2",
                "expected_documents": 11000,
                "use_hf_datasets": True,
                "hf_dataset_name": "naver-clova-ix/cord-v2"
            },
            "sroie": {
                "name": "SROIE",
                "description": "Scanned Receipts OCR and Information Extraction",
                "url": "https://rrc.cvc.uab.es/?ch=13",
                "expected_documents": 1000,
                "use_hf_datasets": True,
                "hf_dataset_name": "darentang/sroie",
                "manual_download": True,  # Requires registration
                "note": "Dataset requires manual download from competition website"
            }
        }

    def download_all_datasets(self,
                             sample_limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Download all configured public datasets.

        Args:
            sample_limit: Optional limit on samples per dataset (for testing)

        Returns:
            List of downloaded document metadata
        """
        logger.info("=" * 80)
        logger.info("Starting Public Datasets Download")
        logger.info("=" * 80)

        all_documents = []

        for dataset_name in ["rvl-cdip", "funsd", "cord", "sroie"]:
            logger.info(f"\nDownloading {dataset_name}...")

            try:
                docs = self.download_dataset(
                    dataset_name=dataset_name,
                    sample_limit=sample_limit
                )
                all_documents.extend(docs)

                self.stats["by_dataset"][dataset_name] = len(docs)

                logger.info(f"✅ Downloaded {len(docs):,} documents from {dataset_name}")

            except Exception as e:
                logger.error(f"❌ Error downloading {dataset_name}: {e}")
                self.stats["errors"] += 1

        self.stats["total_downloaded"] = len(all_documents)

        logger.info("\n" + "=" * 80)
        logger.info(f"Download Complete: {len(all_documents):,} total documents")
        logger.info("=" * 80)

        return all_documents

    def download_dataset(self,
                        dataset_name: str,
                        sample_limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Download specific public dataset.

        Args:
            dataset_name: Name of dataset to download
            sample_limit: Optional limit on number of samples

        Returns:
            List of document metadata
        """
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        config = self.dataset_configs[dataset_name]

        logger.info(f"Downloading {config['name']} ({config['description']})")

        # Use Hugging Face datasets library if available
        if config.get("use_hf_datasets", False):
            return self._download_hf_dataset(
                dataset_name=dataset_name,
                config=config,
                sample_limit=sample_limit
            )
        else:
            return self._download_direct(
                dataset_name=dataset_name,
                config=config,
                sample_limit=sample_limit
            )

    def _download_hf_dataset(self,
                            dataset_name: str,
                            config: Dict,
                            sample_limit: Optional[int]) -> List[Dict[str, Any]]:
        """
        Download dataset using Hugging Face datasets library.

        Args:
            dataset_name: Dataset identifier
            config: Dataset configuration
            sample_limit: Optional sample limit

        Returns:
            List of document metadata
        """
        try:
            from datasets import load_dataset

            logger.info(f"Loading from Hugging Face: {config['hf_dataset_name']}")

            # Load dataset
            hf_dataset_name = config["hf_dataset_name"]

            # Download dataset
            dataset = load_dataset(hf_dataset_name, trust_remote_code=True)

            # Process documents
            documents = []

            # Determine which splits to use
            splits_to_use = []
            if "train" in dataset:
                splits_to_use.append("train")
            if "test" in dataset:
                splits_to_use.append("test")
            if "validation" in dataset:
                splits_to_use.append("validation")

            # If no standard splits, use all available
            if not splits_to_use:
                splits_to_use = list(dataset.keys())

            for split in splits_to_use:
                split_data = dataset[split]

                # Apply sample limit if specified
                if sample_limit:
                    split_data = split_data.select(range(min(sample_limit, len(split_data))))

                logger.info(f"  Processing {split} split: {len(split_data):,} samples")

                for idx, item in enumerate(split_data):
                    # Convert to our format
                    doc_meta = self._convert_hf_item_to_metadata(
                        item=item,
                        dataset_name=dataset_name,
                        split=split,
                        index=idx
                    )

                    # Save image if available
                    if "image" in item and item["image"] is not None:
                        doc_meta["stored_path"] = self._save_image(
                            image=item["image"],
                            dataset_name=dataset_name,
                            doc_id=doc_meta["id"]
                        )

                    documents.append(doc_meta)

                    if (idx + 1) % 1000 == 0:
                        logger.info(f"    Processed {idx + 1:,} / {len(split_data):,}")

            logger.info(f"✅ Loaded {len(documents):,} documents from {dataset_name}")

            return documents

        except ImportError:
            logger.error("Hugging Face datasets library not available. Install with: pip install datasets")
            return []
        except Exception as e:
            logger.error(f"Error downloading {dataset_name}: {e}")
            return []

    def _convert_hf_item_to_metadata(self,
                                    item: Dict,
                                    dataset_name: str,
                                    split: str,
                                    index: int) -> Dict[str, Any]:
        """
        Convert Hugging Face dataset item to our metadata format.

        Args:
            item: HF dataset item
            dataset_name: Dataset name
            split: Dataset split
            index: Item index

        Returns:
            Document metadata
        """
        doc_id = f"{dataset_name}_{split}_{index:08d}"

        metadata = {
            "id": doc_id,
            "source": f"public/{dataset_name}",
            "dataset": dataset_name,
            "split": split,
            "index": index,
        }

        # Extract document type if available
        if dataset_name == "rvl-cdip" and "label" in item:
            label_idx = item["label"]
            categories = self.dataset_configs["rvl-cdip"]["categories"]
            if label_idx < len(categories):
                metadata["document_type"] = categories[label_idx]
        elif dataset_name == "funsd":
            metadata["document_type"] = "form"
        elif dataset_name in ["cord", "sroie"]:
            metadata["document_type"] = "receipt"

        # Extract annotations if available
        if "words" in item:
            metadata["ocr_words"] = item["words"]
        if "bboxes" in item or "boxes" in item:
            metadata["bounding_boxes"] = item.get("bboxes") or item.get("boxes")
        if "ner_tags" in item:
            metadata["ner_tags"] = item["ner_tags"]

        # Add ground truth fields if available
        if dataset_name == "cord" and "ground_truth" in item:
            metadata["ground_truth"] = item["ground_truth"]

        return metadata

    def _save_image(self,
                   image,
                   dataset_name: str,
                   doc_id: str) -> str:
        """
        Save image to disk.

        Args:
            image: PIL Image or image path
            dataset_name: Dataset name
            doc_id: Document ID

        Returns:
            Saved file path
        """
        try:
            from PIL import Image

            # Create output directory
            output_dir = self.download_dir / dataset_name / "images"
            output_dir.mkdir(parents=True, exist_ok=True)

            output_path = output_dir / f"{doc_id}.png"

            # Save image
            if isinstance(image, str):
                # If it's a path, copy it
                shutil.copy(image, output_path)
            else:
                # If it's a PIL Image, save it
                image.save(output_path, "PNG")

            return str(output_path)

        except Exception as e:
            logger.error(f"Error saving image for {doc_id}: {e}")
            return ""

    def _download_direct(self,
                        dataset_name: str,
                        config: Dict,
                        sample_limit: Optional[int]) -> List[Dict[str, Any]]:
        """
        Download dataset directly from URL.

        Args:
            dataset_name: Dataset identifier
            config: Dataset configuration
            sample_limit: Optional sample limit

        Returns:
            List of document metadata
        """
        if config.get("manual_download", False):
            logger.warning(f"⚠️  {dataset_name} requires manual download")
            logger.warning(f"   Visit: {config['url']}")
            logger.warning(f"   Note: {config.get('note', '')}")
            return []

        download_url = config.get("download_url")
        if not download_url:
            logger.error(f"No download URL for {dataset_name}")
            return []

        # Download archive
        archive_path = self.download_dir / f"{dataset_name}.zip"

        logger.info(f"Downloading from {download_url}")

        try:
            urllib.request.urlretrieve(download_url, archive_path)
            logger.info(f"Downloaded to {archive_path}")

            # Extract archive
            extract_dir = self.download_dir / dataset_name
            extract_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Extracting to {extract_dir}")

            if archive_path.suffix == ".zip":
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            elif archive_path.suffix in [".tar", ".tar.gz", ".tgz"]:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_dir)

            logger.info(f"Extracted successfully")

            # Process extracted files
            documents = self._process_extracted_dataset(
                dataset_name=dataset_name,
                extract_dir=extract_dir,
                sample_limit=sample_limit
            )

            # Cleanup archive
            archive_path.unlink()

            return documents

        except Exception as e:
            logger.error(f"Error downloading {dataset_name}: {e}")
            return []

    def _process_extracted_dataset(self,
                                   dataset_name: str,
                                   extract_dir: Path,
                                   sample_limit: Optional[int]) -> List[Dict[str, Any]]:
        """
        Process extracted dataset files.

        Args:
            dataset_name: Dataset identifier
            extract_dir: Extraction directory
            sample_limit: Optional sample limit

        Returns:
            List of document metadata
        """
        documents = []

        # Find all images
        image_extensions = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
        image_files = []

        for ext in image_extensions:
            image_files.extend(extract_dir.rglob(f"*{ext}"))

        logger.info(f"Found {len(image_files):,} images")

        # Apply sample limit
        if sample_limit:
            image_files = image_files[:sample_limit]

        for idx, image_path in enumerate(image_files):
            doc_id = f"{dataset_name}_{idx:08d}"

            doc_meta = {
                "id": doc_id,
                "source": f"public/{dataset_name}",
                "dataset": dataset_name,
                "stored_path": str(image_path),
                "filename": image_path.name,
                "document_type": "unknown"
            }

            documents.append(doc_meta)

            if (idx + 1) % 1000 == 0:
                logger.info(f"  Processed {idx + 1:,} / {len(image_files):,}")

        return documents

    def save_metadata(self,
                     documents: List[Dict[str, Any]],
                     output_file: str):
        """
        Save downloaded documents metadata.

        Args:
            documents: List of document metadata
            output_file: Output JSON file path
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump({
                "total_documents": len(documents),
                "statistics": self.stats,
                "documents": documents
            }, f, indent=2)

        logger.info(f"Saved metadata for {len(documents):,} documents to {output_file}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get download statistics."""
        return self.stats.copy()


# CLI entrypoint
def main():
    """CLI for public dataset download."""
    import argparse

    parser = argparse.ArgumentParser(description="Download Public Document AI Datasets")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--dataset", help="Specific dataset to download (all if not specified)")
    parser.add_argument("--sample-limit", type=int, help="Limit samples per dataset (for testing)")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create downloader
    downloader = PublicDatasetDownloader(download_dir=args.output_dir)

    # Download datasets
    if args.dataset:
        documents = downloader.download_dataset(
            dataset_name=args.dataset,
            sample_limit=args.sample_limit
        )
    else:
        documents = downloader.download_all_datasets(
            sample_limit=args.sample_limit
        )

    # Save metadata
    downloader.save_metadata(
        documents=documents,
        output_file=f"{args.output_dir}/public_datasets_metadata.json"
    )

    print(f"\n{'=' * 80}")
    print(f"Download Complete!")
    print(f"Total Documents: {len(documents):,}")
    print(f"Statistics: {downloader.get_statistics()}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
