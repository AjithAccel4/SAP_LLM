#!/usr/bin/env python3
"""
Download and cache HuggingFace models for SAP_LLM.

This script downloads all required models with progress tracking, retry logic,
and comprehensive error handling.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from huggingface_hub import snapshot_download
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table

# Setup rich console
console = Console()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


# Model configurations
MODELS = {
    "vision_encoder": {
        "repo_id": "microsoft/layoutlmv3-base",
        "description": "Vision Encoder (LayoutLMv3)",
        "size_gb": 1.2,
        "required": True,
    },
    "language_decoder": {
        "repo_id": "meta-llama/Llama-2-7b-hf",
        "description": "Language Decoder (LLaMA-2-7B)",
        "size_gb": 13.5,
        "required": True,
    },
    "reasoning_engine": {
        "repo_id": "mistralai/Mixtral-8x7B-v0.1",
        "description": "Reasoning Engine (Mixtral-8x7B)",
        "size_gb": 87.0,
        "required": True,
    },
    "ocr": {
        "repo_id": "microsoft/trocr-base-handwritten",
        "description": "OCR Model (TrOCR)",
        "size_gb": 0.5,
        "required": False,
    },
}


class ModelDownloader:
    """Download and manage HuggingFace models."""

    def __init__(
        self,
        cache_dir: Path,
        token: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: int = 5,
    ):
        """
        Initialize the model downloader.

        Args:
            cache_dir: Directory to cache downloaded models
            token: HuggingFace API token for gated models
            max_retries: Maximum number of retry attempts
            retry_delay: Delay in seconds between retries
        """
        self.cache_dir = Path(cache_dir)
        self.token = token
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Track download statistics
        self.stats = {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
        }

    def check_disk_space(self, required_gb: float) -> bool:
        """
        Check if sufficient disk space is available.

        Args:
            required_gb: Required space in GB

        Returns:
            True if sufficient space available
        """
        try:
            stat = os.statvfs(self.cache_dir)
            available_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)

            if available_gb < required_gb:
                console.print(
                    f"[red]Insufficient disk space![/red] "
                    f"Required: {required_gb:.1f}GB, Available: {available_gb:.1f}GB"
                )
                return False

            console.print(
                f"[green]Disk space check passed.[/green] "
                f"Available: {available_gb:.1f}GB"
            )
            return True

        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")
            return True  # Proceed if check fails

    def model_exists(self, repo_id: str) -> bool:
        """
        Check if model is already downloaded.

        Args:
            repo_id: HuggingFace repository ID

        Returns:
            True if model exists in cache
        """
        # Convert repo_id to cache directory name
        model_dir = self.cache_dir / repo_id.replace("/", "--")

        if model_dir.exists() and any(model_dir.iterdir()):
            logger.info(f"Model {repo_id} already exists in cache")
            return True

        return False

    def download_model(
        self,
        repo_id: str,
        description: str,
        progress: Progress,
        task_id: TaskID,
    ) -> bool:
        """
        Download a model with retry logic.

        Args:
            repo_id: HuggingFace repository ID
            description: Model description for logging
            progress: Rich progress bar
            task_id: Progress task ID

        Returns:
            True if download successful
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                progress.update(
                    task_id,
                    description=f"[cyan]{description}[/cyan] (Attempt {attempt}/{self.max_retries})",
                )

                # Download model
                snapshot_download(
                    repo_id=repo_id,
                    cache_dir=self.cache_dir,
                    token=self.token,
                    resume_download=True,
                    local_files_only=False,
                )

                progress.update(task_id, description=f"[green]✓ {description}[/green]")
                logger.info(f"Successfully downloaded {repo_id}")
                return True

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Attempt {attempt} failed for {repo_id}: {error_msg}")

                if attempt < self.max_retries:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    progress.update(
                        task_id, description=f"[red]✗ {description} (Failed)[/red]"
                    )

                    # Check for common errors
                    if "401" in error_msg or "403" in error_msg:
                        console.print(
                            f"[red]Authentication error for {repo_id}. "
                            "Please check your HuggingFace token.[/red]"
                        )
                    elif "gated" in error_msg.lower():
                        console.print(
                            f"[yellow]Model {repo_id} is gated. "
                            "Please request access at huggingface.co[/yellow]"
                        )

        return False

    def download_all(
        self,
        models: Optional[List[str]] = None,
        skip_existing: bool = True,
        force: bool = False,
    ) -> Dict[str, bool]:
        """
        Download all or specified models.

        Args:
            models: List of model names to download (None = all)
            skip_existing: Skip models that already exist
            force: Force re-download even if models exist

        Returns:
            Dictionary mapping model names to success status
        """
        # Determine which models to download
        if models is None:
            models_to_download = MODELS.keys()
        else:
            models_to_download = models

        # Calculate total size
        total_size_gb = sum(
            MODELS[name]["size_gb"]
            for name in models_to_download
            if name in MODELS
        )

        # Display summary
        table = Table(title="Models to Download")
        table.add_column("Model", style="cyan")
        table.add_column("Repository", style="magenta")
        table.add_column("Size (GB)", justify="right", style="yellow")
        table.add_column("Required", justify="center")

        for name in models_to_download:
            if name not in MODELS:
                console.print(f"[red]Unknown model: {name}[/red]")
                continue

            config = MODELS[name]
            table.add_row(
                config["description"],
                config["repo_id"],
                f"{config['size_gb']:.1f}",
                "✓" if config["required"] else "○",
            )

        console.print(table)
        console.print(f"\n[bold]Total size:[/bold] {total_size_gb:.1f} GB\n")

        # Check disk space
        if not self.check_disk_space(total_size_gb * 1.5):  # 50% buffer
            if not force:
                return {}

        # Download models with progress tracking
        results = {}
        self.stats["total"] = len(models_to_download)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            for name in models_to_download:
                if name not in MODELS:
                    continue

                config = MODELS[name]
                repo_id = config["repo_id"]
                description = config["description"]

                # Check if model exists
                if skip_existing and not force and self.model_exists(repo_id):
                    console.print(f"[yellow]Skipping {description} (already exists)[/yellow]")
                    results[name] = True
                    self.stats["skipped"] += 1
                    continue

                # Create progress task
                task_id = progress.add_task(
                    f"[cyan]{description}[/cyan]",
                    total=None,
                )

                # Download model
                success = self.download_model(repo_id, description, progress, task_id)
                results[name] = success

                if success:
                    self.stats["successful"] += 1
                else:
                    self.stats["failed"] += 1

        return results

    def print_summary(self, results: Dict[str, bool]):
        """
        Print download summary.

        Args:
            results: Dictionary mapping model names to success status
        """
        console.print("\n" + "=" * 60)
        console.print("[bold]Download Summary[/bold]")
        console.print("=" * 60)

        # Statistics
        console.print(f"Total models: {self.stats['total']}")
        console.print(f"[green]Successful: {self.stats['successful']}[/green]")
        console.print(f"[yellow]Skipped: {self.stats['skipped']}[/yellow]")
        console.print(f"[red]Failed: {self.stats['failed']}[/red]")

        # Failed models
        if self.stats["failed"] > 0:
            console.print("\n[red]Failed models:[/red]")
            for name, success in results.items():
                if not success:
                    config = MODELS.get(name, {})
                    console.print(f"  - {config.get('description', name)}")

        # Cache location
        console.print(f"\n[bold]Cache directory:[/bold] {self.cache_dir}")

        # Environment variable suggestion
        console.print(
            "\n[bold]Set the following environment variable:[/bold]\n"
            f"export HF_CACHE_DIR={self.cache_dir}"
        )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download HuggingFace models for SAP_LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.environ.get("HF_CACHE_DIR", "/models/huggingface_cache"),
        help="Directory to cache models (default: $HF_CACHE_DIR or /models/huggingface_cache)",
    )

    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace API token (default: $HF_TOKEN)",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()),
        help="Specific models to download (default: all)",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts (default: 3)",
    )

    parser.add_argument(
        "--retry-delay",
        type=int,
        default=5,
        help="Delay between retries in seconds (default: 5)",
    )

    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-download existing models",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even with insufficient disk space",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit",
    )

    args = parser.parse_args()

    # Display banner
    console.print(
        Panel.fit(
            "[bold cyan]SAP_LLM Model Downloader[/bold cyan]\n"
            "Download HuggingFace models with progress tracking",
            border_style="cyan",
        )
    )

    # List models
    if args.list:
        table = Table(title="Available Models")
        table.add_column("Name", style="cyan")
        table.add_column("Repository", style="magenta")
        table.add_column("Description", style="white")
        table.add_column("Size (GB)", justify="right", style="yellow")
        table.add_column("Required", justify="center")

        for name, config in MODELS.items():
            table.add_row(
                name,
                config["repo_id"],
                config["description"],
                f"{config['size_gb']:.1f}",
                "✓" if config["required"] else "○",
            )

        console.print(table)
        return 0

    # Check for HuggingFace token
    if not args.token:
        console.print(
            "[yellow]Warning: No HuggingFace token provided. "
            "Some models may require authentication.[/yellow]\n"
            "Set HF_TOKEN environment variable or use --token option.\n"
        )

    # Create downloader
    downloader = ModelDownloader(
        cache_dir=args.cache_dir,
        token=args.token,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )

    # Download models
    try:
        results = downloader.download_all(
            models=args.models,
            skip_existing=not args.no_skip_existing,
            force=args.force,
        )

        # Print summary
        downloader.print_summary(results)

        # Exit code based on results
        if downloader.stats["failed"] > 0:
            # Check if any required models failed
            failed_required = any(
                MODELS[name]["required"]
                for name, success in results.items()
                if not success and name in MODELS
            )

            if failed_required:
                console.print(
                    "\n[red]One or more required models failed to download.[/red]"
                )
                return 1
            else:
                console.print(
                    "\n[yellow]Some optional models failed to download.[/yellow]"
                )
                return 0
        else:
            console.print("\n[green]All models downloaded successfully![/green]")
            return 0

    except KeyboardInterrupt:
        console.print("\n[yellow]Download interrupted by user.[/yellow]")
        return 130

    except Exception as e:
        console.print(f"\n[red]Unexpected error: {e}[/red]")
        logger.exception("Unexpected error during download")
        return 1


if __name__ == "__main__":
    sys.exit(main())
