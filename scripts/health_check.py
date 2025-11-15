#!/usr/bin/env python3
"""
System health check for SAP_LLM.

Checks all services, databases, models, and system resources to ensure
the system is ready for operation.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil
import redis
import requests
import yaml
from azure.cosmos import CosmosClient, exceptions
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.tree import Tree

# Setup rich console
console = Console()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_time=False)],
)
logger = logging.getLogger(__name__)


class HealthChecker:
    """Comprehensive system health checker."""

    def __init__(self, config: Dict[str, Any], verbose: bool = False):
        """
        Initialize the health checker.

        Args:
            config: Configuration dictionary
            verbose: Enable verbose output
        """
        self.config = config
        self.verbose = verbose
        self.checks = {}
        self.start_time = time.time()

    def check_gpu_availability(self) -> Dict[str, Any]:
        """
        Check GPU availability and CUDA setup.

        Returns:
            Dictionary with GPU check results
        """
        console.print("\n[bold cyan]Checking GPU Availability...[/bold cyan]")

        result = {
            "status": "unknown",
            "available": False,
            "cuda_available": False,
            "device_count": 0,
            "devices": [],
            "message": "",
        }

        try:
            import torch

            cuda_available = torch.cuda.is_available()
            result["cuda_available"] = cuda_available

            if cuda_available:
                device_count = torch.cuda.device_count()
                result["device_count"] = device_count
                result["available"] = True

                for i in range(device_count):
                    device_info = {
                        "id": i,
                        "name": torch.cuda.get_device_name(i),
                        "capability": torch.cuda.get_device_capability(i),
                        "total_memory": torch.cuda.get_device_properties(i).total_memory,
                        "memory_allocated": torch.cuda.memory_allocated(i),
                        "memory_reserved": torch.cuda.memory_reserved(i),
                    }
                    result["devices"].append(device_info)

                    console.print(f"  [green]✓[/green] GPU {i}: {device_info['name']}")
                    console.print(f"    Capability: {device_info['capability']}")
                    console.print(
                        f"    Memory: {device_info['total_memory'] / (1024**3):.1f} GB"
                    )

                result["status"] = "healthy"
                result["message"] = f"Found {device_count} GPU(s)"

            else:
                console.print("  [yellow]⚠[/yellow] CUDA not available - will use CPU")
                result["status"] = "warning"
                result["message"] = "CUDA not available, using CPU mode"

        except ImportError:
            console.print("  [red]✗[/red] PyTorch not installed")
            result["status"] = "error"
            result["message"] = "PyTorch not installed"

        except Exception as e:
            console.print(f"  [red]✗[/red] Error checking GPU: {e}")
            result["status"] = "error"
            result["message"] = str(e)

        return result

    def check_model_files(self) -> Dict[str, Any]:
        """
        Check if required model files exist.

        Returns:
            Dictionary with model check results
        """
        console.print("\n[bold cyan]Checking Model Files...[/bold cyan]")

        result = {
            "status": "unknown",
            "models": {},
            "total": 0,
            "found": 0,
            "missing": 0,
            "message": "",
        }

        models_config = self.config.get("models", {})

        # Models to check
        model_checks = {
            "vision_encoder": models_config.get("vision_encoder", {}),
            "language_decoder": models_config.get("language_decoder", {}),
            "reasoning_engine": models_config.get("reasoning_engine", {}),
        }

        for model_name, model_config in model_checks.items():
            model_path = model_config.get("path", "")
            model_path = Path(model_path)

            model_info = {
                "path": str(model_path),
                "exists": False,
                "size": 0,
                "files": [],
            }

            if model_path.exists() and model_path.is_dir():
                # Check for key model files
                key_files = [
                    "config.json",
                    "pytorch_model.bin",
                    "model.safetensors",
                ]

                found_files = []
                total_size = 0

                for key_file in key_files:
                    file_path = model_path / key_file
                    if file_path.exists():
                        found_files.append(key_file)
                        total_size += file_path.stat().st_size

                # Also check for any .bin or .safetensors files
                for ext in ["*.bin", "*.safetensors"]:
                    for file_path in model_path.glob(ext):
                        if file_path.name not in found_files:
                            found_files.append(file_path.name)
                            total_size += file_path.stat().st_size

                if found_files:
                    model_info["exists"] = True
                    model_info["size"] = total_size
                    model_info["files"] = found_files
                    result["found"] += 1

                    console.print(f"  [green]✓[/green] {model_name}")
                    console.print(f"    Path: {model_path}")
                    console.print(f"    Size: {total_size / (1024**3):.2f} GB")
                    if self.verbose:
                        console.print(f"    Files: {', '.join(found_files[:3])}")
                else:
                    result["missing"] += 1
                    console.print(f"  [red]✗[/red] {model_name} (no model files found)")

            else:
                result["missing"] += 1
                console.print(f"  [red]✗[/red] {model_name} (path not found)")

            result["models"][model_name] = model_info
            result["total"] += 1

        # Set overall status
        if result["missing"] == 0:
            result["status"] = "healthy"
            result["message"] = f"All {result['total']} models found"
        elif result["found"] > 0:
            result["status"] = "warning"
            result["message"] = f"{result['missing']}/{result['total']} models missing"
        else:
            result["status"] = "error"
            result["message"] = "No models found"

        return result

    def check_cosmos_db(self) -> Dict[str, Any]:
        """
        Check Cosmos DB connection and health.

        Returns:
            Dictionary with Cosmos DB check results
        """
        console.print("\n[bold cyan]Checking Cosmos DB...[/bold cyan]")

        result = {
            "status": "unknown",
            "connected": False,
            "database": None,
            "container": None,
            "message": "",
        }

        try:
            cosmos_config = self.config.get("pmg", {}).get("cosmos", {})
            endpoint = cosmos_config.get("endpoint")
            key = cosmos_config.get("key")
            database_name = cosmos_config.get("database", "qorsync")
            container_name = cosmos_config.get("container", "pmg")

            if not endpoint or not key or endpoint.startswith("${"):
                result["status"] = "warning"
                result["message"] = "Not configured"
                console.print("  [yellow]⚠[/yellow] Not configured")
                return result

            # Connect
            client = CosmosClient(endpoint, key, connection_timeout=5)

            # Test connection
            client.get_database_account()
            result["connected"] = True

            # Check database
            try:
                database = client.get_database_client(database_name)
                database.read()
                result["database"] = database_name
                console.print(f"  [green]✓[/green] Database: {database_name}")

                # Check container
                try:
                    container = database.get_container_client(container_name)
                    container.read()
                    result["container"] = container_name
                    console.print(f"  [green]✓[/green] Container: {container_name}")

                    result["status"] = "healthy"
                    result["message"] = "Connected and operational"

                except exceptions.CosmosResourceNotFoundError:
                    result["status"] = "warning"
                    result["message"] = "Container not found"
                    console.print(f"  [yellow]⚠[/yellow] Container not found: {container_name}")

            except exceptions.CosmosResourceNotFoundError:
                result["status"] = "warning"
                result["message"] = "Database not found"
                console.print(f"  [yellow]⚠[/yellow] Database not found: {database_name}")

        except exceptions.CosmosHttpResponseError as e:
            result["status"] = "error"
            result["message"] = f"HTTP error: {e.status_code}"
            console.print(f"  [red]✗[/red] Connection failed: {e.message}")

        except Exception as e:
            result["status"] = "error"
            result["message"] = str(e)
            console.print(f"  [red]✗[/red] Error: {e}")

        return result

    def check_mongodb(self) -> Dict[str, Any]:
        """
        Check MongoDB connection and health.

        Returns:
            Dictionary with MongoDB check results
        """
        console.print("\n[bold cyan]Checking MongoDB...[/bold cyan]")

        result = {
            "status": "unknown",
            "connected": False,
            "database": None,
            "collections": [],
            "version": None,
            "message": "",
        }

        try:
            mongodb_config = self.config.get("databases", {}).get("mongodb", {})
            uri = mongodb_config.get("uri", "mongodb://localhost:27017")
            database_name = mongodb_config.get("database", "sap_llm")

            # Connect
            client = MongoClient(uri, serverSelectionTimeoutMS=5000)

            # Test connection
            client.admin.command("ping")
            server_info = client.server_info()

            result["connected"] = True
            result["version"] = server_info.get("version", "Unknown")

            console.print(f"  [green]✓[/green] Connected")
            console.print(f"    Version: {result['version']}")

            # Check database
            db = client[database_name]
            result["database"] = database_name

            # Check collections
            collections = db.list_collection_names()
            result["collections"] = collections

            if collections:
                console.print(f"  [green]✓[/green] Database: {database_name}")
                console.print(f"    Collections: {len(collections)}")
                if self.verbose:
                    for coll in collections[:5]:
                        console.print(f"      - {coll}")
                    if len(collections) > 5:
                        console.print(f"      ... and {len(collections) - 5} more")

                result["status"] = "healthy"
                result["message"] = f"Connected with {len(collections)} collections"
            else:
                result["status"] = "warning"
                result["message"] = "No collections found"
                console.print(f"  [yellow]⚠[/yellow] No collections found")

        except ServerSelectionTimeoutError:
            result["status"] = "error"
            result["message"] = "Connection timeout"
            console.print("  [red]✗[/red] Connection timeout")

        except ConnectionFailure as e:
            result["status"] = "error"
            result["message"] = str(e)
            console.print(f"  [red]✗[/red] Connection failed: {e}")

        except Exception as e:
            result["status"] = "error"
            result["message"] = str(e)
            console.print(f"  [red]✗[/red] Error: {e}")

        return result

    def check_redis(self) -> Dict[str, Any]:
        """
        Check Redis connection and health.

        Returns:
            Dictionary with Redis check results
        """
        console.print("\n[bold cyan]Checking Redis...[/bold cyan]")

        result = {
            "status": "unknown",
            "connected": False,
            "version": None,
            "memory": None,
            "keys": 0,
            "message": "",
        }

        try:
            redis_config = self.config.get("databases", {}).get("redis", {})
            host = redis_config.get("host", "localhost")
            port = redis_config.get("port", 6379)
            db = redis_config.get("db", 0)
            password = redis_config.get("password")

            # Connect
            client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password if password else None,
                decode_responses=True,
                socket_connect_timeout=5,
            )

            # Test connection
            client.ping()
            result["connected"] = True

            # Get info
            info = client.info()
            result["version"] = info.get("redis_version", "Unknown")
            result["memory"] = info.get("used_memory_human", "Unknown")

            # Get key count
            db_info = info.get(f"db{db}", {})
            if isinstance(db_info, dict):
                result["keys"] = db_info.get("keys", 0)

            console.print(f"  [green]✓[/green] Connected to {host}:{port}")
            console.print(f"    Version: {result['version']}")
            console.print(f"    Memory: {result['memory']}")
            console.print(f"    Keys in DB {db}: {result['keys']}")

            result["status"] = "healthy"
            result["message"] = "Connected and operational"

        except redis.ConnectionError as e:
            result["status"] = "error"
            result["message"] = f"Connection failed: {str(e)}"
            console.print(f"  [red]✗[/red] Connection failed: {e}")

        except redis.AuthenticationError:
            result["status"] = "error"
            result["message"] = "Authentication failed"
            console.print("  [red]✗[/red] Authentication failed")

        except Exception as e:
            result["status"] = "error"
            result["message"] = str(e)
            console.print(f"  [red]✗[/red] Error: {e}")

        return result

    def check_api_service(self) -> Dict[str, Any]:
        """
        Check API service health.

        Returns:
            Dictionary with API check results
        """
        console.print("\n[bold cyan]Checking API Service...[/bold cyan]")

        result = {
            "status": "unknown",
            "available": False,
            "url": None,
            "version": None,
            "message": "",
        }

        try:
            api_config = self.config.get("api", {})
            host = api_config.get("host", "0.0.0.0")
            port = api_config.get("port", 8000)

            # Try localhost if host is 0.0.0.0
            if host == "0.0.0.0":
                host = "localhost"

            url = f"http://{host}:{port}"
            result["url"] = url

            # Check health endpoint
            try:
                response = requests.get(f"{url}/health", timeout=5)

                if response.status_code == 200:
                    data = response.json()
                    result["available"] = True
                    result["version"] = data.get("version", "Unknown")

                    console.print(f"  [green]✓[/green] Service available at {url}")
                    console.print(f"    Version: {result['version']}")

                    result["status"] = "healthy"
                    result["message"] = "Service operational"
                else:
                    result["status"] = "warning"
                    result["message"] = f"Unexpected status: {response.status_code}"
                    console.print(
                        f"  [yellow]⚠[/yellow] Service returned status {response.status_code}"
                    )

            except requests.exceptions.ConnectionError:
                result["status"] = "warning"
                result["message"] = "Service not running"
                console.print(f"  [yellow]⚠[/yellow] Service not running at {url}")

        except Exception as e:
            result["status"] = "error"
            result["message"] = str(e)
            console.print(f"  [red]✗[/red] Error: {e}")

        return result

    def check_system_resources(self) -> Dict[str, Any]:
        """
        Check system resource availability.

        Returns:
            Dictionary with system resource check results
        """
        console.print("\n[bold cyan]Checking System Resources...[/bold cyan]")

        result = {
            "status": "unknown",
            "cpu": {},
            "memory": {},
            "disk": {},
            "message": "",
        }

        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()

            result["cpu"] = {
                "percent": cpu_percent,
                "count": cpu_count,
            }

            console.print(f"  CPU: {cpu_percent}% ({cpu_count} cores)")

            # Memory
            memory = psutil.virtual_memory()
            result["memory"] = {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
            }

            console.print(
                f"  Memory: {memory.percent}% "
                f"({memory.available / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB available)"
            )

            # Disk
            disk = psutil.disk_usage("/")
            result["disk"] = {
                "total": disk.total,
                "free": disk.free,
                "percent": disk.percent,
            }

            console.print(
                f"  Disk: {disk.percent}% "
                f"({disk.free / (1024**3):.1f}GB / {disk.total / (1024**3):.1f}GB free)"
            )

            # Determine status based on thresholds
            issues = []
            if cpu_percent > 90:
                issues.append("High CPU usage")
            if memory.percent > 90:
                issues.append("High memory usage")
            if disk.percent > 90:
                issues.append("Low disk space")

            if issues:
                result["status"] = "warning"
                result["message"] = ", ".join(issues)
                console.print(f"  [yellow]⚠[/yellow] {result['message']}")
            else:
                result["status"] = "healthy"
                result["message"] = "Resources within normal range"

        except Exception as e:
            result["status"] = "error"
            result["message"] = str(e)
            console.print(f"  [red]✗[/red] Error: {e}")

        return result

    def run_all_checks(self) -> Dict[str, Any]:
        """
        Run all health checks.

        Returns:
            Dictionary with all check results
        """
        self.checks = {
            "gpu": self.check_gpu_availability(),
            "models": self.check_model_files(),
            "cosmos_db": self.check_cosmos_db(),
            "mongodb": self.check_mongodb(),
            "redis": self.check_redis(),
            "api": self.check_api_service(),
            "system": self.check_system_resources(),
        }

        return self.checks

    def print_summary(self):
        """Print health check summary."""
        duration = time.time() - self.start_time

        console.print("\n" + "=" * 70)
        console.print("[bold]Health Check Summary[/bold]")
        console.print("=" * 70)

        # Create summary table
        table = Table(show_header=True, header_style="bold")
        table.add_column("Component", style="cyan", width=20)
        table.add_column("Status", justify="center", width=15)
        table.add_column("Details", style="dim", width=30)

        status_emoji = {
            "healthy": "[green]✓ Healthy[/green]",
            "warning": "[yellow]⚠ Warning[/yellow]",
            "error": "[red]✗ Error[/red]",
            "unknown": "[dim]? Unknown[/dim]",
        }

        for component, check in self.checks.items():
            status = check.get("status", "unknown")
            message = check.get("message", "")

            # Format component name
            component_name = component.replace("_", " ").title()

            table.add_row(
                component_name,
                status_emoji.get(status, status),
                message,
            )

        console.print(table)

        # Overall status
        statuses = [check["status"] for check in self.checks.values()]
        if all(s == "healthy" for s in statuses):
            overall = "[green]✓ All systems operational[/green]"
        elif any(s == "error" for s in statuses):
            overall = "[red]✗ Critical issues detected[/red]"
        else:
            overall = "[yellow]⚠ Some warnings detected[/yellow]"

        console.print(f"\n[bold]Overall Status:[/bold] {overall}")
        console.print(f"[dim]Health check completed in {duration:.2f}s[/dim]")

    def get_exit_code(self) -> int:
        """
        Get exit code based on health check results.

        Returns:
            0 if all healthy, 1 if warnings, 2 if errors
        """
        statuses = [check["status"] for check in self.checks.values()]

        if any(s == "error" for s in statuses):
            return 2
        elif any(s == "warning" for s in statuses):
            return 1
        else:
            return 0

    def export_report(self, output_path: str):
        """
        Export health check report to JSON file.

        Args:
            output_path: Path to output file
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "duration": time.time() - self.start_time,
            "checks": self.checks,
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        console.print(f"\n[green]Report exported to: {output_path}[/green]")


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        script_dir = Path(__file__).parent
        config_path = script_dir.parent / "configs" / "default_config.yaml"

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Substitute environment variables
    import re

    def substitute_value(value):
        if isinstance(value, str):
            pattern = r'\$\{([^:}]+)(?::-(.*?))?\}'

            def replacer(match):
                var_name = match.group(1)
                default_value = match.group(2)
                return os.environ.get(var_name, default_value or "")

            return re.sub(pattern, replacer, value)
        elif isinstance(value, dict):
            return {k: substitute_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [substitute_value(item) for item in value]
        else:
            return value

    return substitute_value(config)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Health check for SAP_LLM system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (default: configs/default_config.yaml)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "--export",
        type=str,
        metavar="PATH",
        help="Export report to JSON file",
    )

    parser.add_argument(
        "--component",
        choices=["gpu", "models", "cosmos", "mongodb", "redis", "api", "system"],
        help="Check specific component only",
    )

    args = parser.parse_args()

    # Display banner
    console.print(
        Panel.fit(
            "[bold cyan]SAP_LLM Health Check[/bold cyan]\n"
            "System health and readiness verification",
            border_style="cyan",
        )
    )

    try:
        # Load configuration
        config = load_config(args.config)

        # Create health checker
        checker = HealthChecker(config, verbose=args.verbose)

        # Run checks
        if args.component:
            # Run specific check
            component_map = {
                "gpu": checker.check_gpu_availability,
                "models": checker.check_model_files,
                "cosmos": checker.check_cosmos_db,
                "mongodb": checker.check_mongodb,
                "redis": checker.check_redis,
                "api": checker.check_api_service,
                "system": checker.check_system_resources,
            }

            check_func = component_map.get(args.component)
            if check_func:
                checker.checks[args.component] = check_func()
        else:
            # Run all checks
            checker.run_all_checks()

        # Print summary
        checker.print_summary()

        # Export report if requested
        if args.export:
            checker.export_report(args.export)

        # Return appropriate exit code
        return checker.get_exit_code()

    except FileNotFoundError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        return 1

    except KeyboardInterrupt:
        console.print("\n[yellow]Health check interrupted by user.[/yellow]")
        return 130

    except Exception as e:
        console.print(f"\n[red]Unexpected error: {e}[/red]")
        logger.exception("Health check failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
