#!/usr/bin/env python3
"""
Initialize all databases for SAP_LLM.

This script initializes Cosmos DB, MongoDB, and Redis with proper error handling,
validation, and comprehensive logging.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import redis
import yaml
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from pymongo import MongoClient
from pymongo.errors import (
    ConnectionFailure,
    OperationFailure,
    ServerSelectionTimeoutError,
)
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
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


class DatabaseInitializer:
    """Initialize and validate all database connections."""

    def __init__(self, config: Dict[str, Any], dry_run: bool = False):
        """
        Initialize the database manager.

        Args:
            config: Configuration dictionary
            dry_run: If True, only validate connections without making changes
        """
        self.config = config
        self.dry_run = dry_run
        self.results = {
            "cosmos_db": {"status": "pending", "message": ""},
            "mongodb": {"status": "pending", "message": ""},
            "redis": {"status": "pending", "message": ""},
        }

    def init_cosmos_db(self) -> bool:
        """
        Initialize Azure Cosmos DB.

        Returns:
            True if successful
        """
        console.print("\n[bold cyan]Initializing Cosmos DB (PMG)...[/bold cyan]")

        try:
            # Get configuration
            cosmos_config = self.config.get("pmg", {}).get("cosmos", {})
            endpoint = cosmos_config.get("endpoint")
            key = cosmos_config.get("key")
            database_name = cosmos_config.get("database", "qorsync")
            container_name = cosmos_config.get("container", "pmg")

            # Validate configuration
            if not endpoint or not key:
                raise ValueError("Cosmos DB endpoint and key are required")

            if endpoint.startswith("${") or key.startswith("${"):
                raise ValueError(
                    "Cosmos DB credentials not set. Please configure .env file"
                )

            # Connect to Cosmos DB
            console.print(f"  Connecting to {endpoint}...")
            client = CosmosClient(endpoint, key)

            # Test connection
            account_info = client.get_database_account()
            console.print(f"  [green]✓[/green] Connected to Cosmos DB account")
            console.print(f"    Region: {account_info.ConsistencyPolicy}")

            if self.dry_run:
                console.print("  [yellow]Dry run mode - skipping database creation[/yellow]")
                self.results["cosmos_db"] = {
                    "status": "success",
                    "message": "Connection validated (dry run)",
                }
                return True

            # Create database
            console.print(f"  Creating database: {database_name}...")
            try:
                database = client.create_database(database_name)
                console.print(f"  [green]✓[/green] Database created")
            except exceptions.CosmosResourceExistsError:
                database = client.get_database_client(database_name)
                console.print(f"  [yellow]Database already exists[/yellow]")

            # Create container (graph)
            console.print(f"  Creating container: {container_name}...")
            try:
                container = database.create_container(
                    id=container_name,
                    partition_key=PartitionKey(path="/partitionKey"),
                    offer_throughput=400,  # Start with minimal throughput
                )
                console.print(f"  [green]✓[/green] Container created")
                console.print(f"    Throughput: 400 RU/s")
            except exceptions.CosmosResourceExistsError:
                container = database.get_container_client(container_name)
                console.print(f"  [yellow]Container already exists[/yellow]")

            # Create initial graph structure
            console.print("  Initializing graph structure...")

            # Add a test document to verify write access
            test_doc = {
                "id": "_test_connection",
                "partitionKey": "system",
                "label": "test",
                "properties": {
                    "initialized": True,
                    "timestamp": time.time(),
                },
            }

            try:
                container.upsert_item(test_doc)
                console.print(f"  [green]✓[/green] Write access verified")

                # Clean up test document
                container.delete_item(
                    item="_test_connection",
                    partition_key="system"
                )
            except Exception as e:
                console.print(f"  [yellow]Warning: Could not verify write access: {e}[/yellow]")

            console.print("[green]✓ Cosmos DB initialization complete[/green]")
            self.results["cosmos_db"] = {
                "status": "success",
                "message": f"Database: {database_name}, Container: {container_name}",
            }
            return True

        except exceptions.CosmosHttpResponseError as e:
            error_msg = f"Cosmos DB HTTP error: {e.message}"
            console.print(f"[red]✗ {error_msg}[/red]")
            self.results["cosmos_db"] = {"status": "failed", "message": error_msg}
            return False

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            console.print(f"[red]✗ {error_msg}[/red]")
            logger.exception("Cosmos DB initialization failed")
            self.results["cosmos_db"] = {"status": "failed", "message": error_msg}
            return False

    def init_mongodb(self) -> bool:
        """
        Initialize MongoDB.

        Returns:
            True if successful
        """
        console.print("\n[bold cyan]Initializing MongoDB...[/bold cyan]")

        try:
            # Get configuration
            mongodb_config = self.config.get("databases", {}).get("mongodb", {})
            uri = mongodb_config.get("uri", "mongodb://localhost:27017")
            database_name = mongodb_config.get("database", "sap_llm")
            collections_config = mongodb_config.get("collections", {})

            # Default collections
            collections = {
                "documents": "documents",
                "results": "results",
                "exceptions": "exceptions",
                "audit_log": "audit_log",
                "pmg_cache": "pmg_cache",
            }
            collections.update(collections_config)

            # Connect to MongoDB
            console.print(f"  Connecting to {uri}...")
            client = MongoClient(
                uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
            )

            # Test connection
            client.admin.command("ping")
            server_info = client.server_info()
            console.print(f"  [green]✓[/green] Connected to MongoDB")
            console.print(f"    Version: {server_info.get('version', 'Unknown')}")

            if self.dry_run:
                console.print("  [yellow]Dry run mode - skipping database creation[/yellow]")
                self.results["mongodb"] = {
                    "status": "success",
                    "message": "Connection validated (dry run)",
                }
                return True

            # Get or create database
            db = client[database_name]
            console.print(f"  Using database: {database_name}")

            # Create collections with validation
            created_collections = []
            existing_collections = []

            for coll_key, coll_name in collections.items():
                console.print(f"  Creating collection: {coll_name}...")

                if coll_name in db.list_collection_names():
                    console.print(f"    [yellow]Already exists[/yellow]")
                    existing_collections.append(coll_name)
                    continue

                # Create collection with validation schema
                if coll_name == "documents":
                    # Schema for document storage
                    db.create_collection(
                        coll_name,
                        validator={
                            "$jsonSchema": {
                                "bsonType": "object",
                                "required": ["document_id", "created_at"],
                                "properties": {
                                    "document_id": {"bsonType": "string"},
                                    "created_at": {"bsonType": "date"},
                                    "updated_at": {"bsonType": "date"},
                                },
                            }
                        },
                    )
                else:
                    db.create_collection(coll_name)

                created_collections.append(coll_name)
                console.print(f"    [green]✓[/green] Created")

            # Create indexes
            console.print("  Creating indexes...")

            # Documents collection indexes
            if "documents" in collections.values():
                docs_coll = db[collections.get("documents", "documents")]
                docs_coll.create_index("document_id", unique=True)
                docs_coll.create_index("created_at")
                docs_coll.create_index("document_type")
                console.print(f"    [green]✓[/green] Documents indexes created")

            # Results collection indexes
            if "results" in collections.values():
                results_coll = db[collections.get("results", "results")]
                results_coll.create_index("document_id")
                results_coll.create_index("created_at")
                results_coll.create_index("status")
                console.print(f"    [green]✓[/green] Results indexes created")

            # Exceptions collection indexes
            if "exceptions" in collections.values():
                exceptions_coll = db[collections.get("exceptions", "exceptions")]
                exceptions_coll.create_index("document_id")
                exceptions_coll.create_index("created_at")
                exceptions_coll.create_index("exception_type")
                exceptions_coll.create_index("resolved")
                console.print(f"    [green]✓[/green] Exceptions indexes created")

            # Summary
            console.print(f"\n  Collections created: {len(created_collections)}")
            console.print(f"  Collections existing: {len(existing_collections)}")

            console.print("[green]✓ MongoDB initialization complete[/green]")
            self.results["mongodb"] = {
                "status": "success",
                "message": f"Database: {database_name}, Collections: {len(collections)}",
            }
            return True

        except ServerSelectionTimeoutError:
            error_msg = "Connection timeout - MongoDB server not reachable"
            console.print(f"[red]✗ {error_msg}[/red]")
            console.print("  Ensure MongoDB is running and accessible")
            self.results["mongodb"] = {"status": "failed", "message": error_msg}
            return False

        except ConnectionFailure as e:
            error_msg = f"Connection failed: {str(e)}"
            console.print(f"[red]✗ {error_msg}[/red]")
            self.results["mongodb"] = {"status": "failed", "message": error_msg}
            return False

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            console.print(f"[red]✗ {error_msg}[/red]")
            logger.exception("MongoDB initialization failed")
            self.results["mongodb"] = {"status": "failed", "message": error_msg}
            return False

    def init_redis(self) -> bool:
        """
        Initialize Redis.

        Returns:
            True if successful
        """
        console.print("\n[bold cyan]Initializing Redis...[/bold cyan]")

        try:
            # Get configuration
            redis_config = self.config.get("databases", {}).get("redis", {})
            host = redis_config.get("host", "localhost")
            port = redis_config.get("port", 6379)
            db = redis_config.get("db", 0)
            password = redis_config.get("password")
            max_connections = redis_config.get("max_connections", 50)

            # Connect to Redis
            console.print(f"  Connecting to {host}:{port}...")
            client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password if password else None,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )

            # Test connection
            client.ping()
            info = client.info()
            console.print(f"  [green]✓[/green] Connected to Redis")
            console.print(f"    Version: {info.get('redis_version', 'Unknown')}")
            console.print(f"    Mode: {info.get('redis_mode', 'Unknown')}")
            console.print(f"    Memory: {info.get('used_memory_human', 'Unknown')}")

            if self.dry_run:
                console.print("  [yellow]Dry run mode - skipping cache initialization[/yellow]")
                self.results["redis"] = {
                    "status": "success",
                    "message": "Connection validated (dry run)",
                }
                return True

            # Initialize cache structure with namespace prefixes
            console.print("  Initializing cache structure...")

            # Define cache namespaces
            namespaces = [
                "sap_llm:models",       # Model cache
                "sap_llm:documents",    # Document cache
                "sap_llm:results",      # Result cache
                "sap_llm:pmg",          # PMG cache
                "sap_llm:sessions",     # Session cache
                "sap_llm:ratelimit",    # Rate limiting
            ]

            # Set initialization marker for each namespace
            pipe = client.pipeline()
            for namespace in namespaces:
                init_key = f"{namespace}:_initialized"
                pipe.set(init_key, "true", ex=86400)  # 24 hour TTL
            pipe.execute()

            console.print(f"  [green]✓[/green] Cache namespaces initialized: {len(namespaces)}")

            # Test write/read
            test_key = "sap_llm:test:connection"
            test_value = "initialized"
            client.setex(test_key, 60, test_value)

            retrieved = client.get(test_key)
            if retrieved == test_value:
                console.print(f"  [green]✓[/green] Write/Read test passed")
                client.delete(test_key)
            else:
                console.print(f"  [yellow]Warning: Write/Read test failed[/yellow]")

            # Display current stats
            keyspace_info = info.get(f"db{db}", {})
            if keyspace_info:
                console.print(f"  Current keys in DB {db}: {keyspace_info.get('keys', 0)}")

            console.print("[green]✓ Redis initialization complete[/green]")
            self.results["redis"] = {
                "status": "success",
                "message": f"Host: {host}:{port}, DB: {db}",
            }
            return True

        except redis.ConnectionError as e:
            error_msg = f"Connection failed: {str(e)}"
            console.print(f"[red]✗ {error_msg}[/red]")
            console.print("  Ensure Redis is running and accessible")
            self.results["redis"] = {"status": "failed", "message": error_msg}
            return False

        except redis.AuthenticationError:
            error_msg = "Authentication failed - check Redis password"
            console.print(f"[red]✗ {error_msg}[/red]")
            self.results["redis"] = {"status": "failed", "message": error_msg}
            return False

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            console.print(f"[red]✗ {error_msg}[/red]")
            logger.exception("Redis initialization failed")
            self.results["redis"] = {"status": "failed", "message": error_msg}
            return False

    def run_all(self) -> bool:
        """
        Run all database initializations.

        Returns:
            True if all successful
        """
        success_count = 0
        total_count = 3

        # Initialize Cosmos DB (PMG)
        if self.init_cosmos_db():
            success_count += 1

        # Initialize MongoDB
        if self.init_mongodb():
            success_count += 1

        # Initialize Redis
        if self.init_redis():
            success_count += 1

        return success_count == total_count

    def print_summary(self):
        """Print initialization summary."""
        console.print("\n" + "=" * 60)
        console.print("[bold]Database Initialization Summary[/bold]")
        console.print("=" * 60)

        # Create summary table
        table = Table(show_header=True, header_style="bold")
        table.add_column("Database", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Details", style="dim")

        for db_name, result in self.results.items():
            status = result["status"]
            message = result["message"]

            if status == "success":
                status_display = "[green]✓ Success[/green]"
            elif status == "failed":
                status_display = "[red]✗ Failed[/red]"
            else:
                status_display = "[yellow]○ Pending[/yellow]"

            # Format database name
            db_display = db_name.replace("_", " ").title()

            table.add_row(db_display, status_display, message)

        console.print(table)

        # Overall status
        all_success = all(r["status"] == "success" for r in self.results.values())
        if all_success:
            console.print("\n[green]All databases initialized successfully![/green]")
        else:
            console.print("\n[yellow]Some databases failed to initialize.[/yellow]")
            console.print("Check the error messages above for details.")


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Use default config
        script_dir = Path(__file__).parent
        config_path = script_dir.parent / "configs" / "default_config.yaml"

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Substitute environment variables
    return substitute_env_vars(config)


def substitute_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively substitute environment variables in configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with environment variables substituted
    """
    import re

    def substitute_value(value):
        if isinstance(value, str):
            # Pattern: ${VAR_NAME} or ${VAR_NAME:-default}
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
        description="Initialize databases for SAP_LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (default: configs/default_config.yaml)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate connections without making changes",
    )

    parser.add_argument(
        "--database",
        choices=["cosmos", "mongodb", "redis"],
        help="Initialize specific database only",
    )

    args = parser.parse_args()

    # Display banner
    console.print(
        Panel.fit(
            "[bold cyan]SAP_LLM Database Initializer[/bold cyan]\n"
            "Initialize Cosmos DB, MongoDB, and Redis",
            border_style="cyan",
        )
    )

    try:
        # Load configuration
        console.print("\n[bold]Loading configuration...[/bold]")
        config = load_config(args.config)
        console.print("[green]✓ Configuration loaded[/green]")

        if args.dry_run:
            console.print("\n[yellow]Running in DRY RUN mode - no changes will be made[/yellow]")

        # Initialize databases
        initializer = DatabaseInitializer(config, dry_run=args.dry_run)

        if args.database:
            # Initialize specific database
            if args.database == "cosmos":
                success = initializer.init_cosmos_db()
            elif args.database == "mongodb":
                success = initializer.init_mongodb()
            elif args.database == "redis":
                success = initializer.init_redis()
        else:
            # Initialize all databases
            success = initializer.run_all()

        # Print summary
        initializer.print_summary()

        return 0 if success else 1

    except FileNotFoundError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        return 1

    except KeyboardInterrupt:
        console.print("\n[yellow]Initialization interrupted by user.[/yellow]")
        return 130

    except Exception as e:
        console.print(f"\n[red]Unexpected error: {e}[/red]")
        logger.exception("Database initialization failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
