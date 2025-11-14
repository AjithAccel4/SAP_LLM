"""
Example: APOP Event-Driven Document Processing Workflow.

This example demonstrates:
1. CloudEvents bus setup with Kafka
2. Autonomous stage agents registration
3. End-to-end document processing workflow
4. Event replay and debugging
5. Multi-agent coordination

Workflow:
Document Received ’ Preprocessing ’ Classification ’ Extraction ’
Quality Control ’ Business Rules ’ Post-Processing ’ Complete
"""

import logging
import asyncio
from pathlib import Path

from sap_llm.apop.cloudevents_bus import CloudEventsBus, CloudEvent
from sap_llm.apop.stage_agents import (
    PreprocessingAgent,
    ClassificationAgent,
    ExtractionAgent,
    QualityControlAgent,
    BusinessRulesAgent,
    PostProcessingAgent,
)
from sap_llm.apop.orchestrator import AgenticOrchestrator
from sap_llm.pmg.graph_client import ProcessMemoryGraph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def setup_event_bus():
    """Setup CloudEvents bus with Kafka."""

    logger.info("Setting up CloudEvents bus")

    # Initialize event bus
    bus = CloudEventsBus(
        kafka_brokers="localhost:9092",
        topic_prefix="sap_llm",
        consumer_group="sap_llm_processors",
        enable_dlq=True,
    )

    logger.info("CloudEvents bus initialized")

    return bus


async def register_agents(bus: CloudEventsBus):
    """Register autonomous stage agents."""

    logger.info("Registering autonomous stage agents")

    # Create agents
    agents = [
        PreprocessingAgent(),
        ClassificationAgent(),
        ExtractionAgent(),
        QualityControlAgent(quality_threshold=0.8),
        BusinessRulesAgent(),
        PostProcessingAgent(),
    ]

    # Register event handlers
    for agent in agents:
        # Create handler for each agent
        async def create_handler(agent_instance):
            async def handler(event: CloudEvent):
                # Convert CloudEvent to APOP envelope
                from sap_llm.apop.envelope import create_envelope

                envelope = create_envelope(
                    source=event.source,
                    event_type=event.type,
                    data=event.data or {},
                    correlation_id=event.correlation_id,
                    next_action_hint=event.next_action_hint,
                )

                # Process with agent
                result_envelope = await agent_instance.handle_envelope(envelope)

                if result_envelope:
                    # Publish result as new CloudEvent
                    result_event = bus.create_event(
                        source=agent_instance.agent_name,
                        event_type=result_envelope.type,
                        data=result_envelope.data,
                        correlation_id=result_envelope.correlation_id,
                        next_action_hint=result_envelope.next_action_hint,
                    )

                    await bus.publish(result_event)

            return handler

        # Subscribe to event types
        handler = await create_handler(agent)
        await bus.subscribe(agent.subscribes_to, handler)

        logger.info(f"Registered: {agent.agent_name}")

    logger.info(f"Registered {len(agents)} agents")

    return agents


async def process_document(bus: CloudEventsBus, document_path: str):
    """
    Process a single document through the workflow.

    Args:
        bus: CloudEvents bus
        document_path: Path to document
    """
    logger.info(f"Processing document: {document_path}")

    # Create initial event
    event = bus.create_event(
        source="document_ingestion",
        event_type="com.sap.document.received",
        data={
            "document_id": f"DOC_{Path(document_path).stem}",
            "document_path": document_path,
            "document_format": Path(document_path).suffix,
            "ingestion_timestamp": asyncio.get_event_loop().time(),
        },
        correlation_id=f"workflow_{Path(document_path).stem}",
        next_action_hint="com.sap.document.preprocessed",
    )

    # Publish to event bus
    success = await bus.publish(event)

    if success:
        logger.info(f"Document published to event bus: {event.id}")
    else:
        logger.error(f"Failed to publish document: {document_path}")

    return success


async def batch_process_documents(bus: CloudEventsBus, document_paths: list):
    """
    Process multiple documents in parallel.

    Args:
        bus: CloudEvents bus
        document_paths: List of document paths
    """
    logger.info(f"Batch processing {len(document_paths)} documents")

    tasks = [
        process_document(bus, doc_path)
        for doc_path in document_paths
    ]

    results = await asyncio.gather(*tasks)

    successful = sum(results)
    logger.info(f"Batch processing complete: {successful}/{len(document_paths)} successful")

    return results


async def replay_workflow(bus: CloudEventsBus, correlation_id: str):
    """
    Replay a workflow by correlation ID.

    Args:
        bus: CloudEvents bus
        correlation_id: Workflow correlation ID
    """
    logger.info(f"Replaying workflow: {correlation_id}")

    # Define event types in workflow order
    event_types = [
        "com.sap.document.received",
        "com.sap.document.preprocessed",
        "com.sap.document.classified",
        "com.sap.document.extracted",
        "com.sap.document.quality_checked",
        "com.sap.document.rules_validated",
        "com.sap.document.completed",
    ]

    # Handler to collect events
    collected_events = []

    async def collect_handler(event: CloudEvent):
        if event.correlation_id == correlation_id:
            collected_events.append(event)
            logger.info(f"Collected event: {event.type} (id={event.id})")

    # Replay events
    await bus.replay_events(
        event_types=event_types,
        start_time=None,  # From beginning
        handler=collect_handler,
    )

    logger.info(f"Replay complete: collected {len(collected_events)} events")

    # Print workflow trace
    print("\nWorkflow Trace:")
    print("=" * 80)
    for i, event in enumerate(collected_events, 1):
        print(f"{i}. {event.type}")
        print(f"   Source: {event.source}")
        print(f"   Time: {event.time}")
        print(f"   Data: {event.data}")
        print()

    return collected_events


async def monitor_workflow_status(bus: CloudEventsBus, correlation_id: str, timeout: int = 60):
    """
    Monitor workflow status until completion or timeout.

    Args:
        bus: CloudEvents bus
        correlation_id: Workflow correlation ID
        timeout: Timeout in seconds
    """
    logger.info(f"Monitoring workflow: {correlation_id}")

    workflow_status = {
        "correlation_id": correlation_id,
        "status": "in_progress",
        "completed_stages": [],
        "current_stage": None,
        "errors": [],
    }

    async def status_handler(event: CloudEvent):
        if event.correlation_id == correlation_id:
            workflow_status["current_stage"] = event.type
            workflow_status["completed_stages"].append(event.type)

            if event.type == "com.sap.document.completed":
                workflow_status["status"] = "completed"
            elif event.type == "error":
                workflow_status["status"] = "error"
                workflow_status["errors"].append(event.data.get("error"))

            logger.info(f"Stage completed: {event.type}")

    # Subscribe to all event types
    all_event_types = [
        "com.sap.document.received",
        "com.sap.document.preprocessed",
        "com.sap.document.classified",
        "com.sap.document.extracted",
        "com.sap.document.quality_checked",
        "com.sap.document.rules_validated",
        "com.sap.document.completed",
        "error",
    ]

    await bus.subscribe(all_event_types, status_handler)

    # Wait for completion or timeout
    start_time = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start_time < timeout:
        if workflow_status["status"] in ["completed", "error"]:
            break
        await asyncio.sleep(1)

    if workflow_status["status"] == "in_progress":
        logger.warning(f"Workflow monitoring timed out after {timeout}s")
        workflow_status["status"] = "timeout"

    logger.info(f"Workflow status: {workflow_status}")

    return workflow_status


async def main_single_document():
    """Example: Process a single document."""

    logger.info("=" * 80)
    logger.info("APOP Single Document Processing Example")
    logger.info("=" * 80)

    # Setup event bus
    bus = await setup_event_bus()

    # Register agents
    agents = await register_agents(bus)

    # Start consuming events (in background)
    consumer_task = asyncio.create_task(bus.start_consuming())

    # Give consumers time to start
    await asyncio.sleep(2)

    # Process document
    document_path = "./data/invoices/invoice_001.pdf"
    await process_document(bus, document_path)

    # Monitor workflow
    correlation_id = f"workflow_{Path(document_path).stem}"
    status = await monitor_workflow_status(bus, correlation_id, timeout=60)

    logger.info(f"Final status: {status}")

    # Stop consumer
    consumer_task.cancel()
    bus.close()


async def main_batch_processing():
    """Example: Batch process multiple documents."""

    logger.info("=" * 80)
    logger.info("APOP Batch Document Processing Example")
    logger.info("=" * 80)

    # Setup event bus
    bus = await setup_event_bus()

    # Register agents
    agents = await register_agents(bus)

    # Start consuming events
    consumer_task = asyncio.create_task(bus.start_consuming())
    await asyncio.sleep(2)

    # Batch process documents
    document_paths = [
        "./data/invoices/invoice_001.pdf",
        "./data/invoices/invoice_002.pdf",
        "./data/purchase_orders/po_001.pdf",
        "./data/delivery_notes/dn_001.pdf",
    ]

    results = await batch_process_documents(bus, document_paths)

    logger.info(f"Batch processing results: {results}")

    # Wait for processing to complete
    await asyncio.sleep(30)

    # Stop consumer
    consumer_task.cancel()
    bus.close()


async def main_replay_workflow():
    """Example: Replay and debug a workflow."""

    logger.info("=" * 80)
    logger.info("APOP Workflow Replay Example")
    logger.info("=" * 80)

    # Setup event bus
    bus = await setup_event_bus()

    # Replay workflow
    correlation_id = "workflow_invoice_001"
    events = await replay_workflow(bus, correlation_id)

    logger.info(f"Replayed {len(events)} events")

    bus.close()


def main():
    """Run APOP examples."""

    examples = {
        "1": ("Single Document Processing", main_single_document),
        "2": ("Batch Processing", main_batch_processing),
        "3": ("Workflow Replay", main_replay_workflow),
    }

    print("\nAPOP Workflow Examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")

    choice = input("\nSelect example to run (1-3): ").strip()

    if choice in examples:
        name, func = examples[choice]
        logger.info(f"Running: {name}")
        asyncio.run(func())
    else:
        logger.error(f"Invalid choice: {choice}")


if __name__ == "__main__":
    main()
