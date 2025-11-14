"""
CloudEvents Event Bus for APOP.

Implements CloudEvents v1.0 specification with Kafka backend for event-driven orchestration.

Features:
- CloudEvents v1.0 compliant event format
- Kafka event bus for distributed messaging
- Topic-based event routing
- Dead letter queue for failed events
- Event replay capability
- At-least-once delivery guarantee

CloudEvents spec: https://github.com/cloudevents/spec/blob/v1.0.2/cloudevents/spec.md
"""

import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


@dataclass
class CloudEvent:
    """
    CloudEvents v1.0 event format.

    Required attributes:
    - id: Unique event identifier
    - source: Event producer identifier
    - type: Event type (e.g., "com.sap.document.classified")
    - specversion: CloudEvents spec version (1.0)

    Optional attributes:
    - datacontenttype: Content type of data
    - dataschema: Schema URI
    - subject: Subject of the event
    - time: Event timestamp
    - data: Event payload
    """
    id: str
    source: str
    type: str
    specversion: str = "1.0"
    datacontenttype: Optional[str] = "application/json"
    dataschema: Optional[str] = None
    subject: Optional[str] = None
    time: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

    # Extensions (non-standard attributes)
    correlation_id: Optional[str] = None
    next_action_hint: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CloudEvent":
        """Create CloudEvent from dictionary."""
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        # Remove None values
        return {k: v for k, v in d.items() if v is not None}

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "CloudEvent":
        """Create CloudEvent from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def validate(self) -> bool:
        """Validate required attributes."""
        required = ["id", "source", "type", "specversion"]
        for attr in required:
            if getattr(self, attr) is None:
                return False
        return True


class CloudEventsBus:
    """
    CloudEvents event bus with Kafka backend.

    Provides pub/sub messaging for distributed APOP orchestration.

    Architecture:
    - Producers publish CloudEvents to topics
    - Consumers subscribe to topics and process events
    - Dead letter queue for failed events
    - Event replay from Kafka log
    """

    def __init__(self,
                 kafka_brokers: str = "localhost:9092",
                 topic_prefix: str = "sap_llm",
                 consumer_group: str = "sap_llm_group",
                 enable_dlq: bool = True):
        """
        Initialize CloudEvents bus.

        Args:
            kafka_brokers: Kafka broker addresses (comma-separated)
            topic_prefix: Prefix for all topics
            consumer_group: Consumer group ID
            enable_dlq: Enable dead letter queue
        """
        self.kafka_brokers = kafka_brokers
        self.topic_prefix = topic_prefix
        self.consumer_group = consumer_group
        self.enable_dlq = enable_dlq

        # Event handlers: topic -> List[callback]
        self.handlers: Dict[str, List[Callable]] = {}

        # Kafka clients (lazy initialization)
        self._producer = None
        self._consumer = None
        self._admin_client = None

        logger.info(f"CloudEvents bus initialized: brokers={kafka_brokers}")

    def _get_producer(self):
        """Get or create Kafka producer."""
        if self._producer is None:
            try:
                from kafka import KafkaProducer
                self._producer = KafkaProducer(
                    bootstrap_servers=self.kafka_brokers.split(','),
                    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                    acks='all',  # Wait for all replicas
                    retries=3,
                    max_in_flight_requests_per_connection=1,  # Preserve ordering
                )
                logger.info("Kafka producer initialized")
            except ImportError:
                logger.error("kafka-python not installed. Install with: pip install kafka-python")
                raise
        return self._producer

    def _get_consumer(self):
        """Get or create Kafka consumer."""
        if self._consumer is None:
            try:
                from kafka import KafkaConsumer
                self._consumer = KafkaConsumer(
                    bootstrap_servers=self.kafka_brokers.split(','),
                    group_id=self.consumer_group,
                    value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                    auto_offset_reset='earliest',  # Start from beginning
                    enable_auto_commit=False,  # Manual commit for reliability
                    max_poll_records=10,
                )
                logger.info("Kafka consumer initialized")
            except ImportError:
                logger.error("kafka-python not installed. Install with: pip install kafka-python")
                raise
        return self._consumer

    def _get_admin_client(self):
        """Get or create Kafka admin client."""
        if self._admin_client is None:
            try:
                from kafka.admin import KafkaAdminClient
                self._admin_client = KafkaAdminClient(
                    bootstrap_servers=self.kafka_brokers.split(',')
                )
                logger.info("Kafka admin client initialized")
            except ImportError:
                logger.error("kafka-python not installed")
                raise
        return self._admin_client

    def _get_topic_name(self, event_type: str) -> str:
        """
        Get Kafka topic name for event type.

        Args:
            event_type: CloudEvents type (e.g., "com.sap.document.classified")

        Returns:
            Kafka topic name (e.g., "sap_llm.com.sap.document.classified")
        """
        return f"{self.topic_prefix}.{event_type}"

    async def publish(self, event: CloudEvent) -> bool:
        """
        Publish CloudEvent to Kafka.

        Args:
            event: CloudEvent to publish

        Returns:
            True if published successfully
        """
        if not event.validate():
            logger.error(f"Invalid CloudEvent: {event}")
            return False

        topic = self._get_topic_name(event.type)

        try:
            producer = self._get_producer()

            # Set timestamp if not set
            if event.time is None:
                event.time = datetime.utcnow().isoformat() + 'Z'

            # Publish to Kafka
            future = producer.send(topic, value=event.to_dict())
            record_metadata = future.get(timeout=10)  # Wait for acknowledgment

            logger.info(
                f"Published event: type={event.type}, id={event.id}, "
                f"topic={topic}, partition={record_metadata.partition}, "
                f"offset={record_metadata.offset}"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to publish event: {e}", exc_info=True)

            # Send to DLQ if enabled
            if self.enable_dlq:
                await self._send_to_dlq(event, str(e))

            return False

    async def subscribe(self, event_types: List[str], handler: Callable):
        """
        Subscribe to event types.

        Args:
            event_types: List of event types to subscribe to
            handler: Async callback function (event: CloudEvent) -> None
        """
        for event_type in event_types:
            if event_type not in self.handlers:
                self.handlers[event_type] = []

            self.handlers[event_type].append(handler)

            logger.info(f"Subscribed to event type: {event_type}")

    async def start_consuming(self):
        """
        Start consuming events from Kafka.

        Runs indefinitely, processing events and calling registered handlers.
        """
        consumer = self._get_consumer()

        # Subscribe to all registered topics
        topics = [self._get_topic_name(event_type) for event_type in self.handlers.keys()]
        consumer.subscribe(topics)

        logger.info(f"Started consuming from topics: {topics}")

        try:
            while True:
                # Poll for messages
                message_batch = consumer.poll(timeout_ms=1000, max_records=10)

                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        # Parse CloudEvent
                        try:
                            event = CloudEvent.from_dict(message.value)

                            # Get handlers for this event type
                            handlers = self.handlers.get(event.type, [])

                            # Process event with all handlers
                            for handler in handlers:
                                try:
                                    await handler(event)
                                except Exception as e:
                                    logger.error(
                                        f"Handler failed for event {event.id}: {e}",
                                        exc_info=True
                                    )

                                    # Send to DLQ
                                    if self.enable_dlq:
                                        await self._send_to_dlq(event, str(e))

                            # Commit offset (manual commit for at-least-once delivery)
                            consumer.commit()

                        except Exception as e:
                            logger.error(f"Failed to process message: {e}", exc_info=True)

                # Small sleep to prevent busy loop
                await asyncio.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("Stopping consumer...")
        finally:
            consumer.close()

    async def _send_to_dlq(self, event: CloudEvent, error: str):
        """
        Send failed event to dead letter queue.

        Args:
            event: Failed CloudEvent
            error: Error message
        """
        dlq_topic = f"{self.topic_prefix}.dlq"

        try:
            producer = self._get_producer()

            # Create DLQ event with error info
            dlq_event = CloudEvent(
                id=str(uuid.uuid4()),
                source="cloudevents_bus.dlq",
                type="com.sap.event.failed",
                data={
                    "original_event": event.to_dict(),
                    "error": error,
                    "timestamp": datetime.utcnow().isoformat() + 'Z',
                }
            )

            producer.send(dlq_topic, value=dlq_event.to_dict())

            logger.warning(f"Sent event {event.id} to DLQ: {error}")

        except Exception as e:
            logger.error(f"Failed to send to DLQ: {e}", exc_info=True)

    def create_event(self,
                     source: str,
                     event_type: str,
                     data: Dict[str, Any],
                     correlation_id: Optional[str] = None,
                     next_action_hint: Optional[str] = None,
                     subject: Optional[str] = None) -> CloudEvent:
        """
        Create a CloudEvent.

        Args:
            source: Event source identifier
            event_type: Event type
            data: Event data
            correlation_id: Optional correlation ID
            next_action_hint: Optional routing hint
            subject: Optional subject

        Returns:
            CloudEvent
        """
        event = CloudEvent(
            id=str(uuid.uuid4()),
            source=source,
            type=event_type,
            time=datetime.utcnow().isoformat() + 'Z',
            data=data,
            correlation_id=correlation_id,
            next_action_hint=next_action_hint,
            subject=subject,
        )

        return event

    async def replay_events(self,
                            event_types: List[str],
                            start_time: Optional[datetime] = None,
                            handler: Optional[Callable] = None):
        """
        Replay events from Kafka log.

        Args:
            event_types: Event types to replay
            start_time: Start from this timestamp
            handler: Optional handler (uses registered handlers if None)
        """
        logger.info(f"Replaying events: types={event_types}, start_time={start_time}")

        # Create dedicated consumer for replay
        try:
            from kafka import KafkaConsumer
            from kafka import TopicPartition

            replay_consumer = KafkaConsumer(
                bootstrap_servers=self.kafka_brokers.split(','),
                group_id=f"{self.consumer_group}_replay_{uuid.uuid4()}",
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                auto_offset_reset='earliest',
            )

            # Subscribe to topics
            topics = [self._get_topic_name(event_type) for event_type in event_types]
            replay_consumer.subscribe(topics)

            # Seek to start time if specified
            if start_time:
                timestamp_ms = int(start_time.timestamp() * 1000)
                partitions = []
                for topic in topics:
                    topic_partitions = replay_consumer.partitions_for_topic(topic)
                    if topic_partitions:
                        for partition_id in topic_partitions:
                            partitions.append(TopicPartition(topic, partition_id))

                # Seek to timestamp
                for partition in partitions:
                    replay_consumer.seek(partition, timestamp_ms)

            # Process messages
            processed = 0
            for message in replay_consumer:
                event = CloudEvent.from_dict(message.value)

                # Use custom handler or registered handlers
                if handler:
                    await handler(event)
                else:
                    handlers = self.handlers.get(event.type, [])
                    for h in handlers:
                        await h(event)

                processed += 1

                # Stop at end
                if replay_consumer.position(
                    TopicPartition(message.topic, message.partition)
                ) >= replay_consumer.highwater(
                    TopicPartition(message.topic, message.partition)
                ):
                    break

            logger.info(f"Replayed {processed} events")

            replay_consumer.close()

        except Exception as e:
            logger.error(f"Failed to replay events: {e}", exc_info=True)

    def close(self):
        """Close Kafka connections."""
        if self._producer:
            self._producer.close()
        if self._consumer:
            self._consumer.close()
        if self._admin_client:
            self._admin_client.close()

        logger.info("CloudEvents bus closed")


# Example usage
if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)

    # Initialize bus
    bus = CloudEventsBus(
        kafka_brokers="localhost:9092",
        topic_prefix="sap_llm",
        consumer_group="sap_llm_group"
    )

    # Create sample event
    event = bus.create_event(
        source="document_ingestion",
        event_type="com.sap.document.received",
        data={
            "document_id": "DOC123",
            "document_type": "invoice",
            "path": "/uploads/invoice_123.pdf"
        },
        correlation_id="workflow_123",
        next_action_hint="preproc.ocr"
    )

    # Publish event
    async def test_publish():
        success = await bus.publish(event)
        print(f"Published: {success}")

    # Subscribe and consume
    async def handle_event(event: CloudEvent):
        print(f"Received event: type={event.type}, id={event.id}, data={event.data}")

    async def test_consume():
        await bus.subscribe(["com.sap.document.received"], handle_event)
        await bus.start_consuming()

    # Run test
    asyncio.run(test_publish())

    # Run consumer (in separate process/thread in production)
    # asyncio.run(test_consume())

    print("CloudEvents bus module loaded successfully")
