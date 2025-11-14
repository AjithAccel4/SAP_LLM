"""
Base class for all pipeline stages.

Provides common functionality and interface for all processing stages.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from sap_llm.utils.logger import get_logger
from sap_llm.utils.timer import Timer

logger = get_logger(__name__)


class BaseStage(ABC):
    """
    Base class for all pipeline stages.

    All stages must implement the process() method and can optionally
    override validate_input() and validate_output().
    """

    def __init__(self, config: Optional[Any] = None):
        """
        Initialize stage.

        Args:
            config: Stage-specific configuration
        """
        self.config = config
        self.stage_name = self.__class__.__name__
        logger.info(f"Initialized {self.stage_name}")

    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data through this stage.

        Args:
            input_data: Input dictionary

        Returns:
            Output dictionary with stage results

        Raises:
            ValueError: If input validation fails
            RuntimeError: If processing fails
        """
        pass

    def validate_input(self, input_data: Dict[str, Any]) -> None:
        """
        Validate input data.

        Args:
            input_data: Input to validate

        Raises:
            ValueError: If validation fails
        """
        if not isinstance(input_data, dict):
            raise ValueError(f"{self.stage_name}: input_data must be a dictionary")

    def validate_output(self, output_data: Dict[str, Any]) -> None:
        """
        Validate output data.

        Args:
            output_data: Output to validate

        Raises:
            ValueError: If validation fails
        """
        if not isinstance(output_data, dict):
            raise ValueError(f"{self.stage_name}: output_data must be a dictionary")

    def __call__(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute stage with timing and validation.

        Args:
            input_data: Input dictionary

        Returns:
            Output dictionary
        """
        with Timer(f"{self.stage_name}.process"):
            # Validate input
            self.validate_input(input_data)

            # Process
            output_data = self.process(input_data)

            # Validate output
            self.validate_output(output_data)

            return output_data
