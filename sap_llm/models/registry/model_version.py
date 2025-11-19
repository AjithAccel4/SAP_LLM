"""
Model Version Management.

Implements semantic versioning for model lifecycle management.
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelVersion:
    """
    Semantic versioning for models.

    Format: major.minor.patch
    - major: Breaking architecture changes
    - minor: New features, non-breaking changes
    - patch: Bug fixes, retraining with same architecture
    """

    major: int
    minor: int
    patch: int

    @classmethod
    def from_string(cls, version_str: str) -> "ModelVersion":
        """
        Parse version string into ModelVersion.

        Args:
            version_str: Version string (e.g., "1.2.3")

        Returns:
            ModelVersion instance

        Raises:
            ValueError: If version string is invalid
        """
        pattern = r'^(\d+)\.(\d+)\.(\d+)$'
        match = re.match(pattern, version_str)

        if not match:
            raise ValueError(
                f"Invalid version string: {version_str}. "
                f"Expected format: major.minor.patch (e.g., 1.2.3)"
            )

        major, minor, patch = map(int, match.groups())
        return cls(major=major, minor=minor, patch=patch)

    def __str__(self) -> str:
        """Return semantic version string."""
        return f"{self.major}.{self.minor}.{self.patch}"

    def __repr__(self) -> str:
        """Return representation."""
        return f"ModelVersion({self})"

    def __lt__(self, other: "ModelVersion") -> bool:
        """Compare versions."""
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __le__(self, other: "ModelVersion") -> bool:
        """Compare versions."""
        return (self.major, self.minor, self.patch) <= (other.major, other.minor, other.patch)

    def __gt__(self, other: "ModelVersion") -> bool:
        """Compare versions."""
        return (self.major, self.minor, self.patch) > (other.major, other.minor, other.patch)

    def __ge__(self, other: "ModelVersion") -> bool:
        """Compare versions."""
        return (self.major, self.minor, self.patch) >= (other.major, other.minor, other.patch)

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, ModelVersion):
            return False
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)

    def increment_patch(self) -> "ModelVersion":
        """
        Increment patch version.

        Use for: Bug fixes, retraining with same architecture

        Returns:
            New ModelVersion with incremented patch
        """
        return ModelVersion(self.major, self.minor, self.patch + 1)

    def increment_minor(self) -> "ModelVersion":
        """
        Increment minor version and reset patch.

        Use for: New features, architecture improvements

        Returns:
            New ModelVersion with incremented minor
        """
        return ModelVersion(self.major, self.minor + 1, 0)

    def increment_major(self) -> "ModelVersion":
        """
        Increment major version and reset minor/patch.

        Use for: Breaking changes, major architecture overhaul

        Returns:
            New ModelVersion with incremented major
        """
        return ModelVersion(self.major + 1, 0, 0)

    def is_compatible_with(self, other: "ModelVersion") -> bool:
        """
        Check if versions are compatible (same major version).

        Args:
            other: Other version to compare

        Returns:
            True if compatible
        """
        return self.major == other.major


# Initial version for new models
INITIAL_VERSION = ModelVersion(1, 0, 0)
