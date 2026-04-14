from __future__ import annotations

from abc import ABC, abstractmethod

from aingram.capture.config import CaptureConfig
from aingram.capture.filters import apply_filters
from aingram.capture.types import CaptureRecord, ToolHealth


class ToolAdapter(ABC):
    tool_name: str

    def __init__(self, config: CaptureConfig) -> None:
        self.config = config

    @abstractmethod
    def parse_payload(self, raw: dict) -> list[CaptureRecord]:
        """Convert tool-specific JSON into common schema records."""

    @abstractmethod
    def get_installation_instructions(self) -> str:
        """Return setup instructions for this tool's native integration."""

    @abstractmethod
    def health_check(self) -> ToolHealth:
        """Check if this tool's integration is functioning."""

    def apply_filters(self, record: CaptureRecord) -> CaptureRecord | None:
        return apply_filters(record, self.config)
