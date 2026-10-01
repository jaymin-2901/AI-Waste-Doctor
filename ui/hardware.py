"""Safe output and sensor adapters for the science-fair display."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class OutputState:
    """Current simulated or physical output state."""

    light: str = "off"
    metal_detected: bool = False
    status: str = "SIMULATION"


class LightOutput:
    """Output adapter with a safe simulation default."""

    def __init__(self, simulation: bool = True):
        self.simulation = simulation
        self.state = OutputState(status="SIMULATION" if simulation else "UNCONFIGURED")

    def set_category(self, category: str) -> None:
        """Activate the light mapped to a waste category."""
        self.state.light = {
            "Recyclable": "blue",
            "Dry Waste": "yellow",
            "Wet Waste": "green",
        }.get(category, "off")

    def clear(self) -> None:
        """Turn all category outputs off."""
        self.state.light = "off"

    def close(self) -> None:
        """Release output resources and leave outputs off."""
        self.clear()


class MetalSensor:
    """Optional sensor boundary; returns no reading until hardware is configured."""

    def __init__(self):
        self.state = OutputState(status="NOT CONNECTED")

    def read(self) -> Optional[bool]:
        """Return metal presence, or None when no sensor is connected."""
        return None

    def close(self) -> None:
        """Release sensor resources."""
        return None


class SimulatedMetalSensor(MetalSensor):
    """Deterministic sensor for UI and automated tests."""

    def __init__(self, detected: bool = False):
        super().__init__()
        self.detected = detected
        self.state.status = "SIMULATION"

    def read(self) -> bool:
        self.state.metal_detected = self.detected
        return self.detected
