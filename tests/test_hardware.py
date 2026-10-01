"""Tests for science-fair output and sensor adapters."""

from ui.hardware import LightOutput, SimulatedMetalSensor


def test_light_output_maps_categories_and_clears():
    output = LightOutput()

    output.set_category("Recyclable")
    assert output.state.light == "blue"
    output.set_category("Dry Waste")
    assert output.state.light == "yellow"
    output.set_category("Wet Waste")
    assert output.state.light == "green"
    output.clear()
    assert output.state.light == "off"


def test_simulated_metal_sensor_is_explicit():
    sensor = SimulatedMetalSensor(detected=True)

    assert sensor.read() is True
    assert sensor.state.status == "SIMULATION"
