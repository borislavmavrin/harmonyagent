import dataclasses
from dataclasses import dataclass
from pathlib import Path

import yaml
from jinja2 import StrictUndefined, Template

from harmonyagent.agents.agent_config import AgentConfig
from harmonyagent.environments.docker import DockerEnvironmentConfig
from harmonyagent.models.vllm_raw import VllmRawModelConfig


@dataclass
class MockOutput:
    """Mock output object for testing the template"""

    returncode: int
    output: str


SWEBENCH_CONFIG_PATH = Path(__file__).parent.parent.parent / "src" / "harmonyagent" / "config" / "swebench_harmony.yaml"


def test_agent_config_defaults_match_yaml():
    """Check that AgentConfig defaults match the swebench_harmony.yaml values."""
    with open(SWEBENCH_CONFIG_PATH) as f:
        yaml_config = yaml.safe_load(f)

    yaml_agent = yaml_config["agent"]
    defaults = {f.name: _get_field_default(f) for f in dataclasses.fields(AgentConfig)}

    for field_name, yaml_value in yaml_agent.items():
        assert field_name in defaults, f"YAML field '{field_name}' not found in AgentConfig"
        assert defaults[field_name] == yaml_value, (
            f"AgentConfig.{field_name}: default={defaults[field_name]!r} != yaml={yaml_value!r}"
        )

    extra = set(defaults) - set(yaml_agent)
    assert not extra, f"AgentConfig has fields not in YAML: {extra}"


def _get_field_default(f):
    """Get the default value for a dataclass field, calling default_factory if needed."""
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    return dataclasses.MISSING


def test_docker_config_defaults_match_yaml():
    """Check that DockerEnvironmentConfig defaults match the swebench_harmony.yaml values."""
    with open(SWEBENCH_CONFIG_PATH) as f:
        yaml_config = yaml.safe_load(f)

    yaml_env = yaml_config["environment"]

    for field_name, yaml_value in yaml_env.items():
        field = next((f for f in dataclasses.fields(DockerEnvironmentConfig) if f.name == field_name), None)
        assert field is not None, f"YAML field '{field_name}' not found in DockerEnvironmentConfig"
        default = _get_field_default(field)
        assert default is not dataclasses.MISSING, f"DockerEnvironmentConfig.{field_name} has no default"
        assert default == yaml_value, (
            f"DockerEnvironmentConfig.{field_name}: default={default!r} != yaml={yaml_value!r}"
        )

    docker_fields_with_defaults = {
        f.name for f in dataclasses.fields(DockerEnvironmentConfig) if _get_field_default(f) is not dataclasses.MISSING
    }
    extra = docker_fields_with_defaults - set(yaml_env)
    assert not extra, f"DockerEnvironmentConfig has fields not in YAML: {extra}"


def test_vllm_raw_config_defaults_match_yaml():
    """Check that VllmRawModelConfig defaults match the swebench_harmony.yaml values."""
    with open(SWEBENCH_CONFIG_PATH) as f:
        yaml_config = yaml.safe_load(f)

    yaml_model = yaml_config["model"]
    defaults = {f.name: f.default for f in dataclasses.fields(VllmRawModelConfig)}

    for field_name, yaml_value in yaml_model.items():
        assert field_name in defaults, f"YAML field '{field_name}' not found in VllmRawModelConfig"
        assert defaults[field_name] == yaml_value, (
            f"VllmRawModelConfig.{field_name}: default={defaults[field_name]!r} != yaml={yaml_value!r}"
        )

    extra = set(defaults) - set(yaml_model)
    assert not extra, f"VllmRawModelConfig has fields not in YAML: {extra}"


def test_action_observation_template_short_output():
    """Test that short output (< 10000 chars) is displayed in full"""
    # Load the swebench config
    config_path = Path(__file__).parent.parent.parent / "src" / "harmonyagent" / "config" / "swebench_harmony.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Extract the template
    template_str = config["agent"]["action_observation_template"]
    template = Template(template_str, undefined=StrictUndefined)

    # Create mock output with short content
    output = MockOutput(returncode=0, output="Success! Operation completed.\nWarning: minor issue")

    # Render the template
    result = template.render(output=output, cwd="/test")

    # Verify the result contains all parts and no truncation
    assert "<returncode>" in result
    assert "0" in result
    assert "<output>" in result
    assert "Success! Operation completed." in result
    assert "Warning: minor issue" in result
    assert "/test" in result

    # Should not contain truncation elements for short output
    assert "<output_head>" not in result
    assert "<elided_chars>" not in result
    assert "<output_tail>" not in result
    assert "<warning>" not in result


def test_action_observation_template_long_output():
    """Test that long output (> 10000 chars) is truncated with head/tail format"""
    # Load the swebench config
    config_path = Path(__file__).parent.parent.parent / "src" / "harmonyagent" / "config" / "swebench_harmony.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Extract the template
    template_str = config["agent"]["action_observation_template"]
    template = Template(template_str, undefined=StrictUndefined)

    # Create mock output with long content
    long_output = "A" * 8000 + "B" * 3000 + "\n" * 3  # 11000 characters total
    # Total will be > 10000 chars

    output = MockOutput(returncode=1, output=long_output)

    # Render the template
    result = template.render(output=output, cwd="/test")

    # Should contain truncation elements for long output
    assert "<warning>" in result
    assert (
        "The output of your last command was too long and it was truncated. The number of lines truncated: " in result
    )
    assert "The number of lines truncated: 3" in result

    # Should still contain the basic structure
    assert "<returncode>" in result
    assert "1" in result
    assert "/test" in result

    # Verify the head contains first part of output
    head_start = result.find("<output>")
    head_end = result.find("</output>")
    head_content = result[head_start:head_end]
    assert "AAAA" in head_content  # Should contain start of output

    # Verify the tail contains last part of output
    tail_start = result.find("<output>")
    tail_end = result.find("</output>")
    tail_content = result[tail_start:tail_end]
    assert "BBBB" in tail_content  # Should contain end of output


def test_action_observation_template_edge_case_exactly_10000_chars():
    """Test the boundary case where output is around 10000 characters"""
    # Load the swebench config
    config_path = Path(__file__).parent.parent.parent / "src" / "harmonyagent" / "config" / "swebench_harmony.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Extract the template
    template_str = config["agent"]["action_observation_template"]
    template = Template(template_str, undefined=StrictUndefined)

    # Use a large amount of data that will definitely exceed 10000 chars when rendered
    output = MockOutput(returncode=0, output="X" * 10000)

    # Render the template
    result = template.render(output=output, cwd="/test")

    # Should not use truncated format for large output
    assert "<warning>" not in result
    assert (
        "The output of your last command was too long and it was truncated. The number of lines truncated: "
        not in result
    )
    assert "The number of lines truncated: 1" not in result

    # The X's should still be present in head or tail
    assert "XXXX" in result


def test_action_observation_template_just_under_1000_chars():
    """Test that smaller output shows full output without truncation"""
    # Load the swebench config
    config_path = Path(__file__).parent.parent.parent / "src" / "harmonyagent" / "config" / "swebench_harmony.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Extract the template
    template_str = config["agent"]["action_observation_template"]
    template = Template(template_str, undefined=StrictUndefined)

    # Use a reasonably sized output that should be well under 1000 chars when rendered
    output = MockOutput(returncode=0, output="Y" * 800)

    # Render the template
    result = template.render(output=output, cwd="/test")

    # Should show full output without truncation
    assert "<output_head>" not in result
    assert "<elided_chars>" not in result
    assert "<output_tail>" not in result
    assert "<warning>" not in result
    assert "Y" * 800 in result
    assert "/test" in result
