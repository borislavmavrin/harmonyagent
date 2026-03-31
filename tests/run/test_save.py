import json
import tempfile
from pathlib import Path

from openai_harmony import Message, Role

from harmonyagent.agents.harmony_agent import HarmonyAgent
from harmonyagent.environments.local import LocalEnvironment, LocalEnvironmentConfig
from harmonyagent.models.test_models import TapeModel, TapeModelConfig
from harmonyagent.run.save import save_traj


def test_save_traj_includes_class_names():
    """Test that save_traj includes the full class names with import paths."""
    # Create a simple agent setup
    model = TapeModel(config=TapeModelConfig(tape=[]))
    env = LocalEnvironment(LocalEnvironmentConfig())
    agent = HarmonyAgent(model, env, task_id="test")

    # Populate messages using harmony Message objects
    agent._messages = [
        Message.from_role_and_content(Role.SYSTEM, "test system message"),
        Message.from_role_and_content(Role.USER, "test user message"),
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "test_trajectory.json"

        # Save the trajectory
        save_traj(agent, temp_path, exit_status="Submitted", result="test result", print_path=False)

        # Load the saved trajectory
        with temp_path.open() as f:
            saved_data = json.load(f)

        # Verify the structure
        assert "info" in saved_data
        assert "config" in saved_data["info"]

        config = saved_data["info"]["config"]

        # Verify all three class types are present with correct import paths
        assert "agent_type" in config
        assert "model_type" in config
        assert "environment_type" in config

        # Verify the actual class names with module paths
        assert config["agent_type"] == "harmonyagent.agents.harmony_agent.HarmonyAgent"
        assert config["model_type"] == "harmonyagent.models.test_models.TapeModel"
        assert config["environment_type"] == "harmonyagent.environments.local.LocalEnvironment"

        # Verify other expected data is still present
        assert saved_data["info"]["exit_status"] == "Submitted"
        assert saved_data["info"]["submission"] == "test result"
        assert saved_data["trajectory_format"] == "mini-swe-agent-1"


def test_save_traj_with_none_agent():
    """Test that save_traj works correctly when agent is None."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "test_trajectory.json"

        # Save with None agent
        save_traj(None, temp_path, exit_status="Failed", result="no agent", print_path=False)

        # Load the saved trajectory
        with temp_path.open() as f:
            saved_data = json.load(f)

        # Verify basic structure
        assert "info" in saved_data
        assert saved_data["info"]["exit_status"] == "Failed"
        assert saved_data["info"]["submission"] == "no agent"

        # Verify class types are not present when agent is None (since config is not present)
        # When agent is None, there should be no config section at all

        # Verify config is not present when agent is None
        assert "config" not in saved_data["info"]
