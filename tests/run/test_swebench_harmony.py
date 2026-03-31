from unittest.mock import patch

from harmonyagent.config.utils import config_dir
from harmonyagent.models.test_models import TapeModel, TapeModelConfig
from harmonyagent.run.swebench_harmony import main


def test_harmony(tmp_path):
    """Test that exceptions during agent.run() are properly handled and recorded"""
    with (
        patch("harmonyagent.run.swebench_harmony.get_model_class", return_value=TapeModel),
        patch("harmonyagent.run.swebench_harmony.get_model_class_config", return_value=TapeModelConfig),
    ):
        main(
            subset="verified",
            split="test",
            slice_spec="",
            filter_spec="",
            instance_names_from_file="bash-only-20.txt",
            output=str(tmp_path),
            workers=1,
            config_path=config_dir / "swebench_harmony.yaml",
            environment_class="docker",
        )
