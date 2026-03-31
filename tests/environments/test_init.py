from harmonyagent.environments.docker import DockerEnvironment
from harmonyagent.environments.local import LocalEnvironment
from harmonyagent.environments.utils import get_environment_class


class TestGetEnvironmentClass:
    def test_get_environment_class_local(self):
        assert get_environment_class("local") is LocalEnvironment

    def test_get_environment_class_docker(self):
        assert get_environment_class("docker") is DockerEnvironment
