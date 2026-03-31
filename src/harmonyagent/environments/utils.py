"""Environment implementations for mini-SWE-agent."""

from harmonyagent.domain_model import Environment
from harmonyagent.environments.docker import DockerEnvironment, DockerEnvironmentConfig
from harmonyagent.environments.local import LocalEnvironment, LocalEnvironmentConfig

ENVIRONMENT_CLASS_MAPPING = {
    "docker": DockerEnvironment,
    "local": LocalEnvironment,
}

ENVIRONMENT_CLASS_CONFIG_MAPPING = {
    "docker": DockerEnvironmentConfig,
    "local": LocalEnvironmentConfig,
}


def get_environment_class(environment_class: str) -> type[Environment]:
    return ENVIRONMENT_CLASS_MAPPING[environment_class]


def get_environment_class_config_class(environment_class: str) -> type:
    return ENVIRONMENT_CLASS_CONFIG_MAPPING[environment_class]
