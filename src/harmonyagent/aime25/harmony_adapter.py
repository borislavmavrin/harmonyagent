import json
import logging

from harmonyagent.agents.agent_config import AgentConfig
from harmonyagent.agents.harmony_agent import HarmonyAgent
from harmonyagent.environments.docker import DockerEnvironment, DockerEnvironmentConfig
from harmonyagent.models.vllm_raw import VllmRawModel, VllmRawModelConfig

PYTHON_PACKAGES = "sympy scikit-learn shapely matplotlib networkx fraction mpmath"
DEVELOPER_INSTRUCTIONS = """                                                                                                                                           
  You are solving competition math problems (AIME). You have a Python shell.                                                                                               
                                                                                                                                                                           
  ## Strategy                                                                                                                                                              
  - Reason through the problem analytically first.                                                                                                                       
  - Use Python to verify your analytical solution or brute-force check.
  - If your code errors, fix it quickly or switch to a different approach. Don't retry the same failing approach more than once.

  ## Shell usage
  - Run Python via: python3 -c "..." or python3 << 'EOF' ... EOF
  - sympy, numpy, scipy, shapely, networkx, mpmath, matplotlib are available.
  - Keep scripts short and focused. Print only the final result.

  ## Answer format
  - Your final answer must be a single non-negative integer inside \\boxed{}.
  - Write \\boxed{42}, not \\boxed{x=42} or \\boxed{42^\\circ}.
  """.strip()

logging.basicConfig(
    filename="logs/aime25.log",  # Name of the log file
    level=logging.INFO,  # Minimum log level to record (INFO, DEBUG, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log record format
    filemode="a",  # Mode: 'a' for append (default), 'w' for overwrite
)
logger = logging.getLogger(__name__)

agent_config = AgentConfig(
    system_instructions="",
    developer_instructions=DEVELOPER_INSTRUCTIONS,
    instance_template=f"You have the following packages installed in python: {PYTHON_PACKAGES}"
    + "\n Solve:\n {{ task }}",
    action_observation_template="{{ output.output }}",
    reasoning_effort="medium",
    tools=["container.exec"],
    step_limit=50,
    max_new_tokens=10 * 1_024,
    log_verbosity=1_000_000,
)

env_config = DockerEnvironmentConfig(image="python:3.11", timeout=120, cpus=8)
env = DockerEnvironment(env_config)
result = env.execute(f"pip install {PYTHON_PACKAGES}", timeout=60)
logger.info(f"pip install return code: {result['returncode']} result: {result['output']}")


class HarmonyAdapter:
    def __init__(self):

        model = VllmRawModel(VllmRawModelConfig)
        self.agent = HarmonyAgent(model, env, "", agent_config)

    def run(self, task):
        status, result = self.agent.run(task)
        if status == "Submitted":
            return json.loads(result)
        return []
