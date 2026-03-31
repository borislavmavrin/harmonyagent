import random
import time
from dataclasses import asdict

from jinja2 import StrictUndefined, Template
from openai_harmony import (
    Author,
    HarmonyEncodingName,
    Message,
    RenderConversationConfig,
    Role,
    load_harmony_encoding,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from harmonyagent.agents.agent_config import AgentConfig
from harmonyagent.agents.exceptions.agent_exceptions import *
from harmonyagent.agents.harmony_core.harmony_parsers import (
    create_prompt_text,
    get_final_message,
    get_reasoning_message,
    get_tool_args,
    get_tool_call,
    get_tool_name,
    parse_response_text,
    validate_generation_length,
    validate_messages,
)
from harmonyagent.agents.harmony_core.sys_dev_message import *
from harmonyagent.agents.tools.tool_registry import TOOL_REGISTRY
from harmonyagent.agents.tools.utils import pick_tool
from harmonyagent.domain_model import Environment, Model
from harmonyagent.utils.log import logger


def _raise_terminating_exception(retry_state):
    last_root_exception = retry_state.outcome.exception()
    raise RetrialsExceeded(retry_state.attempt_number, str(last_root_exception.__class__), str(last_root_exception))


def _before_sleep_with_prefix(retry_state):
    self = retry_state.args[0]
    last_root_exception = retry_state.outcome.exception()
    logger.warning(
        f"{self._log_prefix}retrying query, attempt {retry_state.attempt_number} exception: {last_root_exception.__class__} {last_root_exception}"
    )


def _linear_schedule(current_temp: float, current_context_size: int, max_context_size: int) -> float:
    return current_temp - 0.5 * current_temp * current_context_size / max_context_size


class HarmonyAgent:
    def __init__(self, model: Model, env: Environment, task_id: str, config: AgentConfig | None = None):
        self.config = config if config is not None else AgentConfig()
        self._messages: list[Message] = []
        self.model = model
        self.env = env
        self.instance_id = task_id
        self.enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.agent_step = 0
        self.tool_registry: dict = {}
        for tool_name in self.config.tools:
            self.tool_registry[tool_name] = TOOL_REGISTRY[tool_name](env)

    @property
    def _log_prefix(self) -> str:
        return f"{self.instance_id} step={self.agent_step} calls={self.model.n_calls} "

    def render_template(self, template: str, **kwargs) -> str:
        template_vars = asdict(self.config)
        return Template(template, undefined=StrictUndefined).render(**kwargs, **template_vars)

    @property
    def messages(self) -> list[dict]:
        return [m.to_dict() for m in self._messages]

    def run(self, task: str) -> tuple[str, str]:
        """Run step() until agent is finished. Return exit status & message"""
        self.agent_step = 0
        self._messages = []
        self._messages.append(
            get_system_message(
                system_instructions=self.config.system_instructions if self.config.system_instructions else None,
                reasoning_effort=self.config.reasoning_effort,
                include_repo_browser_tools=True
                if [tool for tool in self.config.tools if "repo_browser." in tool]
                else False,
                include_container_tools=True if [tool for tool in self.config.tools if "container." in tool] else False,
            )
        )
        self._messages.append(
            get_developer_message(
                self.config.developer_instructions if self.config.developer_instructions else None,
            )
        )
        self._messages.append(
            Message.from_role_and_content(Role.USER, self.render_template(self.config.instance_template, task=task))
        )
        logger.info(
            f"{self._log_prefix}task: {repr(self.render_template(self.config.instance_template, task=task)[: self.config.log_verbosity])}"
        )
        while True:
            try:
                self.step()
            except TerminatingException as e:
                logger.info(
                    f"{self._log_prefix}TerminatingException: {type(e).__name__} {str(e)[: self.config.log_verbosity]}"
                )
                return type(e).__name__, str(e)

    def step(self) -> None:
        """Query the LM, execute the action, return the observation."""

        received_messages, observation_messages, is_final = self.query()
        self._messages.extend(received_messages)
        self._messages.extend(observation_messages)
        if is_final:
            raise Submitted(received_messages)

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_fixed(2),
        before_sleep=_before_sleep_with_prefix,
        retry=retry_if_exception_type(NonTerminatingException),
        retry_error_callback=_raise_terminating_exception,
    )  # query is stateless - does not save messages
    def query(self) -> tuple[list[Message], list[Message], bool]:
        """Query the model and return the response."""
        if 0 < self.config.step_limit <= self.model.n_calls:
            raise LimitsExceeded("step_limit", self.model.n_calls, self.config.step_limit)
        if 0 < self.config.cost_limit <= self.model.cost:
            raise LimitsExceeded("cost_limit", self.model.cost, self.config.cost_limit)
        if self.config.delay:
            random_delay = random.random() * self.config.delay
            time.sleep(random_delay)

        convo_config = RenderConversationConfig()
        convo_config.auto_drop_analysis = False
        prompt_text, prompt_tokens = create_prompt_text(
            self._messages, self.enc, convo_config, self.config.max_context_window
        )

        if self.config.linear_schedule_temp:
            self.model.config.temperature = _linear_schedule(
                self.model.config.temperature, len(prompt_tokens), self.config.max_context_window
            )
        logger.info(f"{self._log_prefix}temperature: {self.model.config.temperature}")
        start_time = time.time()
        response = self.model.query(
            prompt_text,
            stop_token_ids=self.enc.stop_tokens_for_assistant_actions(),
            max_tokens=min(self.config.max_context_window - len(prompt_tokens), self.config.max_new_tokens),
        )
        self.agent_step += 1
        response_text = response["choices"][0]["text"]
        logger.info(f"{self._log_prefix}response_text: {repr(response_text[: self.config.log_verbosity])}")
        response_tokens = self.enc.encode(response_text, allowed_special="all")
        llm_time = time.time() - start_time
        logger.info(
            f"{self._log_prefix}num context tokens: {len(prompt_tokens)} generated tokens: {len(response_tokens)} time taken: {llm_time: .2f} sec {len(response_tokens) / llm_time: .0f} t/s finish_reason: {response['choices'][0]['finish_reason']}"
        )

        validate_generation_length(response, response_tokens, self.config.max_new_tokens)

        received_messages = parse_response_text(self.enc, response_text)
        validate_messages(received_messages)

        reasoning = get_reasoning_message(received_messages)
        if reasoning:
            logger.info(f"{self._log_prefix}reasoning: {repr(reasoning.content[0].text[: self.config.log_verbosity])}")

        tool_call_message = get_tool_call(received_messages)
        final_message = get_final_message(received_messages)

        if tool_call_message and final_message:
            raise ToolCallAndFinalMessage(tool_call_message, final_message)
        elif not tool_call_message and not final_message:
            raise NoToolCallNoFinalMessage(received_messages)

        if final_message:
            logger.info(
                f"{self._log_prefix}final_message: {repr(final_message.content[0].text[: self.config.log_verbosity])}"
            )
            return (received_messages, [], True)

        # tool handling starts
        tool_name = get_tool_name(tool_call_message)
        logger.info(f"{self._log_prefix}tool called: {repr(tool_name)}")
        tool_args = get_tool_args(tool_call_message)
        logger.info(
            f"{self._log_prefix}tool: {tool_name} received arguments: {repr(tool_args[: self.config.log_verbosity])}"
        )

        tool = pick_tool(tool_name, self.tool_registry)

        try:
            start_time = time.time()
            tool_result = tool.use(tool_args)
            tool_time = time.time() - start_time
            observation = self.render_template(self.config.action_observation_template, output=tool_result)
            logger.info(
                f"{self._log_prefix}tool: {tool_name} return code: {tool_result['returncode']} time taken: {tool_time: .2f} sec output: {repr(tool_result['output'][: self.config.log_verbosity])}"
            )
            logger.info(
                f"{self._log_prefix}tool: {tool_name} observation: {repr(observation[: self.config.log_verbosity])}"
            )
        except ExecutionTimeoutError as e:
            observation = self.render_template(self.config.timeout_template, output=e.output, timeout=e.timeout)
            logger.info(
                f"{self._log_prefix}execution timed out after {e.timeout}s cmd={repr(e.cmd[: self.config.log_verbosity])}: {repr(observation[: self.config.log_verbosity])}"
            )
        # tool handling ends

        # create tool_call response message starts
        observation_message = Message.from_author_and_content(
            Author.new(Role.TOOL, tool_call_message.recipient),
            observation,
        ).with_recipient("assistant")
        if tool_call_message.channel:
            observation_message = observation_message.with_channel(tool_call_message.channel)
        # create tool_call response message ends

        return (received_messages, [observation_message], False)
