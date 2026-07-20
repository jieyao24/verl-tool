"""
Clarify Tool for verl-tool - simulated-user responses for <clarify> turns.

Placeholder stand-in for M0/M1 (see rl-retrieval-clarification-agents' CLAUDE.md
§9 point 3 and MILESTONES.md M0): the simulated user replies with a
deterministic sentence built from the gold `key_entity` that
ambiguity_injector.py already attached to the example, instead of a real LLM
call. This unblocks the end-to-end smoke test without needing the vLLM
engine's OpenAI-compatible HTTP endpoint wired up yet.

NOT the final version. Before M2 (real Stage 2 training), swap
`_generate_reply` below for an HTTP call to the same vLLM engine's
OpenAI-compatible server (confirmed to exist at
verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py's `run_server`,
bound via `get_server_address()`) using a `CLARIFY_LLM_URL` env var, mirroring
how `search_retrieval.py` reads `RETRIEVER_URL`. That address is only known
inside the training driver today (see verl/verl/experimental/agent_loop/
agent_loop.py `self.server_addresses`) - it still needs to be exported for the
tool-server subprocess to read (e.g. via train.slurm), which is real but small
plumbing, not solved here.
"""
from .base import BaseTool, register_tool
import re
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


@register_tool
class ClarifyTool(BaseTool):
    # Must match this file's stem ("clarify_tool"), not the <clarify> tag
    # itself - get_tool_cls() in base.py imports verl_tool.servers.tools.{tool_type}
    # and looks up registered_tools[tool_type] using the same string, so file
    # name and tool_type have to agree (search_retrieval.py follows the same
    # rule: tool_type="search_retrieval", not "search").
    tool_type = "clarify_tool"

    def get_usage_inst(self):
        return (
            "You can ask a clarifying question by putting it between "
            "<clarify> and </clarify> tags."
        )

    def parse_action(self, action: str) -> Tuple[str, bool]:
        """Extract the clarifying question from <clarify>...</clarify> tags."""
        if "</clarify>" in action:
            matches = re.findall(r"<clarify>(.*?)</clarify>", action, re.DOTALL)
            if matches:
                return matches[-1].strip(), True
        return "", False

    def get_action_priority(self, action: str, extra_field: dict) -> int:
        # Matches search_retrieval.py's flat-100 convention for its own tag.
        # Known gap shared with that file: if a single turn somehow contains
        # both </search> and </clarify>, whichever tool's priority check runs
        # first wins the tie, not necessarily whichever tag is last in the
        # text (the "last tag wins" rule multihop_env.py/format_reward.py
        # actually use). Not fixed here - would need editing
        # search_retrieval.py too, and a well-format-rewarded policy should
        # rarely trigger this case in practice.
        if "</clarify>" in action:
            _, valid = self.parse_action(action)
            if valid:
                return 100
        return -1

    def _generate_reply(self, clarifying_question: str, extra_field: dict) -> str:
        """Deterministic stand-in for the real simulated-user LLM call.

        Uses the gold `key_entity` ambiguity_injector.py attached to this
        example at dataset-construction time - see AmbiguationResult in
        rl_retrieval_clarification_agents/envs/ambiguity_injector.py.
        """
        key_entity = extra_field.get("key_entity")
        if key_entity:
            return f"I meant {key_entity}."
        return "Sorry, I don't have more detail to add."

    def conduct_action(self, trajectory_id, action, extra_field):
        clarifying_question, is_valid = self.parse_action(action)
        env = self.load_env(trajectory_id)

        if not is_valid:
            observation = ""
            execution_result = ""
            done = False
            valid = False
        else:
            try:
                reply = self._generate_reply(clarifying_question, extra_field)
                observation = f"\n\n<information>{reply}</information>\n\n"
                execution_result = reply
                done = False  # clarify doesn't end the trajectory, same as search
                valid = True
            except Exception as e:
                logger.error(f"Clarify error for trajectory {trajectory_id}: {e}")
                execution_result = f"Clarify error: {str(e)}"
                observation = "\n\n<information>Clarification temporarily unavailable</information>\n\n"
                done = False
                valid = False

        self.update_env(trajectory_id, env, clarifying_question, is_valid, extra_field, execution_result)
        self.save_env(trajectory_id, env)

        return observation, done, valid
