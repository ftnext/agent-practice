# https://github.com/google/adk-python/tree/v1.16.0/contributing/samples/plugin_basic
from google.adk.plugins import BasePlugin


class CountInvocationPlugin(BasePlugin):
    def __init__(self):
        super().__init__(name="count_invocation")
        self.agent_count = 0
        self.tool_count = 0
        self.llm_request_count = 0

    async def before_agent_callback(self, *, agent, callback_context):
        self.agent_count += 1
        print(f"[Plugin] Agent run count: {self.agent_count}")

    async def before_model_callback(self, *, callback_context, llm_request):
        self.llm_request_count += 1
        print(f"[Plugin] LLM request count: {self.llm_request_count}")

    async def before_tool_callback(self, *, tool, tool_args, tool_context):
        self.tool_count += 1
        print(f"[Plugin] Tool call count: {self.tool_count}")
