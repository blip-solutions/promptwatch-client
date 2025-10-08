from typing import Any, Dict, List
from langchain_core.callbacks import BaseCallbackHandler, AsyncCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult



from ..promptwatch_context import PromptWatch

class PromptWatchLangChainCallbackHandler(BaseCallbackHandler):
    """Callback handler to watch the prompt and response."""

    def __init__(self, promptwatch):
        self.promptwatch = promptwatch

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Handle new token."""
        print(f"New token: {token}")

    def on_llm_end(self, response: str, **kwargs) -> None:
        """Handle end of LLM call."""
        print(f"Response: {response}")

class PromptWatchLangChainAsyncCallbackHandler(BaseCallbackHandler):
    """Callback handler to watch the prompt and response."""

    def __init__(self, promptwatch):
        self.promptwatch = promptwatch
        self.pending_actions= {}
    
    @property
    def always_verbose(self) -> bool:
        """Whether to call verbose callbacks even if verbose is False."""
        return True



    async def on_llm_start(self, serialized: Dict[str, Any], messages: list[list[BaseMessage]], *, run_id, parent_run_id = None, tags = None, metadata = None, **kwargs):
        from ..decorators import TEMPLATE_NAME_CONTEXT_KEY
        template_name = self.prompt_watch.get_context(TEMPLATE_NAME_CONTEXT_KEY)
        if template_name:
            prompt_template = PromptWatch.prompt_template_register_cache.get(template_name)


    async def on_llm_end(self, response: LLMResult, *, run_id, parent_run_id = None, **kwargs):
        print(f"Ending LLM with run_id: {run_id}, parent_run_id: {parent_run_id}")
        
    async def on_chat_model_start(self, serialized: Dict[str, Any], messages: list[list[BaseMessage]], *, run_id, parent_run_id = None, tags = None, metadata = None, **kwargs):
        print(f"Starting Chat Model with run_id: {run_id}, parent_run_id: {parent_run_id}")
    
    async def on_llm_error(self, error, *, run_id, parent_run_id = None, **kwargs):
        print(f"LLM Error with run_id: {run_id}, parent_run_id: {parent_run_id}")
        print(f"Error: {error}")
    


    async def on_tool_start(self, serialized, tool_input, *, run_id, parent_run_id = None, tags = None, metadata = None, **kwargs):
        print(f"Starting Tool with run_id: {run_id}, parent_run_id: {parent_run_id}")

    async def on_tool_end(self, response, *, run_id, parent_run_id = None, **kwargs):
        print(f"Ending Tool with run_id: {run_id}, parent_run_id: {parent_run_id}")

    async def on_tool_error(self, error, *, run_id, parent_run_id = None, **kwargs):
        print(f"Tool Error with run_id: {run_id}, parent_run_id: {parent_run_id}")
        

    async def on_agent_action(self, action, *, run_id, parent_run_id = None, **kwargs):
        print(f"Agent Action with run_id: {run_id}, parent_run_id: {parent_run_id}")
        

    async def on_agent_end(self, finish, *, run_id, parent_run_id = None, **kwargs):
        print(f"Agent Finish with run_id: {run_id}, parent_run_id: {parent_run_id}")
        

    async def on_agent_error(self, error, *, run_id, parent_run_id = None, **kwargs):
        print(f"Agent Error with run_id: {run_id}, parent_run_id: {parent_run_id}")
        print(f"Error: {error}")

    async def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id = None, tags = None, metadata = None, **kwargs):
        print(f"Starting Chain with run_id: {run_id}, parent_run_id: {parent_run_id}")
        


    async def on_chain_end(self, outputs, *, run_id, parent_run_id = None, **kwargs):
        print(f"Ending Chain with run_id: {run_id}, parent_run_id: {parent_run_id}")
        

    async def on_chain_error(self, error, *, run_id, parent_run_id = None, **kwargs):
        print(f"Chain Error with run_id: {run_id}, parent_run_id: {parent_run_id}")
        print(f"Error: {error}")