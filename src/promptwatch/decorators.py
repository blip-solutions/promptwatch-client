from promptwatch import PromptWatch
from promptwatch import ChainSequence
from .utils import find_the_caller_in_the_stack
from typing import Callable
from langchain.chains import LLMChain


FORMATTED_PROMPT_CONTEXT_KEY = "formatted_prompt"
TEMPLATE_NAME_CONTEXT_KEY = "template_name"

def format_prompt_decorator(template_name:str):
    def decorator_function(func):
        def wrapper(*args, **kwargs):
            

            #skipping the first argument because it is self
            result = func(*args, **kwargs)
            pw = PromptWatch.get_active_instance()
            if pw is not None:
                _self=args[0]
                pw.add_context(FORMATTED_PROMPT_CONTEXT_KEY, result)
                pw.add_context(TEMPLATE_NAME_CONTEXT_KEY, template_name)
                if pw.current_activity and hasattr(pw, "langchain_callback_handler")and isinstance(pw.current_activity,ChainSequence) and pw.current_activity.sequence_type=="LLMChain":
                    # if current_llm_chain is None, we want to fill it in. Probably running in async mode, so it was not accessible from the callback handler
                    if pw.langchain.langchain_callback_handler.current_llm_chain is None:
                        llm_chain = find_the_caller_in_the_stack(type=LLMChain)
                        if llm_chain:
                            pw.langchain.langchain_callback_handler.current_llm_chain =  llm_chain
                        

                
            return result
        return wrapper
    return decorator_function


def cached_call(cache_handler:Callable):
    def decorator(func):
        def wrapper(*args, **kwargs):
            cached_response = cache_handler(*args, **kwargs)
            if cached_response is not None:
                return cached_response
            else:        
                return func(*args, **kwargs)
        return wrapper
    return decorator