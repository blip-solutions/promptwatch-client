import re

from .data_model import ChainSequence
from .promptwatch_context import PromptWatch


from .utils import find_the_caller_in_the_stack, wrap_a_method
from typing import Callable, Optional


FORMATTED_PROMPT_CONTEXT_KEY = "formatted_prompt"
TEMPLATE_NAME_CONTEXT_KEY = "template_name"
LLM_CHAIN_CONTEXT_KEY="current_llm_chain"

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
                if isinstance(pw.current_activity,ChainSequence):
                    llm_chain = find_the_caller_in_the_stack(type=LLMChain)
                    if llm_chain:
                        pw.add_context(LLM_CHAIN_CONTEXT_KEY, llm_chain)  
                        

                
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


def register_prompt_template(
    template_name: str, prompt_template, version: Optional[str] = "1.0"
):
    """
    Register prompt template into context for more detailed tracking, versioning and evaluation


    Args:
        template_name (str): arbitrary unique template name (should not contain white spaces, max 126 chars)
        prompt_template (Union[BasePromptTemplate,BaseChatPromptTemplate]): Any langchain prompt template
        version (Optional[str], optional): Optional - (major) (SemVer) version of the template. Minor changes are tracked incrementally automatically. Useful if you are going to make a major change in the template, and you with to dive the version a distinct version number.

    ## Examples:

    ### Registering regular prompt (completion)

    ```
    from promptwatch import PromptWatch, register_prompt_template
    from langchain import OpenAI, LLMChain, PromptTemplate

    prompt_template = PromptTemplate.from_template("Finish this sentence {input}")
    my_chain = LLMChain(llm=OpenAI(), prompt=prompt_template)
    register_prompt_template("simple_completion_template",prompt_template, version="1.0")

    with PromptWatch():
        my_chain("The quick brown fox jumped over")
    ```

    ### Registering chat messages template (completion)
    ...

    """
    if not template_name:
        raise Exception("template_name can't be empty")
    if re.search(r"\s", template_name):
        raise Exception("template_name can't contain white spaces")
    if len(template_name) > 126:
        raise Exception("template_name must be less than 126 characters")

    try:
        from .langchain.langchain_support import create_prompt_template_description

        # TODO: hardcoded langchain_callback_handler
        converted_template = create_prompt_template_description(
            prompt_template, template_name=template_name, template_version=version
        )
    except ImportError:
        converted_template = None

    decorator_func = format_prompt_decorator(template_name)

    wrap_a_method(prompt_template, "format_prompt", decorator_func)
    # we need to mark the prompt template somehow... keeping just the reference doesn't work since pydantic is creating copy of it when passed to the constructor
    # that is why we add special field __template_name__ into it... also skipping setattr() since that might be blocked by pydantic as well (depending on the configuration)
    prompt_template.__dict__["__template_name__"] = template_name
    prompt_template.__dict__["__template_version__"] = version

    PromptWatch.prompt_template_register_cache[template_name] = converted_template
    return prompt_template
