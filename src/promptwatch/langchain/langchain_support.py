import logging
from typing import List, Union
from langchain_core.callbacks import BaseCallbackHandler, BaseCallbackManager


from ..data_model import NamedPromptTemplateDescription, PromptTemplateDescription, ChatMessagePromptTemplate
from .callback_handlers import PromptWatchLangChainAsyncCallbackHandler
from ..promptwatch_context import PromptWatch

from langchain_core.prompts import (
    BasePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
    StringPromptTemplate
)
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ChatMessage,
)
from langchain_core.prompts.message import (
    BaseMessagePromptTemplate,
)
from langchain_core.prompts.message import BaseMessagePromptTemplate

logger = logging.getLogger("promptwatch.langchain_support")


class LangChainSupport:

    def __init__(self, promptwatch_context:PromptWatch) -> None:
        self.promptwatch_context=promptwatch_context
        self.langchain_callback_handler = None

    def init_tracing(self, langchain_callback_manager=None)->PromptWatch:
        """Enable langchain tracing

        Args:
            langchain_callback_manager (langchain.callbacks.base.BaseCallbackManager, optional):  If using custom callback manager, pass it here. Otherwise, default callback manager will be used. Defaults to None.

        """

        try:
            import langchain_core.callbacks.manager as langchain_callback_manager_module

            def promptwatch_callback_configure_decorator(func):
                def configure_with_promptwatch(*args, **kwargs):
                    callback_manager: BaseCallbackManager = func(*args, **kwargs)
                    if callback_manager and not any(
                        isinstance(handler, PromptWatchLangChainAsyncCallbackHandler)
                        for handler in callback_manager.handlers
                    ):
                        pw = PromptWatch.get_active_instance()
                        if pw:
                            handler = pw.langchain.get_langchain_callback_handler()
                            if handler:
                                callback_manager.add_handler(handler)
                    return callback_manager
                configure_with_promptwatch.__original_func = func
                return configure_with_promptwatch

            if hasattr(langchain_callback_manager_module,"_configure") and not hasattr(langchain_callback_manager_module._configure,"__original_func"):
                decorated_configure = promptwatch_callback_configure_decorator(langchain_callback_manager_module._configure)    
                setattr(langchain_callback_manager_module, "_configure",  decorated_configure)
            else:

                pass

        except ImportError:
            print(
                "\033[31m"
                + "Unable to auto-initialize PromptWatch tracing. \nWorkaround: Use promptwatch.langchain.PromptWatchLangChainAsyncCallbackHandler and set it into callback argument to your chains"
                + "\033[0m"
            )

        return self

    def get_langchain_callback_handler(self):
        if not self.langchain_callback_handler:
            self.langchain_callback_handler = PromptWatchLangChainAsyncCallbackHandler(
                self.promptwatch_context
            )
            if (
                "langchain_callback_handler"
                not in self.promptwatch_context.tracing_handlers
            ):
                self.promptwatch_context.tracing_handlers["langchain_callback_handler"]=self.langchain_callback_handler

        return self.langchain_callback_handler


def _convert_chat_messages(msg: Union[BaseMessage, List[BaseMessage]]):
    if isinstance(msg, BaseMessage):
        metadata = None
        if msg.additional_kwargs:
            metadata = msg.additional_kwargs
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, SystemMessage):
            role = "system"
        elif msg.type == "function":
            role = "function"
            metadata = metadata or {}
            metadata["name"] = msg.name
        return ChatMessage(role=role, text=msg.content, metadata=metadata)
    elif isinstance(msg, list):
        return [_convert_chat_messages(m) for m in msg]
    else:
        raise ValueError("msg must be either BaseMessage or List[BaseMessage]")


def create_prompt_template_description( langchain_prompt_template:BasePromptTemplate, template_name:str = None, template_version:str=None)->Union[PromptTemplateDescription,NamedPromptTemplateDescription, None ]:

    format=None
    prompt_template=None
    input_params=None
    if hasattr(langchain_prompt_template,"messages") and  langchain_prompt_template.messages:
        msg_templates = []
        input_params=set()
        for msg in langchain_prompt_template.messages:
            role=None
            prompt_template=None
            msg_input_params=None
            format=None
            if isinstance(msg,MessagesPlaceholder):
                msg_templates.append(msg.variable_name)
                continue
            elif isinstance(msg, BaseMessagePromptTemplate):

                if isinstance(msg,HumanMessagePromptTemplate):
                    role="user"
                elif isinstance(msg,AIMessagePromptTemplate):
                    role="assistant"
                elif isinstance(msg,SystemMessagePromptTemplate):
                    role="system"
                elif isinstance(msg, ChatMessagePromptTemplate):
                    role=msg.role
                if hasattr(msg,"prompt"):
                    if hasattr(msg.prompt,"template"):
                        prompt_template = msg.prompt.template
                    if hasattr(msg.prompt,"input_variables"):
                        msg_input_params = msg.prompt.input_variables
                        if msg_input_params:
                            for param in msg_input_params:
                                input_params.add(param)
                    if hasattr(msg.prompt,"template_format"):
                        format=msg.prompt.template_format

                msg_templates.append(ChatMessagePromptTemplate(role=role, prompt_input_params=msg_input_params, prompt_template=prompt_template, format=format))
            elif isinstance(msg,BaseMessage):
                msg_templates.append()

        prompt_template = msg_templates
        input_params=list(input_params)
        format="chat_messages"
    else:
        if hasattr(langchain_prompt_template,"input_variables") and  langchain_prompt_template.input_variables:
            input_params = langchain_prompt_template.input_variables

        prompt_template = langchain_prompt_template.template if hasattr(langchain_prompt_template,"template") else None

        if  hasattr(langchain_prompt_template,"template_format"):
            format=langchain_prompt_template.template_format
    if prompt_template:
        if template_name:
            return NamedPromptTemplateDescription(prompt_template=prompt_template, prompt_input_params=input_params, format=format, template_name=template_name ,template_version=template_version)
        else:
            return PromptTemplateDescription(prompt_template=prompt_template, prompt_input_params=input_params, format=format)
