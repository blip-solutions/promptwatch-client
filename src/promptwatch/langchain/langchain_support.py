"""A Tracer implementation that records to LangChain endpoint."""
from __future__ import annotations
import re
from abc import ABC
from typing import Any, Dict, Optional, Union
import datetime
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, BaseMessagePromptTemplate
from langchain.llms.base import LLM
from langchain.chat_models.base import BaseChatModel
from langchain.chains import LLMChain
from langchain.embeddings.base import Embeddings
from langchain.schema import HumanMessage, ChatMessage as LangChainChatMessage, AIMessage, SystemMessage, BaseMessage
from langchain.callbacks.base import BaseCallbackHandler

from langchain.schema import AgentAction, AgentFinish, LLMResult,  Document
from ..client import Client
from ..data_model import NamedPromptTemplateDescription,PromptTemplateDescription, LlmPrompt, ParallelPrompt, ChainSequence, ChatMessage, Answer, Action, Question, RetrievedDocuments, DocumentSnippet, ChatMessagePromptTemplate
from ..utils import find_the_caller_in_the_stack, is_primitive_type, wrap_a_method
from .caching import CachedLLM, CachedChatLLM
from ..decorators import FORMATTED_PROMPT_CONTEXT_KEY, TEMPLATE_NAME_CONTEXT_KEY, LLM_CHAIN_CONTEXT_KEY

from typing import List, Dict
from ..promptwatch_context import PromptWatch, ContextTrackerSingleton


class LangChainSupport:

    def __init__(self, promptwatch_context:PromptWatch) -> None:
        self.promptwatch_context=promptwatch_context
        self.langchain_callback_handler=None
        
    def init_tracing(self, langchain_callback_manager=None)->PromptWatch:
        """ Enable langchain tracing

        Args:
            langchain_callback_manager (langchain.callbacks.base.BaseCallbackManager, optional):  If using custom callback manager, pass it here. Otherwise, default callback manager will be used. Defaults to None.

        """
        self.langchain_callback_handler = LangChainCallbackHandler(self.promptwatch_context)

        try:
            # this will 
            #for langchain >0.0.153
            import langchain.callbacks.manager as langchain_callback_manager_module
            def promptwatch_callback_configure_decorator(func):
                def configure_with_promptwatch(*args, **kwargs):
                    callback_manager = func(*args, **kwargs)
                    if callback_manager and not any(isinstance(handler, LangChainCallbackHandler) for handler in callback_manager.handlers):
                        pw = PromptWatch.get_active_instance()
                        if pw:
                            handler = pw.langchain.get_langchain_callback_handler()
                            callback_manager.add_handler(handler)
                    return callback_manager
                configure_with_promptwatch.__original_func = func
                return configure_with_promptwatch
            if not hasattr(langchain_callback_manager_module._configure,"__original_func"):
                decorated_configure = promptwatch_callback_configure_decorator(langchain_callback_manager_module._configure)    
                setattr(langchain_callback_manager_module, "_configure",  decorated_configure)
        except ImportError:
            

            try:
                from langchain.callbacks import get_callback_manager
                langchain_callback_manager = langchain_callback_manager or get_callback_manager()
                langchain_callback_manager.add_handler(self.langchain_callback_handler)
            except ImportError:
                print("\033[31m"+"Unable to auto-initialize PromptWatch tracing. \nWorkaround: Use PromptWatch.langchain.get_langchain_callback_handler and set it into callback argument to your chains"+"\033[0m")

        


        return self

    def get_langchain_callback_handler(self):
        if self.langchain_callback_handler  :
            
            self.langchain_callback_handler = LangChainCallbackHandler(self.promptwatch_context)
            if "langchain_callback_handler"  not in self.promptwatch_context.tracing_handlers:
                self.promptwatch_context.tracing_handlers["langchain_callback_handler"]=self.langchain_callback_handler

        return self.langchain_callback_handler
    

    def get_cached_llm(self, llm:LLM, embeddings:Embeddings=None, token_limit:int=None, similarity_limit:float=0.97):
        if isinstance(llm, BaseChatModel):
            return CachedChatLLM(llm, embeddings, token_limit, similarity_limit)
        else:
            return CachedLLM(
                inner_llm=llm,
                cache_embeddings=embeddings, token_limit=token_limit, similarity_limit=similarity_limit)
        
    # def enable_global_cache(self, embeddings:Embeddings, token_limit:int, similarity_limit:float=0.97)->PromptWatchLlmCache:
    #     """Enable global cache for all LLMs"""
    #     prompt_cache = PromptWatchLlmCache(None, embeddings, token_limit, similarity_limit)
    #     langchain.llm_cache=prompt_cache

   

class LangChainCallbackHandler(BaseCallbackHandler, ABC):
    """An implementation of the PromptWatch handler that tracks langchain tracing events"""


    def __init__(self, 
                 prompt_watch:PromptWatch

            ) -> None:
        self.current_llm_chain:Optional[LLMChain]=None
        self.tracing_handlers={}
        self.prompt_watch_session_id=prompt_watch.session_id
        #we keep these in order to reverse
        self.monkey_patched_functions=[]
        
        super().__init__()

        
    @property
    def prompt_watch(self) -> PromptWatch:
        """Whether to call verbose callbacks even if verbose is False."""
        
        prompt_watch_context = ContextTrackerSingleton.get_current(self.prompt_watch_session_id)
        if not prompt_watch_context:
            raise Exception("PromptWatch context could not be resolved")
        return prompt_watch_context


    @property
    def always_verbose(self) -> bool:
        """Whether to call verbose callbacks even if verbose is False."""
        return True
    
    

    def reverse_monkey_patching(self):
        for func in self.monkey_patched_functions:
            #TODO: .. we should restore the state of the things as were before...
            pass
            

    

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""

        prompt_template = None
        prompt_input_values=None
        llm_info=None
        info_message=None
        formatted_prompt=None

        current_llm_chain:LLMChain= self.prompt_watch.get_context(LLM_CHAIN_CONTEXT_KEY)

        template_name = self.prompt_watch.get_context(TEMPLATE_NAME_CONTEXT_KEY)
        if template_name:
            prompt_template = PromptWatch.prompt_template_register_cache.get(template_name)
            
        # this should ensure that all the additional data is available in the context
        if current_llm_chain:
            if current_llm_chain.llm and current_llm_chain.llm.dict:
                llm = current_llm_chain.llm
                if hasattr(current_llm_chain.llm,"inner_llm"): # cachedLLM
                    llm = llm.inner_llm
                llm_info = {k:v for k,v in llm.dict().items() if is_primitive_type(v)}
                llm_info["stop"] = self.prompt_watch.current_activity.inputs.get("stop")
            # lets try to retrieve registered named template first... it's faster
            

            if not prompt_template:
                # lets create anonymous prompt template description
                prompt_template = create_prompt_template_description(current_llm_chain.prompt)

            prompt_input_values = self.prompt_watch.current_activity.inputs

            formatted_prompt = self.prompt_watch.get_context(FORMATTED_PROMPT_CONTEXT_KEY)
            if isinstance(current_llm_chain.prompt,ChatPromptTemplate):
                if not formatted_prompt:
                    # we need to reformat the prompt so we can get the original values, not the strings
                    formatted_prompt = current_llm_chain.prep_prompts([prompt_input_values])[0][0]
                if formatted_prompt:
                    #throwing away the original prompt from langchain tracing and replacing it with the original formatted messages
                    prompts = [convert_chat_messages(formatted_prompt.messages)]
            
            
            
            

            if prompt_template and prompt_template.prompt_input_params and prompt_input_values:
                prompt_input_values = {k:v for k,v in prompt_input_values.items() if k in prompt_template.prompt_input_params and (isinstance(v,str) )}
            else:
                prompt_input_values={k:v for k,v in prompt_input_values.items() if isinstance(v,str)}

            if  prompt_template and isinstance(prompt_template.prompt_template,list):
                non_text_input = {k:v for k,v in  self.prompt_watch.current_activity.inputs.items() if not isinstance(v,str)}
                for k, v in non_text_input.items():
                    if v and isinstance(v,list) and isinstance(v[0],ChatMessage):
                        prompt_input_values[k] = v
                    # this should be necessary anymore... keeping it just in case
                    if v and isinstance(v,list) and isinstance(v[0],BaseMessage):
                        prompt_input_values[k] = convert_chat_messages(v)
                        

        else:
            info_message="Could not retrieve all the additional information needed to for reproducible prompt execution. Consider registering the prompt template."

        if len(prompts)==1:
            
            self.prompt_watch._open_activity(LlmPrompt(
                prompt= prompts[0], #why we have here more than one prompt?
                prompt_template=prompt_template,
                prompt_input_values=prompt_input_values,
                metadata={"llm":llm_info,**(serialized or {}),**(kwargs or {})} 
                ))
        elif len(prompts)>1:
            thoughts = []
            for prompt in prompts:
                thoughts.append( LlmPrompt(
                            prompt=prompt, #why we have here more than one prompt?
                            prompt_template=prompt_template,
                            prompt_input_values=prompt_input_values,
                            info_message=info_message,
                        ))
            self.prompt_watch._open_activity(ParallelPrompt(
                    thoughts=thoughts,
                    metadata={**serialized,**kwargs} if serialized and kwargs else (serialized or kwargs),
                    order=self.prompt_watch.current_session.steps_count+1, 
                    session_id=self.prompt_watch.current_session.id,
                    info_message=info_message,
                )
            )



    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""

    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        if len(response.generations)>1:
            thoughts =  self.prompt_watch.current_activity.thoughts
        else:
            thoughts=[self.prompt_watch.current_activity]
        
        
        if not self.prompt_watch.current_activity.metadata:
            self.prompt_watch.current_activity.metadata={}

        for thought, generated_responses in zip(thoughts, response.generations):
            thought.generated = "\n---\n".join([resp.text for resp in generated_responses])
            thought.metadata["generation_info"] = [resp.generation_info for resp in generated_responses] if len(generated_responses)>1 else generated_responses[0].generation_info

        if response.llm_output is not None:
            self.prompt_watch.current_activity.metadata["llm_output"]=response.llm_output
            token_usage= response.llm_output.get("token_usage")
            if token_usage and isinstance(token_usage,dict):
                total_tokens = token_usage.get("total_tokens")
                
                if total_tokens:
                    self.prompt_watch.current_activity.metadata["total_tokens"] = self.prompt_watch.current_activity.metadata.get("total_tokens",0)+ token_usage["total_tokens"]
        self.prompt_watch._close_current_activity()

        

    
    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when LLM errors."""
        self.prompt_watch._on_error(error, kwargs)

    
    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        """Run when chain starts running."""
        
        if "LLM" in serialized.get("name") :
            current_llm_chain = find_the_caller_in_the_stack(serialized["name"])
            self.prompt_watch.add_context(LLM_CHAIN_CONTEXT_KEY,current_llm_chain)

        self.try_get_retrieved_documents(inputs)
                  


        question = inputs.get("question") 
        if not question and "chat_history" in inputs:
            #try to get question from inputs
            question = next((v for k,v in inputs.items() if k!="chat_history"),None) if len(inputs)==2 else None

        if  question and not self.prompt_watch.chain_hierarchy:
            self.prompt_watch._add_activity(Question(text=question))
        if not self.prompt_watch.current_session.session_name:
            self.prompt_watch.current_session.session_name=question
            if not self.prompt_watch.current_session.start_time:
                self.prompt_watch.current_session.start_time=datetime.datetime.now(tz=datetime.timezone.utc)
            
        
        
        current_chain=ChainSequence(
                inputs=serialize_chain_inputs(inputs),
                metadata={},
                sequence_type=serialized.get("name") or "others"
            )
                                     
        if kwargs:
            current_chain.metadata["input_kwargs"]=kwargs
        self.prompt_watch._open_activity(current_chain)
        

        

        
    def try_get_retrieved_documents(self, inputs:dict):
        retrieved_documents = next((val for key,val in inputs.items() if isinstance(val,list) and val and isinstance(val[0], Document)),None)
        if retrieved_documents:
            docs=[]
            for doc in retrieved_documents:
                metadata = {key:val for key,val in doc.metadata.items() if key!="source"}  if doc.metadata else None
                source = doc.metadata.get("source") if doc.metadata else None
                docs.append(DocumentSnippet(
                    text=doc.page_content, 
                    source=source,
                    metadata=metadata if metadata else None # to not pass empty objects
                    ))
            self.prompt_watch._add_activity(RetrievedDocuments(documents=docs))
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
        
        
        self.prompt_watch._remove_context(LLM_CHAIN_CONTEXT_KEY)
            
        self.prompt_watch.current_activity.outputs=outputs
        if outputs.get("answer"):
            self.prompt_watch._add_activity(Answer(text=outputs["answer"]),as_root=True)
        if kwargs:
            self.prompt_watch.current_activity.metadata["output_kwargs"]=kwargs
        
        if "total_tokens" in self.prompt_watch.current_activity.metadata and len(self.prompt_watch.chain_hierarchy)>1:
            parent_activity = self.prompt_watch.chain_hierarchy[-2]
            if not parent_activity.metadata:
                parent_activity.metadata={"total_tokens":self.prompt_watch.current_activity.metadata["total_tokens"]}
            else:
                parent_activity.metadata["total_tokens"] = parent_activity.metadata.get("total_tokens",0)+ self.prompt_watch.current_activity.metadata["total_tokens"]
        
        self.prompt_watch._close_current_activity()

    
    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when chain errors."""
        self.prompt_watch._on_error(error, kwargs)

    
    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""
        self.prompt_watch._open_activity(
                Action(tool_type=serialized.get("name") or "undefined", input=input_str, input_data=kwargs)
            )

    
    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when tool ends running."""
        self.prompt_watch.current_activity.output=output
        self.prompt_watch.current_activity.output_data=kwargs
        self.prompt_watch._close_current_activity()


        
    

    
    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when tool errors."""
        self.prompt_watch._on_error(error, kwargs)

    
    def on_text(self, text: str, **kwargs: Any) -> Any:
        """Run on arbitrary text."""

    
    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        
        



    
    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""
        answer_text = finish[0].get("output")
        answer_activity = Answer(text=answer_text)
        self.prompt_watch._add_activity(answer_activity, as_root=True)
        if finish.return_values:
            answer_activity.metadata["outputs"]:finish.return_values
        
        
    

        



def register_prompt_template(template_name:str,prompt_template, version:Optional[str]=None):
        """
        Register prompt template into context for more detailed tracking, versioning and evaluation


        Args:
            template_name (str): arbitrary unique template name (should not contain white spaces, max 126 chars)
            prompt_template (Union[BasePromptTemplate,BaseChatPromptTemplate]): Any langchain prompt template
            version (Optional[str], optional): Optional - (major) (SemVer) version of the template. Minor changes are tracked incrementally automatically. Useful if you are going to make a major change in the template, and you with to dive the version a distinct version number.

        ## Examples:
        
        ### Registering regular prompt (completion)

        ```
        from promptwatch import PromptWatch
        from langchain import OpenAI, LLMChain, PromptTemplate

        prompt_template = PromptTemplate.from_template("Finish this sentence {input}")
        my_chain = LLMChain(llm=OpenAI(), prompt=prompt_template)

        with PromptWatch() as pw:
            pw.register_prompt_template("simple_completion_template",prompt_template, version="1.0")
            my_chain("The quick brown fox jumped over")
        ```

        ### Registering chat messages template (completion)
        ...

        """
        if not template_name:
            raise Exception("template_name can't be empty")
        if re.search(r"\s", template_name):
            raise Exception("template_name can't contain white spaces")
        if len(template_name)>126:
            raise Exception("template_name must be less than 126 characters")
        
        #TODO: hardcoded langchain_callback_handler
        converted_template =  create_prompt_template_description(prompt_template, template_name=template_name, template_version=version)
        from ..decorators import format_prompt_decorator
        decorator_func = format_prompt_decorator(template_name)
        
        
        wrap_a_method(prompt_template, "format_prompt", decorator_func)
        # we need to mark the prompt template somehow... keeping just the reference doesn't work since pydantic is creating copy of it when passed to the constructor
        # that is why we add special field __template_name__ into it... also skipping setattr() since that might be blocked by pydantic as well (depending on the configuration)
        prompt_template.__dict__["__template_name__"]=template_name

        PromptWatch.prompt_template_register_cache[template_name] = converted_template
        return prompt_template



def convert_chat_messages( msg:Union[BaseMessage, List[BaseMessage]]):
        if isinstance(msg, BaseMessage):
                if isinstance(msg,HumanMessage):
                    role="user"
                elif isinstance(msg,LangChainChatMessage):
                    role = msg.role
                elif isinstance(msg,AIMessage):
                    role="assistant"
                elif isinstance(msg,SystemMessage):
                    role="system"
                    
                return (ChatMessage(role=role,text=msg.content))
        elif isinstance(msg, list):
            return [convert_chat_messages(msg) for msg in msg]
        else:
            raise ValueError("msg must be either BaseMessage or List[BaseMessage]")
        
def reconstruct_langchain_chat_messages( msg:Union[ChatMessage, List[ChatMessage]]):
        if isinstance(msg, ChatMessage):
                if msg.role=="user":
                    return HumanMessage( content=msg.text)
                    
                elif msg.role=="assistant":
                    return AIMessage( content=msg.text)
                    
                elif msg.role=="system":
                    return SystemMessage( content=msg.text)
                    
                else:    
                    return (LangChainChatMessage(role=msg.role,content=msg.text))
        elif isinstance(msg, list):
            return [reconstruct_langchain_chat_messages(msg) for msg in msg]
        else:
            raise ValueError("msg must be either ChatMessage or List[ChatMessage]")
        
def serialize_chain_inputs(inputs:dict):
    res = {}
    for k,v in inputs.items():
        if isinstance(v, BaseMessage):
            res[k]=convert_chat_messages(v)
        elif isinstance(v, dict):
            res[k]=serialize_chain_inputs(v)
        elif isinstance(v, list):
            if v:
                if isinstance(v[0], BaseMessage):
                    res[k]=convert_chat_messages(v)
                elif isinstance(v[0], dict):
                    res[k]=[serialize_chain_inputs(item) for item in v]
                else:
                    res[k]=v
        else:
            res[k]=v
    return res

def create_prompt_template_description( langchain_prompt_template:BasePromptTemplate, template_name:str = None, template_version:str=None)->Union[PromptTemplateDescription,NamedPromptTemplateDescription, None ]:
        
    format=None
    prompt_template=None
    
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
                msg_templates.append(convert_chat_messages(msg))
            
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


