"""A Tracer implementation that records to LangChain endpoint."""
from __future__ import annotations

from abc import ABC
from typing import Any, Dict, Optional, Union
import datetime

from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.chat import ChatPromptValue, BaseChatPromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, BaseMessagePromptTemplate
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import HumanMessage, ChatMessage as LangChainChatMessage, AIMessage, SystemMessage, BaseMessage
from langchain.callbacks import BaseCallbackHandler, get_callback_manager

from langchain.schema import AgentAction, AgentFinish, LLMResult,  Document

from .data_model import NamedPromptTemplateDescription,PromptTemplateDescription, LlmPrompt, ParallelPrompt, ChainSequence, ChatMessage, Answer, Action, Question, RetrievedDocuments, DocumentSnippet, ChatMessagePromptTemplate
from .utils import _find_the_caller_in_the_stack, _is_primitive_type

from .decorators import FORMATTED_PROMPT_CONTEXT_KEY, TEMPLATE_NAME_CONTEXT_KEY

from typing import List, Dict
from . import PromptWatch


class LangChainCallbackHandler(BaseCallbackHandler, ABC):
    """An implementation of the PromptWatch handler that tracks langchain tracing events"""


    def __init__(self, 
                 prompt_watch:PromptWatch

            ) -> None:
        self.current_llm_chain:Optional[LLMChain]=None
        self.tracing_handlers={}
        #we keep these in order to reverse
        self.monkey_patched_functions=[]
        self.prompt_watch=prompt_watch
        super().__init__()
        
        
        
             
    


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

        # this should ensure that all the additional data is available in the context
        if self.current_llm_chain:
            if self.current_llm_chain.llm and self.current_llm_chain.llm.dict:
                llm_info = {k:v for k,v in self.current_llm_chain.llm.dict().items() if _is_primitive_type(v)}
                llm_info["stop"] = self.prompt_watch.current_activity.inputs.get("stop")
            # lets try to retrieve registered named template first... it's faster
            template_name = self.prompt_watch.get_context(TEMPLATE_NAME_CONTEXT_KEY)
            if template_name:
                prompt_template = self.prompt_watch.prompt_template_register_cache.get(template_name)

            # this is useful for chat messages... since langchain converts them into text, but we want to keep the original format
            # for completion models is should be the same as *prompts* parameter
            # TODO: convert this into ChatMessages list and set as prompt instead prompts[0]
            formatted_prompt = self.prompt_watch.get_context(FORMATTED_PROMPT_CONTEXT_KEY)
            if isinstance(formatted_prompt,ChatPromptValue):
                #throwing away the original prompt from langchain tracing and replacing it with the original formatted messages
                prompts = [self.convert_chat_messages(formatted_prompt.messages)]

            if not prompt_template:
                # lets create anonymous prompt template description
                prompt_template = self.create_prompt_template_description(self.current_llm_chain.prompt)

            
            prompt_input_values = self.prompt_watch.current_activity.inputs
            if prompt_template and prompt_template.prompt_input_params and prompt_input_values:
                prompt_input_values = {k:v for k,v in prompt_input_values.items() if k in prompt_template.prompt_input_params and isinstance(v,str)}
            else:
                prompt_input_values={k:v for k,v in prompt_input_values.items() if isinstance(v,str)}
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
        pass

    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        if len(response.generations)>1:
            thoughts =  self.prompt_watch.current_activity.thoughts
        else:
            thoughts=[self.prompt_watch.current_activity]
        
        if hasattr(self,"current_llm_chain"):
            self.current_llm_chain = None
        if not self.prompt_watch.current_activity.metadata:
            self.prompt_watch.current_activity.metadata={}

        for thought, generated_responses in zip(thoughts, response.generations):
            thought.generated = "\n---\n".join([resp.text for resp in generated_responses])
            thought.metadata["generation_info"] = [resp.generation_info for resp in generated_responses]

        if response.llm_output is not None:
            self.prompt_watch.current_activity.metadata["llm_output"]=response.llm_output
            if "token_usage" in response.llm_output:
                token_usage = response.llm_output["token_usage"]
                if "total_tokens" in token_usage:
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
        
        if serialized.get("name").startswith("LLM") :
            self.current_llm_chain = _find_the_caller_in_the_stack(serialized["name"])

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
            #self.prompt_watch.client.start_session(self.prompt_watch.current_session)

        current_chain=ChainSequence(
                inputs=inputs,
                metadata={},
                sequence_type=serialized.get("name") or "others"
            )
                                     
        if kwargs:
            current_chain.metadata["input_kwargs"]=kwargs
        self.prompt_watch._open_activity(current_chain)
        

    # def _trace_function_call(self, handler_key:str, function_name:str, args, kwargs, result):
    #     handler = self.tracing_handlers.get(handler_key)
    #     if handler:
    #         handler(function_name, args, kwargs, result)
        

        
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
        
        if self.current_llm_chain:
            self.current_llm_chain =None
            
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
        pass

    
    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        
        pass
        



    
    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""
        answer_text = finish[0].get("output")
        answer_activity = Answer(text=answer_text)
        self.prompt_watch._add_activity(answer_activity, as_root=True)
        if finish.return_values:
            answer_activity.metadata["outputs"]:finish.return_values
        
        
    
    def convert_chat_messages(self, msg:Union[BaseMessage, List[BaseMessage]]):
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
            return [self.convert_chat_messages(msg) for msg in msg]
        

    def create_prompt_template_description(self, langchain_prompt_template:BasePromptTemplate, template_name:str = None, template_version:str=None)->Union[PromptTemplateDescription,NamedPromptTemplateDescription, None ]:
        
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
                    msg_templates.append(self.convert_chat_messages(msg))
                
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


