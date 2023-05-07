"""A Tracer implementation that records to LangChain endpoint."""
from __future__ import annotations
import base64
import threading
from typing import Any, Dict, Optional,List, Union, Tuple
import datetime
import os
import re
from .data_model import (ActivityBase, Log, Session)
import logging
from .data_model import Session, ActivityBase,  ChainSequence, Log, Answer, Action, Question, RetrievedDocuments, DocumentSnippet
from .client import Client
from uuid import uuid4
import types
from .utils import wrap_a_method, classproperty
from .caching import PromptWatchCacheManager
from abc import ABCMeta


class ContextTrackerSingleton(ABCMeta,type):
    """
    Singleton metaclass for ensuring only one instance of a class per thread
    """

    _thread_local = threading.local()

    def __call__(cls, *args, **kwargs):
        """Call method for the singleton metaclass."""
        if not hasattr(ContextTrackerSingleton._thread_local, "_instance"):
            ContextTrackerSingleton._thread_local._instance = super(ContextTrackerSingleton, cls).__call__(*args, **kwargs)
        else:
            ContextTrackerSingleton._thread_local._instance._cached_init_(*args, **kwargs)
        return ContextTrackerSingleton._thread_local._instance 
        
    def get_current():
        if hasattr(ContextTrackerSingleton._thread_local, "_instance"):
            return ContextTrackerSingleton._thread_local._instance
        else:
            return None


    def remove_active_instance(cls):
        del ContextTrackerSingleton._thread_local._instance 

class PromptWatch(metaclass=ContextTrackerSingleton):
    
    prompt_template_register_cache={}

    def __init__(self, session_id: Optional[str] = None, tracking_project: Optional[str] = None, tracking_tenant: Optional[str] = None, api_key: Optional[str] = None):
        """
        PromptWatch context to track all the activities inside your LLM chain

        ## Parameters:

        session_id: Optional[str] - ID of previously used session. It can be assigned using uuid4 or uuid5, or you can save the generated session ID (PromptWatch.get_current_session().id) into the app context to retrieve it later.
        tracking_project: Optional[str] - Optional label for the session to pair it with a certain app or project. Use any arbitrary string (no whitespace and less than 256 chars).
        tracking_tenant: Optional[str] - Optional label for the session to pair it with a tenant (customer of yours). Use any arbitrary string (no whitespace and less than 256 chars). It can also be used to track costs attached to the customer.
        api_key: Optional[str] - API key for accessing the PromptWatch API.

        
        ### Example of use:
        ```
        with PromptWatch(tracking_tenant="myCustomerTenantId", tracking_project="nameOfThisTrackingProject", api_key="<your api key>"):
           agent.run(question) 
        ```
        """
        self.logger = logging.getLogger("PromptWatch")
        self.session_id = session_id
        self.tracking_project=tracking_project
        self.tracking_tenant=tracking_tenant

        if not api_key:
            api_key=os.environ.get("PROMPTWATCH_API_KEY")
        if not api_key:
            raise Exception("Unable to find PromptWatch API key. Either set api key as a parameter to PromptWatch(api_key='<your api key>') or set it up as an env. variable PROMPTWATCH_API_KEY. You can generate your API key here: https://app.promptwatch.io/get-api-key")
        else:
            tenant_id  = self._decode_tenant_from_api_key(api_key)
            if tenant_id.startswith("temp_"):
                if session_id:
                    self.logger.warn("Setting up session_id with a temporary PromptWatch API Key is not allowed. You session_id will be ignored")
                self.session_id = tenant_id[5:]
                GREEN = '\033[32m'
                RESET = '\033[0m'
                print(f"{GREEN}You are using a temporary API key. Do not use in production.")
                print(f"Visit the the detail of this session at https://www.promptwatch.io/sessions?temp-api-key={api_key} {RESET}")
                


        self.client = Client(api_key=api_key)
        
        
        self.chain_hierarchy:List[ChainSequence]=[]
        self.pending_session_save=True
        self.pending_stack:List[ActivityBase]=[]
        
        self.current_session=None
        self.context={}
        self._cache_manager = PromptWatchCacheManager(self)
        self.tracing_handlers={}

    @property
    def caching(self):
        return self._cache_manager

    @property
    def langchain(self):
        if not hasattr(self,"_langchain"):
            from .langchain.langchain_support import LangChainSupport
            self._langchain = LangChainSupport(self)
            
        
        return self._langchain
    


    def __enter__(self):
        

        if not self.tracing_handlers:
            # lets enable tracing by default
            self.langchain.init_tracing()
            #raise Exception("PromptWatch: LangChain callback handler is not set. Please call langchain_tracing() before entering the context.")

        if not self.current_session:
            self.start_session()
        
        

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.finish_session()
        except Exception as ex:
            self.logger.warn(f"Failed to persist the session: {ex}")

       



    @classmethod
    def get_current_session(cls) -> Session:
        """_summary_

        Raises:
            Exception: If called outside PromptWatch context

        Returns:
            Session: _description_
        """
        instance = cls.get_active_instance()
        if not instance:
            raise Exception("No session is active. Run this method only inside with PromptWatch context block")
        return instance.current_session
    
    @classmethod
    def get_active_instance(cls) -> Optional[PromptWatch]:
        """
        Returns active instance of PromptWatch if run inside PromptWatch context
        """
        return ContextTrackerSingleton.get_current()
    


    @classmethod
    def log(cls, text: str, error_msg:Optional[str]=None, metadata: Optional[Dict[str, Any]] = None):
        """_summary_

        Log arbitrary text message
        """

        cls.log_activity(Log(text=text, 
                                error=error_msg,
                                metadata=metadata, 
                                ))

    @classmethod
    def log_activity(cls, activity: ActivityBase):
        """
        Log any activity

        ## Example:
        ```
        from promptwatch import  PromptWatch, Question

        with PromptWatch():
            PromptWatch.log_activity(Question(text="Who am I?"))
        ```
        """
        instance = PromptWatch.get_active_instance()
        if not instance:
            raise Exception(
                "Invalid operation: you must enter an session before logging")
        instance._add_activity(activity)


    @property
    def current_activity(self) -> ActivityBase:
        if self.chain_hierarchy:
            return self.chain_hierarchy[-1]
        


    def register_prompt_template(self,template_name:str,prompt_template, version:Optional[str]=None):
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
        
        from .langchain.langchain_support import register_prompt_template
        register_prompt_template(template_name,prompt_template, version=version)
        


    def add_context(self, key:str, value:Any):
        self.context[key]=(self.current_activity ,value)
    
    def get_context(self, key:str) -> Any:
        if key in self.context:
            (_, value) =  self.context[key]
            return value
    
    def _end_context_scope(self):
        keys_to_delete = []
        for key, val in self.context.items():
            activity, _ = val
            if activity == self.current_activity:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del self.context[key]

    def _remove_context(self, key:str):
        """ Remove context at the end of current chain"""
        if key in self.context:
            del self.context[key]


        

        

    def _open_activity(self, activity:ActivityBase):
            
        if not self.current_session.session_name and isinstance(activity,ChainSequence) and activity.inputs:
            # if current session doesn't have a name, use first non empty input of first chain
            self.current_session.session_name = next((v for k,v in activity.inputs.items() if v and isinstance(v,str)), None)

        if not self.current_session.start_time:
            self.start_session()
        self.current_session.steps_count+=1
        activity.order=self.current_session.steps_count

        if self.current_activity:
            if isinstance(self.current_activity,ChainSequence) or isinstance(self.current_activity,Action):
                activity.parent_activity_id=self.current_activity.id
            else:
                activity.parent_activity_id=self.current_activity.parent_activity_id
    
        self.pending_stack.append(activity)
        self.chain_hierarchy.append(activity)
        return
        
    

    def _add_activity(self, activity:ActivityBase, as_root:bool=False):

        if not self.current_session.session_name and isinstance(activity,Question) and activity.text:
            # if current session doesn't have a name, use first non empty input of first chain
            self.current_session.session_name = activity.text

        if not self.current_session.start_time:
            self.start_session()
        
        self.current_session.steps_count+=1
        activity.order=self.current_session.steps_count

        activity.end_time=datetime.datetime.now(tz=datetime.timezone.utc)
        if self.current_activity and not as_root:
            if isinstance(self.current_activity,ChainSequence):
                activity.parent_activity_id=self.current_activity.id
            else:
                activity.parent_activity_id=self.current_activity.parent_activity_id
        self.pending_stack.append(activity)
        
        
        
    
    def _close_current_activity(self):
        self._end_context_scope()
        self.current_activity.end_time=datetime.datetime.now(tz=datetime.timezone.utc)
        if self.chain_hierarchy and self.current_activity==self.current_activity:
            closing_chain = self.chain_hierarchy.pop()
            closing_chain.end_time = datetime.datetime.now(tz=datetime.timezone.utc)
            if closing_chain not in self.pending_stack:
                self.pending_stack.append(closing_chain)
        
        self._flush_stack()

    def _flush_stack(self):
        if  self.pending_session_save and self.current_session.steps_count:
            try:
                self.client.start_session(self.current_session)
                self.pending_session_save=False
            except Exception as ex:
                self.logger.warn(f"Failed to persist the session: {ex}")
            
        if self.pending_stack and  not self.pending_session_save:
            try:
                self.client.save_activities(self.current_session.id, self.pending_stack)
                for activity in self.pending_stack:
                    if activity.end_time:
                        self.pending_stack.remove(activity)
            except Exception as ex:
                self.logger.warn(f"Failed to persist activities to PromptWatch: {ex}")


    def start_session(self):
        self.current_session=Session(start_time=datetime.datetime.now(tz=datetime.timezone.utc), tracking_project=self.tracking_project, tracking_tenant=self.tracking_tenant)
        self.logger.info(f"Starting PromptWatch session: {self.current_session.id}")
        self.current_session.start_time=self.current_session.start_time or datetime.datetime.now(tz=datetime.timezone.utc)
        self.pending_session_save=True
        
        
  

    def finish_session(self):
        
        self.current_session.end_time=datetime.datetime.now(tz=datetime.timezone.utc)
        self._flush_stack()
        if self.current_session.steps_count:
            self.client.finish_session(self.current_session)
        ContextTrackerSingleton.remove_active_instance()

    def _on_error(self, error, kwargs):
        self.current_activity.error=str(error)
        if kwargs:
            if not self.current_activity.metadata:
                self.current_activity.metadata={}
            self.current_activity.metadata["error_kwargs"]=kwargs
        self.current_session.is_error=True
        self._close_current_activity()

    def _decode_tenant_from_api_key(self, api_key)->str:
        # Decrypt the encrypted API key value
        try:
            decoded_api_key =base64.b64decode( api_key).decode("utf-8")
            # Convert the decrypted bytes to string
            return decoded_api_key.split(":")[0]
        except:
            raise Exception("This doesn't seems to be a valid PromptWatch API Key")