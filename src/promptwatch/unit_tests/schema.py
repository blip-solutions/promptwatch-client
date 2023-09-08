from abc import ABC, abstractmethod
from typing import *
from promptwatch.data_model import NamedPromptTemplateDescription,ActivityBase, LlmPrompt

from uuid import uuid4

from datetime import datetime,timedelta

import pydantic

if pydantic.__version__ <"2.0.0":
    from pydantic import BaseModel, validator, Field
else:
    from pydantic.v1 import BaseModel, validator, Field







class LabeledOutput(BaseModel): 
    """ LabeledGenerationOutput represents a single example of a generation output. It is a View Model for data in the database, for UnitTest specifically"""
    label:int
    """ Label represents whether it is a positive or a negative example"""
    value:Union[str,dict]
    """ LLM output - generated text"""

    @validator('label')
    def validate_label(cls, value):
        if value not in [0,1]:
            raise ValueError("Label must be either 0 or 1")
        return value
        
class TestCase(BaseModel):
     id:str=None
     for_template_name:str=None
     for_tracking_project:str=None
     inputs:dict
     outputs:List[LabeledOutput]


    
class TestCaseResult(BaseModel):
    evaluation_method:Optional[str]
    evaluation_metadata:dict=None
    test_case:TestCase
    activity_id:Optional[str]
    llm_prompt:Optional[LlmPrompt]
    conversation_session_id:Optional[str] = None
    response_time_ms:Optional[int]
    passed:bool
    order:Optional[int]
    reasoning:Optional[str]
    score:Optional[float]
    generated_result:Union[dict,str,None]=None
    error_description:Optional[str]=None





class PromptUnitTestRunResultsSummary(BaseModel):
    overall_score:Optional[float]=None
    total_processed:int=0
    passed:int=0
    failed:int=0
    
class PromptUnitTestRun(BaseModel):
    run_id:str=Field(default_factory=lambda: str(uuid4()))
    session_id:str
    """ PromptWatch session ID"""
    evaluation_method:str
    test_name:str
    start_time:datetime=None 
    end_time:datetime=None
    conditions:Optional["PromptUnitTestConditions"]
    results:PromptUnitTestRunResultsSummary=Field(default_factory= PromptUnitTestRunResultsSummary)
    

    @property
    def duration(self)->timedelta:
        return self.end_time-self.start_time
    

class ConversationTopic(BaseModel):
    name:str=Field(..., description="Name of the topic")
    conversation_goal:str=Field(..., description="The goal the persona is trying to achieve during the conversation")
    opening_line:str

class SimulationPersona(BaseModel):
    name:str=Field(..., description="Name and title of the persona")
    role:str = Field(..., description="Role of the persona regarding the use case he/she should be involved in")
    description:str=Field(..., description="Description of the persona (around 50 words)")
    conversation_topics:List[ConversationTopic]


class ConversationSimulationDefinition(BaseModel):
    chatbot_description:str
    personas:List[SimulationPersona]

class PromptUnitTestConditions(BaseModel):
     for_template_name:Optional[str]
     for_tracking_project:Optional[str]
     for_test_cases_in_file:Optional[str]
     for_simulated_conversations:Optional[str]


     
PromptUnitTestRun.update_forward_refs()

class TestCaseResultsList(BaseModel):
    __root__: List[TestCaseResult]  

