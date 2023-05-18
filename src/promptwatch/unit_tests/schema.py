from typing import *
from promptwatch.data_model import NamedPromptTemplateDescription,ActivityBase, LlmPrompt

from pydantic import BaseModel, validator, Field
from uuid import uuid4

from datetime import datetime,timedelta







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
        passed:bool
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
    
class PromptUnitTestConditions(BaseModel):
     for_template_name:Optional[str]
     for_tracking_project:Optional[str]
     for_test_cases_in_file:Optional[str]
     
PromptUnitTestRun.update_forward_refs()

class TestCaseResultsList(BaseModel):
    __root__: List[TestCaseResult]  

