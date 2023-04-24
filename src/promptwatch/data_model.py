from pydantic import BaseModel, Field, validator, Extra
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union, Literal, Any
from uuid import uuid4

def validate_identifier(value, field):
    if value and ' ' in value:
        raise ValueError(f'{field} - whitespace not allowed')
    if value and len(value) > 256:
        raise ValueError('field length must be less than 256 characters')
    
class Session(BaseModel):
    """Base class for TracerSession."""
    id:str = Field(default_factory=lambda : str(uuid4()))
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    session_name:Optional[str]
    steps_count:int=0
    tracking_project:Optional[str]
    tracking_tenant:Optional[str]
    is_error:bool=False

    @validator('tracking_project')
    def check_tracking_project(cls, value):
        validate_identifier(value, "tracking_project")
        return value
    
    @validator('tracking_tenant')
    def check_tracking_tenant(cls, value):
        validate_identifier(value, "tracking_tenant")
        return value




class ActivityBase(BaseModel):
    
    id:str = Field(default_factory=lambda: str(uuid4()))
    order:Optional[int]
    parent_activity_id:Optional[str]=None
    start_time: datetime = Field(default_factory=lambda :datetime.now(tz=timezone.utc))
    end_time: Optional[datetime] = None
    metadata:Optional[Dict[str,Any]]
    error:Optional[str]=None
    info_message:Optional[str]=None

    # def __init__(self, *args, **kwargs: Any) -> None:
    #     if len(args)==1 and isinstance(args[0], Session):
    #         session=args[0]
    #     else:
    #         session:Session = kwargs.get("session")
            
    #     if session:
    #         kwargs["session_id"]=session.id
    #         session.steps_count+=1
    #         kwargs["order"]=session.steps_count
    #     return super(ActivityBase, self).__init__(**kwargs)

    class Config:
        extra = Extra.forbid
        
class ActivityList(BaseModel):
    __root__: List[ActivityBase] 

    def __init__(self, data:List[ActivityBase]):
        super().__init__(__root__=data)

class Question(ActivityBase):
    activity_type:Literal['question'] = 'question'
    text:str

    def __str__(self) -> str:
        return f"{self.activity_type}: {self.text}"



class DocumentSnippet(BaseModel):
    id:Optional[str]
    text:str
    source:Optional[str]
    metadata:Optional[Dict[str,Any]]

class RetrievedDocuments(ActivityBase):
    activity_type:Literal['retrieved-docs'] = 'retrieved-docs'
    documents:List[DocumentSnippet]
    scores:Optional[List[float]]

    def __str__(self) -> str:
        return f"{self.activity_type}: {len(self.documents)}"
    
class Answer(ActivityBase):
    activity_type:Literal['answer'] = 'answer'
    text:str

    def __str__(self) -> str:
        return f"{self.activity_type}: {self.text}"
    

class Action(ActivityBase):
    activity_type:Literal['action'] = 'action'
    tool_type:str
    input:str
    input_data:Optional[Dict[str,Any]]
    output:Optional[str]
    output_data:Optional[Dict[str,Any]]

    def __str__(self) -> str:
        return f"{self.activity_type}: {self.tool_type}({self.input}) \n\t-> {self.output}"


class ChainSequence(ActivityBase):
    activity_type:Literal['action'] = 'chain-sequence'
    sequence_type:str
    inputs:Dict[str,Any]
    outputs:Optional[Dict[str,Any]]

    def __str__(self) -> str:
        return f"{self.activity_type}: {self.sequence_type}"


class Log(ActivityBase):
    activity_type:Literal['log'] = 'log'
    text:str

    def __str__(self) -> str:
        return f"{self.activity_type}: {self.text}"




class LlmPrompt(ActivityBase):
    activity_type:Literal['llm-prompt'] = 'llm-prompt'
    prompt:Union[str,List["ChatMessage"]]
    prompt_template:Union["NamedPromptTemplateDescription", "PromptTemplateDescription", None ]
    prompt_input_values:Optional[Dict[str,Union[str,"ChatMessage", List["ChatMessage"]]]]
    caption:Optional[str]
    generated:Optional[str]

    def __str__(self) -> str:
        return f"{self.activity_type}: {self.generated}"

class ParallelPrompt(ActivityBase):
    activity_type:Literal['parallel-prompt'] = 'parallel-prompt'
    prompts:List[LlmPrompt]

    def __init__(self, thoughts:List[LlmPrompt]):
        super().__init__(self)
        for thought in thoughts:
            thought.parent_activity_id = self.id
        return self
    
    def __str__(self) -> str:
        thoughts_strings = '\n\t'.join(t.generated  for t in self.prompts)
        return f"{self.activity_type}: \t{thoughts_strings}"

    

    




class PromptTemplateDescription(BaseModel):
    prompt_template:Union[str, List[Union["ChatMessagePromptTemplate",str, "ChatMessage"]]]     # Union["ChatMessagePromptTemplate","ChatMessage",List["ChatMessage"]]]]
    prompt_input_params:Optional[List[str]]
    format:Optional[str]
    
class ChatMessage(BaseModel):
    """ Explicit chat message used as a parameter value for chat history or template"""
    role:Optional[str]
    text:str

class ChatMessagePromptTemplate(PromptTemplateDescription):
    role:Optional[str]
    
    
class NamedPromptTemplateDescription(PromptTemplateDescription):
    template_name:Optional[str]
    template_version:Optional[str]

    

LlmPrompt.update_forward_refs()
PromptTemplateDescription.update_forward_refs()
ChatMessagePromptTemplate.update_forward_refs()
NamedPromptTemplateDescription.update_forward_refs()