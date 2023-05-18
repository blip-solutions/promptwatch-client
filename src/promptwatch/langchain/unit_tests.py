from typing import Callable, Union,List
from promptwatch.unit_tests.schema import Callable, Union
from langchain import  LLMChain
from promptwatch.data_model import ChatMessage
from .langchain_support import reconstruct_langchain_chat_messages
from typing import Dict, Any
import types

from ..unit_tests.unit_tests import UnitTestSession
from langchain.schema import BaseMemory

from langchain.chains.base import Chain
from langchain.agents import Agent
from langchain.llms.base import LLM
from langchain.chat_models.base import BaseChatModel


    
def convert_inputs(inputs: dict) -> dict:
        res = {}
        for k,v in inputs.items():
            if isinstance(v,list):
                # if it is a list, it is likely a list of messages for history
                try:
                    # let's try to parse it as a list of messages
                    promptwatch_chat_msgs = [ChatMessage.parse_obj(m) for m in v]

                    res[k]=reconstruct_langchain_chat_messages(promptwatch_chat_msgs)
                except:
                    res[k]=v
            else:
                res[k]=v
        return res

class TestCaseMemory(BaseMemory):
    """Memory that will always include the right inputs from testcase, in order to reproduce the same conditions as for the test case"""
    unit_test_session: UnitTestSession  #: :meta private:
    memory_key: str 


    @property
    def memory_variables(self) -> List[str]:
        res = list(self.unit_test_session._pending_test_case.inputs.keys())
        if self.memory_key not in res:
            res.append(self.memory_key)
        return res

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        res = convert_inputs(self.unit_test_session._pending_test_case.inputs)
        if self.memory_key not in res:
            res[self.memory_key] = []
        return res

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Nothing should be saved"""
        pass

    def clear(self) -> None:
        """Nothing to clear"""
        pass


class LangChainTools:
    """Tools for running UnitTests"""

    def __init__(self, unit_test_session: UnitTestSession):
        self._unit_test_session = unit_test_session
        self._memory=None
        

    def get_test_memory(self, memory_key:str= "history" ) -> TestCaseMemory:
        """Will return LangChain memory that will always include the right inputs from TestCase, in order to reproduce the same inputs"""
        if self._memory is None:
            self._memory = TestCaseMemory(unit_test_session=self._unit_test_session, memory_key=memory_key)
        return self._memory
    

    def get_eval_result_for_langchain_object(self, langchain_object:Union[Chain, Agent, LLM, BaseChatModel]):
        """Will evaluate langchain_object, and return the output"""
        if isinstance(langchain_object,Chain):
            return langchain_object.run()
        elif isinstance(langchain_object,Agent):
            return langchain_object.act()
        elif isinstance(langchain_object,LLM):
            return langchain_object.predict()
        elif isinstance(langchain_object,BaseChatModel):
            return langchain_object.chat()
        else:
            raise Exception(f"Unknown langchain_object type: {type(langchain_object)}")
