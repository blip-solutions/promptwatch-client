import json
import os
from typing import *
from promptwatch.constants import EnvVariables
from promptwatch.promptwatch_context import PromptWatch
from promptwatch.data_model import NamedPromptTemplateDescription,ActivityBase, LlmPrompt, ChainSequence
from .schema import *
from .evaluation import DEFAULT_COSINE_SCORE_EVALUATION, TestCaseEvaluationWrapper, EvaluationStrategyBase
import logging
from abc import ABC,abstractmethod
from tqdm import tqdm
from datetime import datetime, timezone


        

class UnitTest:

    def __init__(self, test_name:str, evaluation:EvaluationStrategyBase=DEFAULT_COSINE_SCORE_EVALUATION) -> None:
        """ Initialize new test definition. If called with the same parameters, it will be reused """
        self.test_name=test_name
        self.evaluation_strategy=evaluation
        self._entered=False
        #avoiding circular imports
        
        # it ok to create new instance... PromptWatch is a singleton and if someone want's to initialize extra params, he can wrapping it int  PromptWatch context 
        self.prompt_watch=None 
        self.conditions=None
        self.test_cases_generator=None
        from promptwatch.client import Client
        self.client=Client()

        self.evaluator=None
        
       
            

    def for_test_cases(self, examples:List["TestCase"])->"UnitTest":
        self.test_cases_generator = (e for e in examples)
        return self

    def for_test_cases_in_file(self, file_path:str)->"UnitTest":
        def iterate(file_path):
            with open(file_path) as f:
                
                for line in f:
                    try:
                        data = json.loads(line)
                    except Exception as e:
                        raise Exception(f"Failed to parse line {line} in file {file_path}. Data are expected to be in json line format.\nError: {e}") 
                    
                    test_case = TestCase(data)
                        
                    yield test_case

        self.test_cases_generator = iterate(file_path)
        self.conditions = PromptUnitTestConditions(for_test_cases_in_file=os.path.basename(file_path))
        return self
        
    
    def for_prompt_template(self,registered_template:Union[str, NamedPromptTemplateDescription]):
        """ This will pull and iterate all labeled runs for the registered template
        Args:
            param registered_template (str): registered_template name or instance of NamedPromptTemplateDescription (result of register_prompt_template)
         
        """
        # if isinstance(registered_template, NamedPromptTemplateDescription):
        #     for_template_name =  registered_template.template_name
        if isinstance(registered_template, str):
            for_template_name = registered_template
        else:
            raise ValueError("registered_template parameter must be either str or NamedPromptTemplateDescription")
        
        def iterate():
            results=None
            skip=0
            page_size=500
            while skip==0 or results:
                results= self.client.get_test_cases(for_template_name=for_template_name, skip=skip, limit=page_size)
                for res in results:
                    yield (res)
                skip+=page_size
                if len(results)<page_size:
                    break
                results=None
        
        self.conditions = PromptUnitTestConditions(for_template_name=for_template_name)
        self.test_cases_generator = iterate()
        return self
    
    def for_project_sessions(self,for_tracking_project:str=None):
        """  This will pull and iterate all top level chains in sessions that have been created with the context of the given tracking project
        
        """
        for_tracking_project=for_tracking_project or os.environ.get(EnvVariables.PROMPTWATCH_TRACKING_PROJECT)
        if not for_tracking_project:
            
            raise ValueError("tracking_project parameter is required")
        def iterate():
            results=None
            skip=0
            page_size=500
            while skip==0 or results:
                results= self.client.get_test_cases(for_tracking_project=for_tracking_project, skip=skip, limit=page_size)
                for res in results:
                    yield (res)
                skip+=page_size
                if len(results)<page_size:
                    break
                results=None
        self.conditions = PromptUnitTestConditions(for_tracking_project=for_tracking_project)
        self.test_cases_generator = iterate()
        return self


    def get_all_test_cases(self)->List[TestCase]:
        """ Useful when you just want to retrieve the test cases without running the test"""
        return list(self.test_cases_generator)
        

    
    def __enter__(self):
        
        self._entered=True
        if not self.test_cases_generator:
            raise Exception("Invalid use. Please define the scope of test by calling one of for_test_cases, for_test_cases_in_file, for_prompt_template or for_project_sessions methods before entering the context")
        
        self.prompt_watch=PromptWatch(tracking_project=self.conditions.for_tracking_project if self.conditions else None)
        self.prompt_watch.__enter__()
        self.prompt_watch.session_id
        self.unit_test_run = PromptUnitTestRun(
            test_name=self.test_name,
            session_id=self.prompt_watch.current_session.id, 
            start_time=datetime.now(tz=timezone.utc), 
            conditions=self.conditions,
            evaluation_method=self.evaluation_strategy.evaluation_method
            )
        self._session= UnitTestSession( 
            unit_test=self, 
            unit_test_run=self.unit_test_run, 
            evaluation_strategy=self.evaluation_strategy, 
            test_cases_generator=self.test_cases_generator
            )
        self._session.__enter__()
        self.prompt_watch.set_session_name(self.test_name)
        return self._session

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._entered=False
        self.prompt_watch.__exit__(exc_type, exc_val, exc_tb)
        self._session.__exit__(exc_type, exc_val, exc_tb)
        pass
    

class UnitTestSession:

    def __init__(self,unit_test:UnitTest, unit_test_run:PromptUnitTestRun,  evaluation_strategy: EvaluationStrategyBase,test_cases_generator:Iterator) -> None:
        """ Handle for running UnitTestSession """

        
        self.test_name = unit_test.test_name
        self.prompt_watch=unit_test.prompt_watch
        
        # entering the PromptWatch context to start the session
        
        # hooking up the activity callback to capture the LlmPrompt
        self.prompt_watch.add_on_activity_callback(self._add_llm_prompt_activity_to_result)
        self.client=self.prompt_watch.client

        self.unit_test_run=unit_test_run
        self.client.save_unit_test_run(self.unit_test_run)
        self.test_results=self.unit_test_run.results
        
        self.evaluation_strategy=evaluation_strategy
        self._test_cases_generator=test_cases_generator
        
        #init fields       
        self._pending_test_case:TestCaseEvaluationWrapper=None
        self.outbound_stack=[]
        
        
    


    def __enter__(self):
        return self
   

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._pending_test_case:
            self.add_failed_result(self._pending_test_case.inner_test_case, str(exc_val))
        self.unit_test_run.end_time=datetime.now(tz=timezone.utc)
        self.print_results()
        try:
            self.client.save_unit_test_run(self.unit_test_run)
            if self.outbound_stack:
                self.client.save_unit_test_cases_result(self.test_name, self.unit_test_run.run_id,test_case_results=self.outbound_stack)
        except Exception as e:
            raise Exception(f"Failed to save unit test results. Error: {e}")

    @property
    def langchain(self):
        if not hasattr(self, "langchain_tools"):
            try:
                import langchain
                from ..langchain.unit_tests import LangChainTools
                self.langchain_tools = LangChainTools(self)
            except ImportError as e:
                raise ImportError("Failed to import langchain. Please install langchain to use this features")
        return self.langchain_tools

    def add_evaluation_result(self, test_case_result:"TestCaseResult"):
        if self._pending_test_case is not None and self._pending_test_case.inner_test_case!=test_case_result.test_case:
            raise Exception("Invalid call, you can't set evaluation result for an example while another example is already pending.")
        self.outbound_stack.append(test_case_result)
        score = test_case_result.score  if test_case_result.score is not None else 0
        if self.test_results.total_processed>=1:
            self.test_results.total_processed += 1
            # Calculate the new overall score ... -> de-average the old score and add the new score .. and average again
            # new_total_score = ((total_score * N-1) + new_score) / (N)
            self.test_results.overall_score =((self.test_results.overall_score*(self.test_results.total_processed-1)) + score) /self.test_results.total_processed 
        else:
            self.test_results.total_processed += 1
            self.test_results.overall_score = score
        
        if test_case_result.passed:
            self.test_results.passed+=1
        else:
            self.test_results.failed+=1
        self._pending_test_case=None
        try:
            self.client.save_unit_test_cases_result(self.test_name, self.unit_test_run.run_id,test_case_results= self.outbound_stack)
            self.outbound_stack.clear()
        except Exception as e:
            logging.info(f"Failed to send test case result: {e}. Will retry later")
        
        
    def add_failed_result(self, example:"TestCase", error_description:str):
        self.add_evaluation_result(TestCaseResult(test_case=example, score=0, error_description=str(error_description), passed=False))
           
        
    def test_cases(self)->Iterator["TestCaseEvaluationWrapper"]:
        for example in tqdm(self._test_cases_generator, desc=f"Running test {self.test_name}"):
            if self._pending_test_case:
                raise Exception("Invalid call, you can't get next example until you set evaluation result for the previous one. Use `test_case.evaluate_result(generated_str)` method before iterating to next example. ")
            self._pending_test_case=TestCaseEvaluationWrapper(test_session=self, test_case=example)
            yield self._pending_test_case

    def skip(self, n=1):
        """ Useful when you want to skip some examples, without evaluating them.
        Args:
            n (int, optional): Number of examples to skip. Defaults to 1.
        """
        print("skipping")
        if self._pending_test_case:
            _skipped=[self._pending_test_case]
            self._pending_test_case=None
        else:
            _skipped=[]
        try:
            for _ in range(n-1):
                _skipped.append(next(self._test_cases_generator))
        except StopIteration:
            pass
        return _skipped


    
    def _add_llm_prompt_activity_to_result(self, activity:ActivityBase):
        """ Callback to capture the LlmPrompt """
        if self.unit_test_run.conditions and self.unit_test_run.conditions.for_template_name and self._pending_test_case and isinstance(activity,LlmPrompt):
            self._pending_test_case.llm_prompt =activity
            self._pending_test_case.activity_id=activity.id
        elif self.unit_test_run.conditions and self.unit_test_run.conditions.for_tracking_project and isinstance(activity,ChainSequence):
            if not self._pending_test_case.activity_id and activity.parent_activity_id is None:
                self._pending_test_case.activity_id=activity.id

        
        

    def print_results(self):
        # Define some color codes
        RED = '\033[31m'
        GREEN = '\033[32m'
        MAGENTA = '\033[35m'

        # Define reset code to restore the default text color
        RESET = '\033[0m'

        print("--------------")
        print(f"Unit test {MAGENTA}{self.unit_test_run.test_name}{RESET} finished")
        if self.unit_test_run.end_time is not None and self.unit_test_run.start_time is not None:
            total_duration_sec=(self.unit_test_run.end_time-self.unit_test_run.start_time).total_seconds()
        print(f"  Total Duration: {MAGENTA}{total_duration_sec:.2f}{RESET} seconds")
        print("--------------")
        print("Results:")

        if self.unit_test_run.results is not None:
            
            print(f"  Overall Score: {MAGENTA}{round((self.unit_test_run.results.overall_score or 0)*100 )}%{RESET}")

            print(f"  Total Processed: {MAGENTA}{self.unit_test_run.results.total_processed}{RESET}")
            print(f"  Passed: {GREEN if self.unit_test_run.results.passed else MAGENTA }{self.unit_test_run.results.passed}{RESET}")
            print(f"  Failed: {RED  if self.unit_test_run.results.failed else MAGENTA }{self.unit_test_run.results.failed}{RESET}")
        print("--------------")
        link_url = f"https://www.promptwatch.io/unit-tests?unit-test-run={self.unit_test_run.run_id}"
        print(f"{MAGENTA}You can see the full results here: {link_url}{RESET}")

        