from typing import *

from pydantic import BaseModel, validator
from functools import lru_cache
from math import sqrt
from abc import ABC, abstractmethod, abstractproperty
import json
from typing import TYPE_CHECKING, Any
from .schema import TestCase, TestCaseResult, LabeledOutput

if TYPE_CHECKING:
    from .unit_tests import UnitTestSession

try:
    from langchain.embeddings.base import Embeddings
except ImportError:
    class Embeddings:
        """ Fake class to mock missing langchain dependency."""
        pass

class EvaluationStrategyBase(ABC):
    
    @property
    @abstractmethod
    def evaluation_method(self)->str:
        pass

    @abstractmethod
    def evaluate(self, generated_result:str, test_case:"TestCase")->"TestCaseResult":
        pass
    
    

class CosineScoreEvaluationStrategy(EvaluationStrategyBase):
    """ Evaluation strategy that uses cosine similarity to compare generated result with expected result.

    For negative examples (label=0) the score is calculated as cosine distance (1-cosine_similarity) from the (un)-expected result
    For positive examples (label=1) the score is calculated as cosine similarity to the expected result

    """

    def __init__(self, embed_function:Union[str,Callable[[str],List[float]],Embeddings]= "OpenAIEmbeddings", positive_score_threshold=0.90, negative_score_threshold=0.1, auto_unwrap_dict=True ) -> None:
        """ Evaluation strategy that uses cosine similarity to compare generated result with expected result.

        For negative examples (label=0) the score is calculated as cosine distance (1-cosine_similarity) from the (un)-expected result
        For positive examples (label=1) the score is calculated as cosine similarity to the expected result

        Due to the nature of how cosine distance works, having result far from negative example will still result in low score... which is desired behavior, 
        as it is not possible to know what is the "best" result for negative example, but this also means that adding test case with only negative examples will distort the score.

        If more examples for the same input is provided, only the "best" score is used.


        Args:
            embed_function (Union[str,Callable[[str],List[float]],Embeddings], optional): Embedding function to use. Defaults to "OpenAIEmbeddings".
            positive_score_threshold (float, optional): Score threshold for positive examples to be considered as passed. Defaults to 0.9
            negative_score_threshold (float, optional): Score threshold for negative examples to be considered as passed.  Defaults to 0.1
            auto_unwrap_dict (bool, optional): If True, will automatically unwrap dict values if only one value is inside the dict. Defaults to True. (this is useful when comparing output from a chain )
        """

        

        self.embed_function=embed_function
        self.positive_score_threshold=positive_score_threshold
        self.negative_score_threshold=negative_score_threshold
        self.auto_unwrap_dict=auto_unwrap_dict

    @property
    def evaluation_method(self)->str:
            return "cos_sim_score"

    @lru_cache(maxsize=1000)
    def _embed(self,str_value)->List[float]:
        if not hasattr(self,"embedding_generator"):
            if callable(self.embed_function):
                self.embedding_generator=self.embed_function

            elif isinstance(self.embed_function,Embeddings):
                self.embedding_generator =  self.embed_function.embed_query
            elif isinstance(self.embed_function,str):
                if self.embed_function=="OpenAIEmbeddings":
                    from langchain.embeddings import OpenAIEmbeddings
                    self.embedding_generator =  OpenAIEmbeddings().embed_query
                else:
                    raise ValueError(f"Unknown embed_function {self.embed_function}. Please provide langchain embeddings or a function that takes a string and returns a list of floats.")
            else:
                raise ValueError("embed_function must be a function or a langchain embeddings object")
        
        return self.embedding_generator(str_value)


    def _get_cosine_similarity(self,example_1:str, example_2:str)->float:
        """ Safe cosine similarity function that is not relying on normalized vectors."""
        vec1 = self._embed(example_1)
        vec2 = self._embed(example_2)
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sqrt(sum(a ** 2 for a in vec1))
        magnitude2 = sqrt(sum(b ** 2 for b in vec2))
        return dot_product / (magnitude1 * magnitude2)


    def cosine_similarity(self, generated_result:Union[str,dict], eval_example:"LabeledOutput")->float:
        generated_result_str = generated_result if isinstance(generated_result,str) else json.dumps(generated_result)
        eval_example_str = eval_example.value if isinstance(eval_example.value,str) else json.dumps(eval_example.value)
        if eval_example.label==1: 
            return self._get_cosine_similarity(generated_result_str,eval_example_str)
        else:
            return self._get_cosine_similarity(generated_result_str, eval_example_str)


    def evaluate(self,generated_result:Union[str,dict], test_case:"TestCase")->"TestCaseResult":
        closest_to_positive=-1
        closest_to_negative=-1
        closest_positive_sample=None
        closest_negative_sample=None
        error_reasoning=None


        for output in test_case.outputs:
            if self.auto_unwrap_dict and isinstance(generated_result,str) and isinstance(output.value,dict) and len(output.value)==1 :
                # this is because LLMChain returns dict with one value
                output.value= list(output.value.values())[0]

            
            if type(generated_result) != type(output.value):
                score=0
                error_reasoning=f"Type mismatch (expected {type(output.value).__name__}, got {type(generated_result).__name__})"
                break
            elif isinstance(generated_result,dict) and isinstance(output.value,dict):
                # if we need to check the structure
                if set(generated_result.keys()) != set(output.value.keys()):
                    score=0
                    error_reasoning=f"Missing keys: {set(output.value.keys()) - set(generated_result.keys())} \n Unexpected keys: {set(generated_result.keys()) - set(output.value.keys())}"
                    break
                

            cosine_similarity = self.cosine_similarity(generated_result, output)
            if output.label==1 and cosine_similarity > closest_to_positive:
                closest_to_positive = cosine_similarity
                closest_positive_sample=output
                
            elif  output.label==0 and  cosine_similarity > closest_to_negative:
                closest_to_negative = cosine_similarity
                closest_negative_sample=output
        
        if error_reasoning:
            passed=False
            score=0
            closest_sample=None
        else:
            max_score=max(closest_to_positive,closest_to_negative)
            if max_score==closest_to_positive:
                # we priorities positive match 
                score = max(closest_to_positive,0) # we cap the score at 0
                passed=score>=self.positive_score_threshold
                closest_sample =closest_positive_sample

            else:
                score = 1-max(closest_to_negative,0) # we calculate negative score as distance from the negative example ... the further the better
                passed=score>=self.negative_score_threshold
                closest_sample =closest_negative_sample
                # Score for negative example is not comparable to similarity to positive example
                # we however still wan't so give some credit to the fact that the generated result was further away from the negative example.. 
                # the further the better, but it still doesn't mean it's a good result

        
        eval_metadata={
                "positive_score_threshold":self.positive_score_threshold,
                "negative_score_threshold":self.negative_score_threshold
        }
        if closest_sample:
            eval_metadata["closest_reference_example"]=closest_sample.dict()
        return TestCaseResult(
            test_case=test_case, 
            passed=passed, 
            score=score, 
            evaluation_method=self.evaluation_method, 
            generated_result=generated_result,
            reasoning=error_reasoning,
            evaluation_metadata=eval_metadata
            )
        

class TestCaseEvaluationWrapper:

    def __init__(self, test_session: "UnitTestSession", test_case:"TestCase") -> None:
        self.unit_test_session:UnitTestSession=test_session
        self.inner_test_case=test_case
        self.llm_prompt=None
        self.activity_id=None
        self.iterations=0
        
        
        
    @property
    def inputs(self)->dict:
        return self.inner_test_case.inputs
    

        
    
    @property
    def outputs(self)->List["LabeledOutput"]:
        return self.inner_test_case.outputs
        
    def evaluate(self, result_generator:Callable[..., None], continue_on_error:bool=False):
        """ 
        Evaluate a LLM generation function over the expected results 
        Args:
            param result_generator (str): a function that takes the inputs and returns a string ... usually a LLM 

        Example with LangChain:

        ```python
        from langchain import PromptTemplate, LLMChain
        from langchain.llms import OpenAI
        from promptwatch import register_prompt_template

        registered_prompt_template = register_prompt_template("my_template_name",PromptTemplate.from_template("Finish this sentence: {sentence}"))
        llm = OpenAI()
        llmChain = LLMChain(llm, prompt=registered_prompt_template)
        with UnitTest("my_test").for_template_name("my_template_name") as test:
            for example in test.test_cases():
                example.evaluate(llmChain)
            
        ```
        
        """
        if self.iterations==0:
            # sanity checks
            try:
                from langchain import LLMChain
                if self.unit_test_session.unit_test_run.conditions and self.unit_test_session.unit_test_run.conditions.for_template_name:
                    # if we are running test for template, we expect to evaluate llm_chain.run method, not the instance itself
                    if isinstance(result_generator, LLMChain):
                        result_generator=result_generator.run
                    
            except ImportError as e:
                # we ignore error if langchain cant be imported
                pass


        try:
            generated = result_generator(self.inputs)
            if self.unit_test_session.unit_test_run.conditions and self.unit_test_session.unit_test_run.conditions.for_template_name and not isinstance(generated, str):
                raise Exception(f"we expect that the evaluated method/object to return a string for 'for_template_name' unit test, but got: {type(generated)}")
                

            if hasattr(result_generator,"output_keys") and isinstance(generated,dict):
                # langchain chains have input / output_keys and also options for return more than output. (return intermediate results, return only outputs etc.)
                # we want to compare ONLY outputs
                output_keys = result_generator.output_keys
                #filter out the results
                generated = {k:v for k,v in generated.items() if k in output_keys}

            self.evaluate_result(generated)
        except Exception as e:
                
            self.mark_as_failed(str(e))
            if "Missing some input keys:" in str(e):
                # this is likely due to missing test memory...
                raise Exception(f"Got exception {str(e)}. \nThis is likely due to missing test memory. Please see: docs.promptwatch.com/docs/unit_testing/unit_tests_reference_guide#langchain-test-memory")
            
            if not continue_on_error:
                raise e
            
        self.iterations+=1

    def skip(self):
        self.unit_test_session.skip()

    def mark_as_failed(self, error_description:str):
        """ Useful to log an error during the generation.
        Generally speaking, if you handle generation on your own, you should wrap it into try catch block so the test could continue. You can use this method to log the error and mark the results for this example as failed.
        """
        self.unit_test_session.add_failed_result(self.inner_test_case, error_description=str(error_description))

    def evaluate_result(self, generated_result:str):
        """Evaluate the final generated results. 
        Unlike `evaluate`, this method is used when you already have the generated result and you want to evaluate it.

        Args:
            generated_result (str): the result to compare with the expected results
        """
        eval_result = self.unit_test_session.evaluation_strategy.evaluate(generated_result, self.inner_test_case)
        eval_result.llm_prompt=self.llm_prompt
        eval_result.activity_id=self.activity_id
        self.unit_test_session.add_evaluation_result(eval_result)

DEFAULT_COSINE_SCORE_EVALUATION=CosineScoreEvaluationStrategy()