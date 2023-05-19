from .data_model import Session, ActivityBase, ActivityList
import requests
from typing import List, Optional,Tuple
import os
import logging
from datetime import datetime
from .unit_tests.schema import TestCase,  PromptUnitTestRun, TestCaseResult, TestCaseResultsList
from pydantic.json import pydantic_encoder


class Client:

    def __init__(self, api_key:str=None) -> None:
        if not api_key:
            api_key=os.environ.get("PROMPTWATCH_API_KEY")
        if not api_key:
            raise Exception("Unable to find PromptWatch API key. Either set api key as a parameter to PromptWatch(api_key='<your api key>') or set it up as an env. variable PROMPTWATCH_API_KEY. You can generate your API key here: https://app.promptwatch.io/get-api-key")
        self.api_key=api_key    
        self.logger = logging.getLogger("PromptWatchClient")
        self.tracker_api_host_url = os.environ.get("PROMPTWATCH_HOST_URL","https://api.promptwatch.io").rstrip("/")
        self.headers={
                            "x-api-key":self.api_key, 
                            "Content-Type":"application/json"
                            }
                        
    def _request(self, method:str, endpoint:str, data_str:Optional[str]=None,json_dict:Optional[dict]=None, params:Optional[dict]=None, timeout=5, ignore_errors=True)-> requests.Response:

        try:
            response = requests.request(method, f"{self.tracker_api_host_url}{endpoint}", json=json_dict, data=data_str, params=params, headers=self.headers, timeout=timeout)
            if response.status_code>300:
                if not ignore_errors:
                    raise Exception(f"Error sending data to the server: Error {response.status_code} - {response.reason} ({response.text})")
                self.logger.error(f"Error sending data to the server: Error {response.status_code} - {response.reason} ({response.text})")
                return None
            return response
        except Exception as ex:
            self.logger.exception(ex)
            if not ignore_errors:
                raise ex

    
    def start_session(self, session:Session):
       self._request("POST", f"/sessions/start", data_str=session.json())
    def finish_session(self, session:Session):
       self._request("POST", f"/sessions/finish", data_str=session.json())

    def restore_session(self, session_id:str):
       response = self._request("POST", f"/sessions/{session_id}/restore")
       data = response.json()
       session_data = data.get("session")
       if session_data:
           return Session(**session_data)
       

    def save_activities(self, session_id:str,   activities:List[ActivityBase]):
    
        payload = ActivityList(activities)
        self._request("POST",f"/sessions/{session_id}/activities", data_str=payload.json(), ignore_errors=True)
        
        
    def get_session(self, session_id:str)-> Session:
        response = requests.get(f"{self.tracker_api_host_url}/sessions/{session_id}", headers=self.headers)
        
    
        if response.status_code==200:
            return Session(**response.json())
        else:
            raise Exception(f"Error response from server: {response.status_code}: {response.text}")
        

    def get_from_cache(self, cache_namespace_key:str,  query_embedding:List[float], min_similarity:float)-> Tuple[str,float]:
        response = self._request("POST", f"/prompt-cache/{cache_namespace_key}/get", json_dict={"embedding":query_embedding},params={ "min_similarity":min_similarity})
        if response:
            data = response.json()
            if data:
                result = data.get("result")
                similarity = data.get("similarity")
                return result, similarity
        
    def add_into_cache(self, cache_namespace_key:str, id:str, embedding:List[float], result:str):
       response = self._request("POST", f"/prompt-cache/{cache_namespace_key}", json_dict={"embedding":embedding, "id":id,"result":result})
    
    def clear_cache(self, cache_namespace_key:str, until:Optional[datetime]=None):
       response = self._request("POST", f"/prompt-cache/{cache_namespace_key}/clear", params={ "until":until})
    

    
    def get_test_cases(self, for_tracking_project=None, for_template_name=None, skip=0, limit=1000)->TestCase:
        
        query_params = {
            "for_tracking_project": for_tracking_project,
            "for_template_name": for_template_name,
            "skp": skip,
            "limit": limit
        }
        response = self._request("GET", f"/test-cases", params=query_params,  timeout=None, ignore_errors=False)
        if response:
            data = response.json()
            if data:
                return [TestCase(**item) for item in data]
            else:
                return []
        
    def save_unit_test_run(self, unit_test_run: PromptUnitTestRun, timeout=None):
        
      
        response = self._request("POST", f"/tests/{unit_test_run.test_name}/runs", data_str=unit_test_run.json(), timeout=None, ignore_errors=False)
       

    def save_unit_test_cases_result(self, test_name, run_id, test_case_results:List[TestCaseResult]):
        response = self._request("POST", f"/tests/{test_name}/runs/{run_id}/result-details", data_str=TestCaseResultsList(__root__=test_case_results).json(), timeout=None, ignore_errors=False)
        
       