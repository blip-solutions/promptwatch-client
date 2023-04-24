from .data_model import Session, ActivityBase, ActivityList
import requests
from typing import List, Optional
import os
import logging




class Client:

    def __init__(self, api_key:str) -> None:
        self.api_key=api_key    
        self.logger = logging.getLogger("PromptWatchClient")
        self.tracker_api_host_url = os.environ.get("PROMPTWATCH_HOST_URL","https://api.promptwatch.io").rstrip("/")
        self.headers={
                            "x-api-key":self.api_key, 
                            "Content-Type":"application/json"
                            }
                        
    def _request(self, method:str, endpoint:str, data:Optional[dict]=None, params:Optional[dict]=None)-> requests.Response:

        try:
            response = requests.request(method, f"{self.tracker_api_host_url}{endpoint}", data=data, params=params, headers=self.headers)
            if response.status_code>300:
                self.logger.error(f"Error sending data to the server: Error {response.status_code} - {response.reason} ({response.text})")
            return response
        except Exception as ex:
            self.logger.exception(ex)

    
    def start_session(self, session:Session):
       self._request("POST", f"/sessions/start", data=session.json())
    def finish_session(self, session:Session):
       self._request("POST", f"/sessions/finish", data=session.json())

    def restore_session(self, session_id:str):
       response = self._request("POST", f"/sessions/{session_id}/restore")
       data = response.json()
       session_data = data.get("session")
       if session_data:
           return Session(**session_data)
       

    def save_activities(self, session_id:str,   activities:List[ActivityBase]):
    
        payload = ActivityList(activities)
        self._request("POST",f"/sessions/{session_id}/activities", data=payload.json())
        
        
    def get_session(self, session_id:str)-> Session:
        response = requests.get(f"{self.tracker_api_host_url}/sessions/{session_id}", headers=self.headers)
        
    
        if response.status_code==200:
            return Session(**response.json())
        else:
            raise Exception(f"Error response from server: {response.status_code}: {response.text}")
        
