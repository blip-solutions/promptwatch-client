from .data_model import Session, ActivityBase, ActivityList
import aiohttp
from typing import Dict, List, Optional, Tuple, Any, Union
import os
import logging
from datetime import datetime
from .unit_tests.schema import TestCase, PromptUnitTestRun, TestCaseResult, TestCaseResultsList


class AsyncClient:

    def __init__(self, api_key: str = None) -> None:
        if not api_key:
            api_key = os.environ.get("PROMPTWATCH_API_KEY")
        if not api_key:
            raise Exception("Unable to find PromptWatch API key. Either set api key as a parameter to PromptWatch(api_key='<your api key>') or set it up as an env. variable PROMPTWATCH_API_KEY. You can generate your API key here: https://app.promptwatch.io/get-api-key")
        self.api_key = api_key
        self.logger = logging.getLogger("PromptWatchAsyncClient")
        self.tracker_api_host_url = os.environ.get("PROMPTWATCH_HOST_URL", "https://api.promptwatch.io").rstrip("/")
        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        self.default_timeout = 5
        self._http_session = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp session"""
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession(headers=self.headers)
        return self._http_session

    async def close(self) -> None:
        """Close the HTTP session"""
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
            self._http_session = None
    
    async def __aenter__(self):
        """Support for async context manager pattern"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup when exiting context manager"""
        await self.close()

    async def _request(
        self, 
        method: str, 
        endpoint: str, 
        data_str: Optional[str] = None,
        json_dict: Optional[dict] = None, 
        params: Optional[dict] = None, 
        timeout: Optional[float] = None, 
        ignore_errors: bool = True
    ) -> Optional[aiohttp.ClientResponse]:
        """Centralized async request handler for all API calls"""
        url = f"{self.tracker_api_host_url}{endpoint}"
        timeout_obj = aiohttp.ClientTimeout(total=timeout or self.default_timeout)
        session = await self._get_session()

        try:
            async with session.request(
                method=method, 
                url=url, 
                json=json_dict, 
                data=data_str, 
                params=params, 
                timeout=timeout_obj
            ) as response:
                if response.status >= 300:
                    error_msg = f"Error sending data to the server: Error {response.status} - {response.reason} ({await response.text()})"
                    if not ignore_errors:
                        raise Exception(error_msg)
                    self.logger.error(error_msg)
                    return None
                
                # Ensure the response is fully read
                await response.read()
                return response
        except Exception as ex:
            self.logger.exception(ex)
            if not ignore_errors:
                raise
            return None

    async def start_session(self, session: Session) -> None:
        await self._request("POST", "/sessions/start", data_str=session.json())
        
    async def finish_session(self, session: Session) -> None:
        await self._request("POST", "/sessions/finish", data_str=session.json())

    async def restore_session(self, session_id: str) -> Optional[Session]:
        response = await self._request("POST", f"/sessions/{session_id}/restore")
        if not response:
            return None
            
        data = await response.json()
        session_data = data.get("session")
        if session_data:
            return Session(**session_data)
        return None

    async def save_activities(self, session_id: str, activities: List[ActivityBase]) -> None:
        payload = ActivityList(activities)
        await self._request(
            "POST", 
            f"/sessions/{session_id}/activities", 
            data_str=payload.json()
        )
        
    async def get_session(self, session_id: str) -> Session:
        response = await self._request(
            "GET", 
            f"/sessions/{session_id}", 
            ignore_errors=False
        )
        
        if response:
            data = await response.json()
            return Session(**data)
        raise Exception(f"Failed to retrieve session: {session_id}")


    
    async def get_test_cases(
        self, 
        for_tracking_project=None, 
        for_template_name=None, 
        skip=0, 
        limit=1000
    ) -> List[TestCase]:
        query_params = {
            "for_tracking_project": for_tracking_project,
            "for_template_name": for_template_name,
            "skp": skip,
            "limit": limit
        }
        
        response = await self._request(
            "GET", 
            "/test-cases", 
            params=query_params, 
            timeout=None, 
            ignore_errors=False
        )
        
        if response:
            data = await response.json()
            if data:
                return [TestCase(**item) for item in data]
        return []
        
    async def save_unit_test_run(self, unit_test_run: PromptUnitTestRun) -> None:
        await self._request(
            "POST", 
            f"/tests/{unit_test_run.test_name}/runs", 
            data_str=unit_test_run.json(), 
            timeout=None, 
            ignore_errors=False
        )

    async def save_unit_test_cases_result(self, test_name: str, run_id: str, test_case_results: List[TestCaseResult]) -> None:
        await self._request(
            "POST", 
            f"/tests/{test_name}/runs/{run_id}/result-details", 
            data_str=TestCaseResultsList(__root__=test_case_results).json(), 
            timeout=None, 
            ignore_errors=False
        )
