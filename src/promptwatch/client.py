from .data_model import Session, ActivityBase, ActivityList
import requests
from typing import Dict, List, Optional, Tuple, Any, Union
import os
import logging
from datetime import datetime
from .unit_tests.schema import (
    TestCase,
    PromptUnitTestRun,
    TestCaseResult,
    TestCaseResultsList,
)


class Client:

    def __init__(self, api_key: str = None) -> None:
        if not api_key:
            api_key = os.environ.get("PROMPTWATCH_API_KEY")
        if not api_key:
            raise Exception("Unable to find PromptWatch API key. Either set api key as a parameter to PromptWatch(api_key='<your api key>') or set it up as an env. variable PROMPTWATCH_API_KEY. You can generate your API key here: https://app.promptwatch.io/get-api-key")
        self.api_key = api_key
        self.logger = logging.getLogger("PromptWatchClient")
        self.tracker_api_host_url = os.environ.get(
            "PROMPTWATCH_HOST_URL", "https://api.promptwatch.io"
        ).rstrip("/")
        self.http_session = requests.Session()
        self.http_session.headers.update(
            {"x-api-key": self.api_key, "Content-Type": "application/json"}
        )
        self.default_timeout = 5

    def _request(
        self,
        method: str,
        endpoint: str,
        data_str: Optional[str] = None,
        json_dict: Optional[dict] = None,
        params: Optional[dict] = None,
        timeout: Optional[int] = None,
        ignore_errors: bool = True,
    ) -> Optional[requests.Response]:
        """Centralized request handler for all API calls"""
        url = f"{self.tracker_api_host_url}{endpoint}"
        timeout = timeout or self.default_timeout

        try:
            response = self.http_session.request(
                method=method,
                url=url,
                json=json_dict,
                data=data_str,
                params=params,
                timeout=timeout,
            )

            if response.status_code > 300:
                error_msg = f"Error sending data to the server: Error {response.status_code} - {response.reason} ({response.text})"
                if not ignore_errors:
                    raise Exception(error_msg)
                self.logger.error(error_msg)
                return None
            return response
        except Exception as ex:
            self.logger.exception(ex)
            if not ignore_errors:
                raise
            return None

    def start_session(self, session: Session) -> None:
        self._request("POST", "/sessions/start", data_str=session.json())

    def finish_session(self, session: Session) -> None:
        self._request("POST", "/sessions/finish", data_str=session.json())

    def restore_session(self, session_id: str) -> Optional[Session]:
        response = self._request("POST", f"/sessions/{session_id}/restore")
        if not response:
            return None

        data = response.json()
        session_data = data.get("session")
        if session_data:
            return Session(**session_data)
        return None

    def save_activities(self, session_id: str, activities: List[ActivityBase]) -> None:
        payload = ActivityList(activities)
        self._request(
            "POST", f"/sessions/{session_id}/activities", data_str=payload.json()
        )

    def get_session(self, session_id: str) -> Session:
        response = self._request("GET", f"/sessions/{session_id}", ignore_errors=False)

        if response and response.status_code == 200:
            return Session(**response.json())
        raise Exception(f"Failed to retrieve session: {session_id}")

    def get_test_cases(
        self, for_tracking_project=None, for_template_name=None, skip=0, limit=1000
    ) -> List[TestCase]:
        query_params = {
            "for_tracking_project": for_tracking_project,
            "for_template_name": for_template_name,
            "skp": skip,
            "limit": limit
        }

        response = self._request(
            "GET", "/test-cases", params=query_params, timeout=None, ignore_errors=False
        )

        if response:
            data = response.json()
            if data:
                return [TestCase(**item) for item in data]
        return []

    def save_unit_test_run(self, unit_test_run: PromptUnitTestRun) -> None:
        self._request(
            "POST",
            f"/tests/{unit_test_run.test_name}/runs",
            data_str=unit_test_run.json(),
            timeout=None,
            ignore_errors=False,
        )

    def save_unit_test_cases_result(
        self, test_name: str, run_id: str, test_case_results: List[TestCaseResult]
    ) -> None:
        self._request(
            "POST",
            f"/tests/{test_name}/runs/{run_id}/result-details",
            data_str=TestCaseResultsList(__root__=test_case_results).json(),
            timeout=None,
            ignore_errors=False,
        )
