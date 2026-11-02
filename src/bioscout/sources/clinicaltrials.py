"""ClinicalTrials.gov API v2.

Docs: https://clinicaltrials.gov/data-api/api
"""

from __future__ import annotations

import json
from typing import Any

from bioscout.http import CachedHttp
from bioscout.models import TrialSearchResult, TrialSummary

CTGOV = "https://clinicaltrials.gov/api/v2"

STATUSES = {
    "RECRUITING",
    "NOT_YET_RECRUITING",
    "ACTIVE_NOT_RECRUITING",
    "COMPLETED",
    "TERMINATED",
    "WITHDRAWN",
    "SUSPENDED",
    "ENROLLING_BY_INVITATION",
    "UNKNOWN",
}


class ClinicalTrials:
    def __init__(self, http: CachedHttp) -> None:
        self.http = http

    async def search(
        self,
        query: str | None = None,
        condition: str | None = None,
        intervention: str | None = None,
        status: list[str] | None = None,
        max_results: int = 10,
    ) -> TrialSearchResult:
        params: dict[str, Any] = {"pageSize": max(1, min(max_results, 50)), "format": "json"}
        if query:
            params["query.term"] = query
        if condition:
            params["query.cond"] = condition
        if intervention:
            params["query.intr"] = intervention
        wanted = [s.upper() for s in (status or []) if s.upper() in STATUSES]
        if wanted:
            params["filter.overallStatus"] = ",".join(wanted)
        data = json.loads(await self.http.get_text(f"{CTGOV}/studies", params))
        label = " / ".join(x for x in (query, condition, intervention) if x)
        return TrialSearchResult(query=label, trials=[parse_study(s) for s in data.get("studies", [])])


def parse_study(study: dict[str, Any]) -> TrialSummary:
    p = study.get("protocolSection", {})
    ident = p.get("identificationModule", {})
    status = p.get("statusModule", {})
    design = p.get("designModule", {})
    arms = p.get("armsInterventionsModule", {})
    return TrialSummary(
        nct_id=ident.get("nctId", ""),
        title=ident.get("briefTitle", ""),
        status=status.get("overallStatus", ""),
        phases=design.get("phases", []),
        conditions=p.get("conditionsModule", {}).get("conditions", []),
        interventions=[i.get("name", "") for i in arms.get("interventions", [])],
        enrollment=design.get("enrollmentInfo", {}).get("count"),
        start_date=status.get("startDateStruct", {}).get("date"),
    )
