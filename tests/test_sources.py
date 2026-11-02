from __future__ import annotations

import pytest

from bioscout.cache import ResponseCache
from bioscout.config import CacheMode, Settings
from bioscout.http import CachedHttp, CacheMissError
from bioscout.sources.clinicaltrials import CTGOV, ClinicalTrials
from bioscout.sources.europepmc import EPMC, EuropePMC, parse_jats_sections
from tests.conftest import fixture, ok


async def test_fulltext_resolves_pmid_and_reads_top_level_sections(http, api):
    api.get(f"{EPMC}/search").mock(return_value=ok(fixture("epmc_search_oa.json")))
    api.get(f"{EPMC}/PMC9000002/fullTextXML").mock(return_value=ok(fixture("epmc_fulltext.xml")))

    ft = await EuropePMC(http).fulltext(pmid="10000002")

    assert ft.available and ft.pmcid == "PMC9000002"
    assert [s.title for s in ft.sections] == ["Introduction", "Methods", "Results"]
    assert "PD-1 blockade" in ft.sections[0].text
    assert "Fisher exact test" in ft.sections[1].text  # nested sec folded into its parent


async def test_fulltext_explains_when_not_open_access(http, api):
    api.get(f"{EPMC}/search").mock(return_value=ok(fixture("epmc_search_closed.json")))
    ft = await EuropePMC(http).fulltext(pmid="10000001")
    assert not ft.available
    assert "abstract" in ft.reason


def test_jats_budget_truncates():
    sections, truncated = parse_jats_sections(fixture("epmc_fulltext.xml"), max_chars=60)
    assert truncated
    assert sum(len(s.text) for s in sections) <= 62


async def test_trials_parse_and_status_filter(http, api):
    route = api.get(f"{CTGOV}/studies").mock(return_value=ok(fixture("ctgov_studies.json")))
    result = await ClinicalTrials(http).search(condition="colorectal cancer", status=["recruiting", "bogus"])

    trial = result.trials[0]
    assert trial.nct_id == "NCT90000001"
    assert trial.phases == ["PHASE2"]
    assert trial.interventions == ["Synthetic-mab"]
    assert trial.enrollment == 120
    assert route.calls[0].request.url.params["filter.overallStatus"] == "RECRUITING"


async def test_readonly_mode_never_touches_network(tmp_path, api):
    settings = Settings(cache_dir=tmp_path, cache_mode=CacheMode.READONLY)
    http = CachedHttp(settings, cache=ResponseCache(tmp_path / "r.sqlite"))
    with pytest.raises(CacheMissError):
        await ClinicalTrials(http).search(query="anything")
    assert api.calls.call_count == 0
    await http.aclose()
