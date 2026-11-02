from __future__ import annotations

import httpx

from bioscout.sources.pubmed import EUTILS, PubMed, normalize_pmids
from tests.conftest import fixture, ok


def mock_eutils(api):
    api.get(f"{EUTILS}/esearch.fcgi").mock(return_value=ok(fixture("esearch.json")))
    api.get(f"{EUTILS}/esummary.fcgi").mock(return_value=ok(fixture("esummary.json")))
    api.get(f"{EUTILS}/efetch.fcgi").mock(return_value=ok(fixture("efetch.xml")))


async def test_search_keeps_relevance_order_and_metadata(http, api):
    mock_eutils(api)
    result = await PubMed(http).search("microsatellite instability immunotherapy", max_results=2)

    assert result.total_found == 1342
    assert [a.pmid for a in result.articles] == ["10000002", "10000001"]
    first = result.articles[0]
    assert first.year == 2024
    assert first.first_author == "Kim J"
    assert first.doi == "10.0000/synth.a"
    assert first.pmcid == "PMC9000002"


async def test_search_sends_api_key_and_year_filter(http, api):
    mock_eutils(api)
    await PubMed(http).search("MSI", min_year=2020)

    params = api.calls[0].request.url.params
    assert params["api_key"] == "test-key"
    assert params["datetype"] == "pdat"
    assert params["mindate"] == "2020"


async def test_abstracts_parse_labels_inline_markup_and_mesh(http, api):
    mock_eutils(api)
    batch = await PubMed(http).abstracts(["10000002", "10000001"])

    a, b = batch.articles
    assert a.abstract[0].label == "BACKGROUND"
    assert a.abstract[0].text == "Mismatch repair deficiency produces microsatellite instability."
    assert "Microsatellite Instability" in a.mesh_terms
    assert a.pmcid == "PMC9000002"
    assert b.year == 2021  # from <MedlineDate>2021 Jan-Feb</MedlineDate>
    assert b.abstract[0].label is None
    assert batch.not_found == []


async def test_check_pmids_flags_unknown_and_malformed(http, api):
    mock_eutils(api)
    check = await PubMed(http).check_pmids(["10000002", "PMID: 10000001", "99999999", "not-a-pmid"])

    assert check.existing == ["10000002", "10000001"]
    assert check.missing == ["99999999", "not-a-pmid"]


async def test_second_identical_call_is_served_from_cache(http, api):
    mock_eutils(api)
    pubmed = PubMed(http)
    await pubmed.abstracts(["10000002"])
    await pubmed.abstracts(["10000002"])

    assert api.calls.call_count == 1
    assert http.stats == {"hits": 1, "misses": 1}


async def test_retries_on_429_then_succeeds(http, api, monkeypatch):
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    api.get(f"{EUTILS}/efetch.fcgi").mock(side_effect=[httpx.Response(429), ok(fixture("efetch.xml"))])
    batch = await PubMed(http).abstracts(["10000002"])
    assert batch.articles


def test_normalize_pmids_dedupes_and_strips():
    assert normalize_pmids([" 123", "PMID:123", "456", "abc"]) == ["123", "456"]


async def _no_sleep(_s: float) -> None:
    return None
