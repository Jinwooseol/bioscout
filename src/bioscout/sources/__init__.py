"""External biomedical data sources. Each takes a shared CachedHttp."""

from bioscout.sources.clinicaltrials import ClinicalTrials
from bioscout.sources.europepmc import EuropePMC
from bioscout.sources.pubmed import PubMed

__all__ = ["PubMed", "EuropePMC", "ClinicalTrials"]
