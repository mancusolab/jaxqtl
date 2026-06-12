# pattern: Imperative Shell

from pathlib import Path

import polars as pl
import pytest

from jaxqtl.io import GenoioData


REPO_ROOT = Path(__file__).resolve().parents[2]
GENO_PREFIX = REPO_ROOT / "tutorial" / "input" / "chr22_N100"


def _source_iids() -> list[str]:
    fam_path = GENO_PREFIX.with_suffix(".fam")
    return [line.split()[1] for line in fam_path.read_text().splitlines()]


def _sample_iids(sample_info: pl.DataFrame) -> list[str]:
    return sample_info.get_column("iid").to_list()


def test_sample_info_has_iid_and_preserves_source_order() -> None:
    data = GenoioData.load(str(GENO_PREFIX))

    assert "iid" in data.sample_info.columns
    assert _sample_iids(data.sample_info) == _source_iids()


def test_replace_individuals_freezes_reversed_iid_order() -> None:
    data = GenoioData.load(str(GENO_PREFIX))
    reversed_sample_info = data.sample_info.reverse()

    replaced = data.replace_individuals(reversed_sample_info)

    assert _sample_iids(replaced.sample_info) == list(reversed(_source_iids()))


def test_filter_individuals_matches_cli_keep_and_drop_expectations() -> None:
    data = GenoioData.load(str(GENO_PREFIX))
    source_iids = _source_iids()

    keep_iids = [source_iids[2], source_iids[0], source_iids[-1]]
    kept = data.filter_individuals(keep_iids, "keep")
    assert _sample_iids(kept.sample_info) == [source_iids[0], source_iids[2], source_iids[-1]]

    drop_iids = [source_iids[1], source_iids[3]]
    dropped = data.filter_individuals(drop_iids, "drop")
    assert _sample_iids(dropped.sample_info) == [iid for iid in source_iids if iid not in drop_iids]


def test_replace_individuals_rejects_duplicate_iids() -> None:
    data = GenoioData.load(str(GENO_PREFIX))
    duplicate_sample_info = data.sample_info.head(2).with_columns(pl.lit(_source_iids()[0]).alias("iid"))

    with pytest.raises(ValueError, match="duplicate IID"):
        data.replace_individuals(duplicate_sample_info)
