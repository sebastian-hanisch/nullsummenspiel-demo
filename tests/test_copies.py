"""Die kopierte Datei nsm_rng.py (wörtlich, Quelle im Kopf der Datei): Vergleich mit dem Original, solange es im Portfolio-Ordner liegt."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SOURCES = {"nsm_rng.py": ("pearson-demo", "prs_rng.py")}


@pytest.mark.parametrize("copy", list(SOURCES))
def test_copy_is_verbatim_apart_from_the_header(copy):
    folder, original = SOURCES[copy]
    source = ROOT.parent / folder / original
    if not source.exists():
        pytest.skip("Original liegt nicht im Portfolio-Ordner")
    mine = (ROOT / copy).read_text(encoding="utf-8").splitlines()
    assert mine[0].startswith(f"# Wörtliche Kopie von {folder}/{original}")
    assert mine[1:] == source.read_text(encoding="utf-8").splitlines(), "die Kopie weicht vom Original ab: Änderungen gehören ins Original, danach neu kopieren"


def test_the_copy_runs_on_its_own():
    from nsm_rng import SplitMix64
    assert [SplitMix64(0).next() for _ in range(1)] == [16294208416658607535]
    assert SplitMix64(1).below(10) in range(10)
