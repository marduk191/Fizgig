"""Every model family's description is consistent: it validates, its files and its abilities agree."""
import pytest

from fizgig.families import registry

FAMILIES = sorted(registry.FAMILIES)


@pytest.mark.parametrize("key", FAMILIES)
def test_description_validates(key):
    assert registry.get(key).validate() == []


@pytest.mark.parametrize("key", FAMILIES)
def test_model_files_have_unique_prefs(key):
    keys = [m.pref_key for m in registry.get(key).model_files]
    assert len(keys) == len(set(keys)), f"{key}: a Preferences key is used twice"


@pytest.mark.parametrize("key", FAMILIES)
def test_preview_speed_lora_has_a_file_row(key):
    d = registry.get(key)
    sp = d.preview_speed()
    if sp is not None and sp.pref_key:
        assert sp.pref_key in {m.pref_key for m in d.model_files}, f"{key}: {sp.name} has no Preferences row"


@pytest.mark.parametrize("key", FAMILIES)
def test_presets_are_named_and_distinct(key):
    names = [n for n, _ in registry.get(key).presets]
    assert len(names) == len(set(names)), f"{key}: two presets share a name"


@pytest.mark.parametrize("key", FAMILIES)
def test_text_encoder_builds_are_complete(key):
    """Every listed text encoder build has its file, a full SHA-256, a size and a measured cosine, and none repeats
    the family's own file (docs/drivers/README.md, "Text encoder builds")."""
    import re
    d = registry.get(key)
    own = {(m.repo, m.path) for m in d.model_files if m.role == "text_encoder"}
    seen = set()
    for b in d.text_encoder_builds:
        assert b.label and b.repo and b.path.endswith(".safetensors"), b
        assert re.fullmatch(r"[0-9a-f]{64}", b.sha256), (b.label, b.sha256)
        assert b.size_gb > 0 and 0 < b.cosine <= 1, b
        assert (b.repo, b.path) not in own and b.sha256 not in seen, b
        seen.add(b.sha256)
