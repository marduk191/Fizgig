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
