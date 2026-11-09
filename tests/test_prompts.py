import importlib

import pytest


@pytest.fixture()
def prompts(tmp_path, monkeypatch):
    monkeypatch.setenv("PETZA_DATA", str(tmp_path))
    import petza.config as config

    importlib.reload(config)
    import petza.prompts as prompts

    return importlib.reload(prompts)


def test_default_is_active_and_hebrew(prompts):
    store = prompts.load()
    assert store["active"] == prompts.DEFAULT_NAME
    assert "גרנולציה" in store["prompts"][prompts.DEFAULT_NAME]
    assert "גרנולציה" in prompts.active_rules().text


def test_save_select_delete(prompts):
    prompts.put("mine", "---\nlabels: [wound, slough]\n---\nTrace the edge.")
    assert prompts.load()["active"] == "mine"
    rules = prompts.active_rules()
    assert rules.labels == ["wound", "slough"] and rules.text == "Trace the edge."
    prompts.select(prompts.DEFAULT_NAME)
    assert prompts.load()["active"] == prompts.DEFAULT_NAME
    prompts.select("mine")
    prompts.delete("mine")
    store = prompts.load()
    assert "mine" not in store["prompts"] and store["active"] == prompts.DEFAULT_NAME
    with pytest.raises(ValueError):
        prompts.delete(prompts.DEFAULT_NAME)
    with pytest.raises(KeyError):
        prompts.select("missing")


def test_edited_default_survives_reload(prompts):
    prompts.put(prompts.DEFAULT_NAME, "edited")
    assert prompts.load()["prompts"][prompts.DEFAULT_NAME] == "edited"
