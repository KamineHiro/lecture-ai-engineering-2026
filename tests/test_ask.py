from types import SimpleNamespace

from google.genai import types

from app import ask as ask_module


def test_ask_passes_temperature_to_generate_content(monkeypatch):
    captured = {}
    response = SimpleNamespace(usage_metadata=None, text="answer")

    class Models:
        def generate_content(self, **kwargs):
            captured.update(kwargs)
            return response

    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(
        ask_module.genai,
        "Client",
        lambda **kwargs: SimpleNamespace(models=Models()),
    )

    assert ask_module.ask("question", 0.7) == "answer"
    assert captured["model"] == ask_module.MODEL
    assert captured["contents"] == "question"
    assert isinstance(captured["config"], types.GenerateContentConfig)
    assert captured["config"].temperature == 0.7


def test_main_prompts_for_question_and_temperature(monkeypatch, capsys):
    answers = iter(["試したい質問", "1.2"])
    captured = {}

    monkeypatch.setattr(ask_module.sys, "argv", ["app.ask"])
    monkeypatch.setattr(
        ask_module.sys, "stdin", SimpleNamespace(isatty=lambda: True)
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    monkeypatch.setattr(
        ask_module,
        "ask",
        lambda question, temperature: captured.update(
            question=question, temperature=temperature
        )
        or "answer",
    )

    ask_module.main()

    assert captured == {"question": "試したい質問", "temperature": 1.2}
    assert "[temperature] 1.2" in capsys.readouterr().out