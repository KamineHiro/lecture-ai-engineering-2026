"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import sys

from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str, temperature: float | None = None) -> str:
    client = genai.Client(api_key=get_api_key())
    if temperature is None:
        response = client.models.generate_content(model=MODEL, contents=question)
    else:
        config = types.GenerateContentConfig(temperature=temperature)
        response = client.models.generate_content(
            model=MODEL, contents=question, config=config
        )

    usage = response.usage_metadata
    input_tokens = getattr(usage, "prompt_token_count", None)
    output_tokens = getattr(usage, "candidates_token_count", None)
    print(
        f"[トークン数] 入力: {input_tokens if input_tokens is not None else '不明'}, "
        f"出力: {output_tokens if output_tokens is not None else '不明'}"
    )

    return response.text


def _parse_temperature(value: str) -> float:
    try:
        temperature = float(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError("temperatureは数値で指定してください") from e
    if not 0 <= temperature <= 2:
        raise argparse.ArgumentTypeError("temperatureは0.0から2.0の範囲で指定してください")
    return temperature


def _prompt_temperature() -> float | None:
    while True:
        value = input("temperature (0.0-2.0、Enterでモデル既定値): ").strip()
        if not value:
            return None
        try:
            return _parse_temperature(value)
        except argparse.ArgumentTypeError as e:
            print(e)


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?")
    parser.add_argument(
        "--temperature",
        type=_parse_temperature,
        help="生成時のtemperature（0.0〜2.0）",
    )
    args = parser.parse_args()

    question = args.question
    temperature = args.temperature
    if question is None:
        if sys.stdin.isatty():
            question = input(f"質問（Enterで既定値: {DEFAULT_QUESTION}）: ").strip()
            question = question or DEFAULT_QUESTION
            if temperature is None:
                temperature = _prompt_temperature()
        else:
            question = DEFAULT_QUESTION

    print(f"[model] {MODEL}")
    print(f"[temperature] {temperature if temperature is not None else 'モデル既定値'}")
    print(f"[質問] {question}\n")
    try:
        print(ask(question, temperature))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
