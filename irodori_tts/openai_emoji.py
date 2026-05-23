from __future__ import annotations

import os

from .gradio_emoji_palette import EMOJI_PALETTE_ITEMS


DEFAULT_OPENAI_EMOJI_MODEL = "gpt-4.1-mini"


def add_speech_emojis_with_openai(text: str, *, model: str | None = None) -> str:
    text_value = str(text).strip()
    if text_value == "":
        raise ValueError("text is required.")
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise ValueError("OPENAI_API_KEY is not set.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is required. Install it with `pip install openai`.") from exc

    allowed = "\n".join(
        f"- {item.emoji}: {item.label} ({item.description})" for item in EMOJI_PALETTE_ITEMS
    )
    instructions = (
        "あなたは日本語TTS用の演出タグ付けアシスタントです。"
        "入力文の意味、感情、息遣い、読み方に合う絵文字を自然に追加してください。"
        "使ってよい絵文字は許可リストだけです。文章の意味は変えず、文や改行の順番も保ってください。"
        "絵文字は必要な箇所だけに控えめに入れてください。説明、引用符、Markdown、前置きは出力しないでください。"
    )
    prompt = (
        "許可リスト:\n"
        f"{allowed}\n\n"
        "入力文:\n"
        f"{text_value}\n\n"
        "絵文字を追加した本文だけを返してください。"
    )

    client = OpenAI()
    response = client.responses.create(
        model=(model or os.environ.get("OPENAI_EMOJI_MODEL") or DEFAULT_OPENAI_EMOJI_MODEL),
        instructions=instructions,
        input=prompt,
        max_output_tokens=max(256, min(4096, len(text_value) * 3)),
    )
    output_text = getattr(response, "output_text", None)
    if output_text is None:
        output_text = _extract_response_text(response)
    output_text = str(output_text).strip()
    if output_text == "":
        raise RuntimeError("OpenAI returned an empty response.")
    return output_text


def _extract_response_text(response: object) -> str:
    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text is not None:
                parts.append(str(text))
    return "".join(parts)
