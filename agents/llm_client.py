"""
LLM client factory for the MGL agents.

All settings come from the environment (never from files in this folder):
  ANTHROPIC_API_KEY       Anthropic key          (or ANTHROPIC_AUTH_TOKEN for compatible gateways)
  ANTHROPIC_BASE_URL      optional API base URL override
  AGENT_MODEL             model id (default below, or GLM default when a Z.AI base URL is set)
  AGENT_REASONING_EFFORT  GLM reasoning effort: high (default) or max
  AGENT_ZAI_HOST_HINT     substring identifying a Z.AI-style base URL (default "z.ai")
"""

import os

DEFAULT_MODEL_ANTHROPIC = os.environ.get("AGENT_DEFAULT_MODEL", "claude-opus-5-5")
DEFAULT_MODEL_ZAI = os.environ.get("AGENT_DEFAULT_GLM_MODEL", "glm-5.3")
DEFAULT_REASONING_EFFORT = "high"


def get_client():
    from anthropic import Anthropic  # imported lazily so --help works without the SDK
    token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if token and not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = token
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN) in your environment.")
    return Anthropic()


def get_model() -> str:
    override = os.environ.get("AGENT_MODEL")
    if override:
        return override
    hint = os.environ.get("AGENT_ZAI_HOST_HINT", "z.ai")
    if hint and hint in os.environ.get("ANTHROPIC_BASE_URL", ""):
        return DEFAULT_MODEL_ZAI
    return DEFAULT_MODEL_ANTHROPIC


def _is_glm(model: str) -> bool:
    return model.lower().startswith("glm")


def generation_kwargs(max_tokens: int) -> dict:
    if not _is_glm(get_model()):
        return {"max_tokens": max_tokens}
    effort = os.environ.get("AGENT_REASONING_EFFORT", DEFAULT_REASONING_EFFORT)
    return {
        "max_tokens": max(max_tokens, 16384),
        "extra_body": {"thinking": {"type": "enabled"}, "reasoning_effort": effort},
    }


def response_text(response) -> str:
    return "".join(getattr(b, "text", "") for b in response.content
                   if getattr(b, "type", "text") == "text")


def ask_json(system: str, user: str, max_tokens: int = 2048) -> dict | None:
    """One request expecting a JSON object back; one retry asking for JSON only."""
    import json
    import re

    def parse(text: str):
        text = re.sub(r"^```[a-z]*\n?|\n?```$", "", text.strip())
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        return None

    client, model = get_client(), get_model()
    msgs = [{"role": "user", "content": user}]
    raw = response_text(client.messages.create(model=model, system=system, messages=msgs,
                                               **generation_kwargs(max_tokens)))
    result = parse(raw)
    if result is None:
        msgs += [{"role": "assistant", "content": raw},
                 {"role": "user", "content": "Return ONLY the JSON object. Start with { and end with }."}]
        result = parse(response_text(client.messages.create(model=model, system=system, messages=msgs,
                                                            **generation_kwargs(max_tokens))))
    return result
