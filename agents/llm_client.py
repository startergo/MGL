"""
LLM client factory for the MGL agents. Default provider: Z.AI GLM (Anthropic-compatible API).

All settings come from the environment (never from files in this folder):
  AGENT_PROVIDER          "glm" (default) or "anthropic"
  ANTHROPIC_AUTH_TOKEN    Z.AI API key for GLM      (ANTHROPIC_API_KEY also accepted)
  ANTHROPIC_API_KEY       Anthropic key when AGENT_PROVIDER=anthropic
  ANTHROPIC_BASE_URL      API base URL; defaults to the public Z.AI endpoint for GLM,
                          SDK default for anthropic
  AGENT_MODEL             model id override (default glm-5.3 / claude-opus-5-5)
  AGENT_REASONING_EFFORT  GLM reasoning effort: high (default) or max
"""

import os

PROVIDER_GLM = "glm"
PROVIDER_ANTHROPIC = "anthropic"

DEFAULT_MODEL_GLM = os.environ.get("AGENT_DEFAULT_GLM_MODEL", "glm-5.3")
DEFAULT_MODEL_ANTHROPIC = os.environ.get("AGENT_DEFAULT_MODEL", "claude-opus-5-5")
DEFAULT_GLM_BASE_URL = "https://api.z.ai/api/anthropic"   # public endpoint, not a secret
DEFAULT_REASONING_EFFORT = "high"


def get_provider() -> str:
    p = os.environ.get("AGENT_PROVIDER", PROVIDER_GLM).strip().lower()
    if p not in (PROVIDER_GLM, PROVIDER_ANTHROPIC):
        raise SystemExit(f"AGENT_PROVIDER must be '{PROVIDER_GLM}' or '{PROVIDER_ANTHROPIC}', got '{p}'")
    return p


def glm_keys():
    """Z.AI keys in rotation order: ANTHROPIC_AUTH_TOKEN, then ANTHROPIC_AUTH_TOKEN_2 .. _9."""
    names = ["ANTHROPIC_AUTH_TOKEN"] + [f"ANTHROPIC_AUTH_TOKEN_{i}" for i in range(2, 10)]
    keys = []
    for n in names:
        v = (os.environ.get(n) or "").strip()
        if v and v not in keys:
            keys.append(v)
    return keys


def get_client(key_index: int = 0):
    from anthropic import Anthropic  # imported lazily so --help / --no-llm work without the SDK
    provider = get_provider()
    keys = glm_keys()
    token = keys[key_index] if key_index < len(keys) else os.environ.get("ANTHROPIC_AUTH_TOKEN")
    key = os.environ.get("ANTHROPIC_API_KEY")
    if provider == PROVIDER_GLM:
        api_key = token or key
        base_url = os.environ.get("ANTHROPIC_BASE_URL") or DEFAULT_GLM_BASE_URL
        if not api_key:
            raise SystemExit("Set ANTHROPIC_AUTH_TOKEN to your Z.AI API key "
                             "(or AGENT_PROVIDER=anthropic with ANTHROPIC_API_KEY).")
        return Anthropic(api_key=api_key, base_url=base_url)
    api_key = key or token
    if not api_key:
        raise SystemExit("Set ANTHROPIC_API_KEY for AGENT_PROVIDER=anthropic.")
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    return Anthropic(api_key=api_key, base_url=base_url) if base_url else Anthropic(api_key=api_key)


def get_model() -> str:
    override = os.environ.get("AGENT_MODEL")
    if override:
        return override
    return DEFAULT_MODEL_GLM if get_provider() == PROVIDER_GLM else DEFAULT_MODEL_ANTHROPIC


def _is_glm(model: str) -> bool:
    return model.lower().startswith("glm")


def generation_kwargs(max_tokens: int) -> dict:
    """GLM: thinking on with reasoning effort (Z.AI fields via extra_body); reasoning
    tokens count toward max_tokens, so give headroom. Other models: just max_tokens."""
    if not _is_glm(get_model()):
        return {"max_tokens": max_tokens}
    effort = os.environ.get("AGENT_REASONING_EFFORT", DEFAULT_REASONING_EFFORT)
    return {
        "max_tokens": max(max_tokens, 16384),
        "extra_body": {"thinking": {"type": "enabled"}, "reasoning_effort": effort},
    }


def response_text(response) -> str:
    """Join text blocks (thinking blocks are skipped)."""
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
