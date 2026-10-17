#!/usr/bin/env python3
"""Check that the configured LLM provider/key works (loads agents/.env; never prints the key).

    python3 agents/tools/check_api.py
Exit 0 = OK, 1 = failure (reason printed)."""
import os
import sys
from pathlib import Path

AGENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENTS))
env = AGENTS / ".env"
if env.exists():
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

import llm_client  # noqa: E402

key = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY") or ""
print(f"provider : {llm_client.get_provider()}")
print(f"model    : {llm_client.get_model()}")
print(f"base url : {os.environ.get('ANTHROPIC_BASE_URL') or (llm_client.DEFAULT_GLM_BASE_URL if llm_client.get_provider() == 'glm' else 'SDK default')}")
print(f"key      : {'set (' + str(len(key)) + ' chars, ends …' + key[-4:] + ')' if key and not key.startswith('<') else 'MISSING (edit agents/.env)'}")
try:
    r = llm_client.get_client().messages.create(model=llm_client.get_model(), max_tokens=16,
                                                messages=[{"role": "user", "content": "Reply with: ok"}])
    print(f"result   : OK — {llm_client.response_text(r).strip()[:40]!r} "
          f"({r.usage.input_tokens} in / {r.usage.output_tokens} out tokens)")
except SystemExit as e:
    print(f"result   : FAILED — {e}"); sys.exit(1)
except Exception as e:  # noqa: BLE001
    msg = str(e)
    hint = ""
    if "1113" in msg or "Insufficient balance" in msg:
        hint = "  → key is valid but the Z.AI account has no balance / resource package"
    elif "401" in msg or "authentication" in msg.lower() or "invalid" in msg.lower():
        hint = "  → key rejected: check ANTHROPIC_AUTH_TOKEN in agents/.env"
    print(f"result   : FAILED — {type(e).__name__}: {msg[:160]}\n{hint}"); sys.exit(1)
