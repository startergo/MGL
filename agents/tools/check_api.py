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

print(f"provider : {llm_client.get_provider()}")
print(f"model    : {llm_client.get_model()}")
print(f"base url : {os.environ.get('ANTHROPIC_BASE_URL') or (llm_client.DEFAULT_GLM_BASE_URL if llm_client.get_provider() == 'glm' else 'SDK default')}")
keys = llm_client.glm_keys() if llm_client.get_provider() == "glm" else [os.environ.get("ANTHROPIC_API_KEY") or ""]
keys = [k for k in keys if k and not k.startswith("<")]
if not keys:
    print("key      : MISSING (edit agents/.env)"); sys.exit(1)
failed = 0
for i, key in enumerate(keys):
    label = "ANTHROPIC_AUTH_TOKEN" + ("" if i == 0 else f"_{i + 1}")
    print(f"key #{i + 1}   : {label} set ({len(key)} chars, ends …{key[-4:]})")
    try:
        r = llm_client.get_client(i).messages.create(model=llm_client.get_model(), max_tokens=16,
                                                     messages=[{"role": "user", "content": "Reply with: ok"}])
        print(f"  result : OK — {llm_client.response_text(r).strip()[:40]!r} "
              f"({r.usage.input_tokens} in / {r.usage.output_tokens} out tokens)")
    except SystemExit as e:
        print(f"  result : FAILED — {e}"); failed += 1
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        hint = ""
        if "1308" in msg:
            hint = "  → key is valid; its usage window is exhausted (resets at the time shown)"
        elif "1113" in msg or "Insufficient balance" in msg:
            hint = "  → key is valid but the Z.AI account has no balance / resource package"
        elif "401" in msg or "authentication" in msg.lower() or "invalid" in msg.lower():
            hint = f"  → key rejected: check {label} in agents/.env"
        print(f"  result : FAILED — {type(e).__name__}: {msg[:160]}\n{hint}"); failed += 1
sys.exit(1 if failed == len(keys) else 0)
