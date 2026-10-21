# Name That LLM - own extra-data generation via the Kaggle Benchmarks Model Proxy (free quota).
# Host-matched request: one user message, no system prompt, provider defaults:
#   "<prompt> Write a single paragraph of about N words, with no title or headings."
# Prompts: seyitkaangunes/ntl-extra-api-passages prompt bank (CC BY 4.0), ids not used by other public proxy datasets.
# N: drawn per (prompt, model) from the competition train word-count distribution (numbers only), clipped 40-250.
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import kaggle_benchmarks as kbench

BANK = __BANK__
COST_CAP_USD = __CAP__   # per model run (float, or {slug: usd}); stop early if the free quota is costlier than expected


@kbench.task(name="__TASK__", description="Generate single-paragraph passages for authorship-attribution research.")
def generate(llm) -> int:
    model = str(getattr(llm, 'model', None) or getattr(llm, 'name', None) or 'unknown')
    norm = model.split('/')[-1].replace('@', '-')            # e.g. anthropic/claude-opus-4-8@default -> claude-opus-4-8-default
    slug = next((m for m in BANK['models'] if m == norm), None) or next((m for m in BANK['models'] if m in norm), None)
    out = Path('/kaggle/working') / ('passages_' + model.replace('/', '_') + '.jsonl')
    spent, ok, errors = 0.0, 0, 0
    done = set(BANK.get('done', {}).get(slug or '', []))
    for r in BANK['records']:
        if r['prompt_id'] in done:   # this model already answered this prompt in an earlier run
            continue
        if slug is None:
            print('model not in target list:', model, flush=True)
            return 0
        n = r['n'][slug]
        msg = f"{r['prompt']} Write a single paragraph of about {n} words, with no title or headings."
        rec = {'prompt_id': r['prompt_id'], 'model_slug': slug, 'model_seen': model, 'target_words': n,
               'user_message': msg, 'timestamp': datetime.now(timezone.utc).isoformat()}
        t0 = time.monotonic()
        try:
            with kbench.chats.new('p_' + r['prompt_id']) as chat:
                text = llm.prompt(msg)
                usage = chat.usage
            cost = ((getattr(usage, 'input_tokens_cost_nanodollars', 0) or 0) +
                    (getattr(usage, 'output_tokens_cost_nanodollars', 0) or 0)) / 1e9
            rec.update(text=text, status='ok' if isinstance(text, str) and text.strip() else 'empty',
                       input_tokens=getattr(usage, 'input_tokens', None), output_tokens=getattr(usage, 'output_tokens', None),
                       cost_usd=cost, seconds=round(time.monotonic() - t0, 2))
            spent += cost
            ok += rec['status'] == 'ok'
        except Exception as exc:   # never log exception text (may contain request internals)
            rec.update(status='error', error_type=type(exc).__name__)
            errors += 1
            if type(exc).__name__ == 'PermissionDeniedError' and errors >= 3:   # provider's daily free quota is used up
                with out.open('a') as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
                print(json.dumps({'quota_exhausted': True, 'model': model}), flush=True)
                break
        with out.open('a') as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
        cap = COST_CAP_USD.get(slug, 0.5) if isinstance(COST_CAP_USD, dict) else COST_CAP_USD
        if spent >= cap:
            print(json.dumps({'cost_cap_stop': True, 'spent_usd': spent}), flush=True)
            break
    print(json.dumps({'model': model, 'ok': ok, 'spent_usd': round(spent, 4)}), flush=True)
    return ok


generate.run(kbench.llm)
