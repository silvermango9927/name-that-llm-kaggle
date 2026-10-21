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

BANK = {'models': ['gpt-5.5-2026-04-23', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra', 'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.8-flash', 'claude-opus-4-6-default', 'claude-opus-4-8-default', 'claude-opus-5-default', 'claude-sonnet-5-default'], 'records': [{'prompt_id': 'g0541', 'prompt': 'Describe what it feels like to run across a pool of oobleck, a cornstarch-and-water mixture that firms up under each pounding step but swallows you if you stand still.', 'n': {'gpt-5.5-2026-04-23': 171, 'gpt-5.6-luna': 53, 'gpt-5.6-sol': 71, 'gpt-5.6-terra': 112, 'gpt-6-astra': 123, 'gemini-3.1-pro-preview': 87, 'gemini-3.6-flash': 239, 'gemini-3.8-flash': 48, 'claude-opus-4-6-default': 81, 'claude-opus-4-8-default': 85, 'claude-opus-5-default': 244, 'claude-sonnet-5-default': 66}}, {'prompt_id': 'g0542', 'prompt': 'Explain why a dry stick of spaghetti bent from both ends almost never breaks into just two pieces, a puzzle that once intrigued Richard Feynman, and how twisting it first finally makes it snap cleanly in two.', 'n': {'gpt-5.5-2026-04-23': 156, 'gpt-5.6-luna': 152, 'gpt-5.6-sol': 100, 'gpt-5.6-terra': 105, 'gpt-6-astra': 43, 'gemini-3.1-pro-preview': 50, 'gemini-3.6-flash': 72, 'gemini-3.8-flash': 63, 'claude-opus-4-6-default': 119, 'claude-opus-4-8-default': 108, 'claude-opus-5-default': 55, 'claude-sonnet-5-default': 119}}, {'prompt_id': 'g0543', 'prompt': "Describe the moment in Seoul in 2016 when AlphaGo played its now-famous 'Move 37' against Lee Sedol, capturing how commentators first took it for a blunder before realizing they had witnessed a move no human would have dared to play.", 'n': {'gpt-5.5-2026-04-23': 68, 'gpt-5.6-luna': 77, 'gpt-5.6-sol': 46, 'gpt-5.6-terra': 55, 'gpt-6-astra': 94, 'gemini-3.1-pro-preview': 213, 'gemini-3.6-flash': 162, 'gemini-3.8-flash': 145, 'claude-opus-4-6-default': 181, 'claude-opus-4-8-default': 222, 'claude-opus-5-default': 211, 'claude-sonnet-5-default': 140}}, {'prompt_id': 'g0544', 'prompt': 'Write a lyrical description of the Pythagorean idea of the music of the spheres, in which the planets hum in mathematical harmonies so constant that human ears have never noticed them.', 'n': {'gpt-5.5-2026-04-23': 53, 'gpt-5.6-luna': 156, 'gpt-5.6-sol': 99, 'gpt-5.6-terra': 61, 'gpt-6-astra': 153, 'gemini-3.1-pro-preview': 68, 'gemini-3.6-flash': 233, 'gemini-3.8-flash': 192, 'claude-opus-4-6-default': 45, 'claude-opus-4-8-default': 193, 'claude-opus-5-default': 104, 'claude-sonnet-5-default': 67}}, {'prompt_id': 'g0545', 'prompt': 'Write a vivid reconstruction of the last day of Ötzi the Iceman, the 5,300-year-old man found in an Alpine glacier, drawing on clues from his final meal, his copper axe, his unfinished arrows and the arrowhead lodged in his shoulder.', 'n': {'gpt-5.5-2026-04-23': 119, 'gpt-5.6-luna': 197, 'gpt-5.6-sol': 67, 'gpt-5.6-terra': 193, 'gpt-6-astra': 194, 'gemini-3.1-pro-preview': 78, 'gemini-3.6-flash': 233, 'gemini-3.8-flash': 77, 'claude-opus-4-6-default': 44, 'claude-opus-4-8-default': 124, 'claude-opus-5-default': 213, 'claude-sonnet-5-default': 79}}, {'prompt_id': 'g0546', 'prompt': 'Explain how a siphon lifts liquid up and over an obstacle, and why physicists still argue over whether gravity or atmospheric pressure deserves the credit.', 'n': {'gpt-5.5-2026-04-23': 192, 'gpt-5.6-luna': 43, 'gpt-5.6-sol': 45, 'gpt-5.6-terra': 110, 'gpt-6-astra': 160, 'gemini-3.1-pro-preview': 49, 'gemini-3.6-flash': 88, 'gemini-3.8-flash': 250, 'claude-opus-4-6-default': 108, 'claude-opus-4-8-default': 90, 'claude-opus-5-default': 42, 'claude-sonnet-5-default': 179}}, {'prompt_id': 'g0547', 'prompt': 'Write a brief story in which a hiker in a remote Australian canyon stumbles upon a grove of trees that were thought to have vanished alongside the dinosaurs.', 'n': {'gpt-5.5-2026-04-23': 170, 'gpt-5.6-luna': 105, 'gpt-5.6-sol': 226, 'gpt-5.6-terra': 78, 'gpt-6-astra': 40, 'gemini-3.1-pro-preview': 113, 'gemini-3.6-flash': 94, 'gemini-3.8-flash': 105, 'claude-opus-4-6-default': 79, 'claude-opus-4-8-default': 146, 'claude-opus-5-default': 109, 'claude-sonnet-5-default': 81}}, {'prompt_id': 'g0548', 'prompt': 'Explain the hydraulic jump, the circular ring that forms where water from a tap spreads across the bottom of a sink and suddenly thickens, and how it relates to the tidal bores that surge up great rivers.', 'n': {'gpt-5.5-2026-04-23': 75, 'gpt-5.6-luna': 56, 'gpt-5.6-sol': 45, 'gpt-5.6-terra': 155, 'gpt-6-astra': 75, 'gemini-3.1-pro-preview': 77, 'gemini-3.6-flash': 112, 'gemini-3.8-flash': 96, 'claude-opus-4-6-default': 125, 'claude-opus-4-8-default': 59, 'claude-opus-5-default': 76, 'claude-sonnet-5-default': 59}}, {'prompt_id': 'g0549', 'prompt': "Explain Cantor's diagonal argument and why it proves there are more real numbers than whole numbers, walking the reader through the trick of building a number that differs from every entry on any supposedly complete list.", 'n': {'gpt-5.5-2026-04-23': 107, 'gpt-5.6-luna': 150, 'gpt-5.6-sol': 73, 'gpt-5.6-terra': 84, 'gpt-6-astra': 45, 'gemini-3.1-pro-preview': 98, 'gemini-3.6-flash': 215, 'gemini-3.8-flash': 95, 'claude-opus-4-6-default': 76, 'claude-opus-4-8-default': 150, 'claude-opus-5-default': 59, 'claude-sonnet-5-default': 160}}, {'prompt_id': 'g0550', 'prompt': 'Write a passage explaining why ancient Roman concrete has survived two thousand years in harbors and domes while modern concrete cracks within decades, including the discovery of lime clasts that let it heal its own cracks.', 'n': {'gpt-5.5-2026-04-23': 48, 'gpt-5.6-luna': 67, 'gpt-5.6-sol': 133, 'gpt-5.6-terra': 186, 'gpt-6-astra': 64, 'gemini-3.1-pro-preview': 49, 'gemini-3.6-flash': 41, 'gemini-3.8-flash': 79, 'claude-opus-4-6-default': 157, 'claude-opus-4-8-default': 200, 'claude-opus-5-default': 132, 'claude-sonnet-5-default': 216}}], 'done': {'claude-opus-4-8-default': [], 'claude-sonnet-5-default': [], 'claude-opus-4-6-default': [], 'claude-opus-5-default': []}}
COST_CAP_USD = {'claude-opus-4-8-default': 0.5, 'claude-sonnet-5-default': 0.5, 'claude-opus-4-6-default': 0.5, 'claude-opus-5-default': 0.5}   # per model run (float, or {slug: usd}); stop early if the free quota is costlier than expected


@kbench.task(name="ntl-gen-202610190724", description="Generate single-paragraph passages for authorship-attribution research.")
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
