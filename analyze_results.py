import json

with open("results/experiments.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("="*60)
print("EXPERIMENT 1: ZERO-SHOT VS FEW-SHOT")
print("="*60)
for item in data["experiment_1_zero_vs_few_shot"]:
    print(f"ID: {item['test_id']}")
    print(f"  Input: {item['input_text'][:70]}...")
    print(f"  Expected:  {item['expected']}")
    print(f"  Zero-shot: {item['zero_shot']['raw_output']} (latency: {item['zero_shot']['latency_sec']}s)")
    print(f"  Few-shot:  {item['few_shot']['raw_output']} (latency: {item['few_shot']['latency_sec']}s)")
    print()

print("="*60)
print("EXPERIMENT 2: DIRECT VS CHAIN-OF-THOUGHT")
print("="*60)
for item in data["experiment_2_cot"]:
    print(f"Problem: {item['problem_id']} - {item['title']}")
    print(f"  Ground Truth: {item['ground_truth']}")
    print(f"  Direct Output Preview:\n{item['direct']['raw_output'][:200]}...\n  [Length: {len(item['direct']['raw_output'])}, Latency: {item['direct']['latency_sec']}s]")
    print(f"  CoT Output Preview:\n{item['chain_of_thought']['raw_output'][:200]}...\n  [Length: {len(item['chain_of_thought']['raw_output'])}, Latency: {item['chain_of_thought']['latency_sec']}s]")
    print()

print("="*60)
print("EXPERIMENT 3: STRUCTURED OUTPUT STRATEGIES")
print("="*60)
strat_a = data["experiment_3_structured_output"]["strategy_a_naive"]
strat_b = data["experiment_3_structured_output"]["strategy_b_explicit_schema"]
strat_c = data["experiment_3_structured_output"]["strategy_c_api_json_mode"]

def summarize_strategy(name, items):
    valid_json_count = sum(1 for x in items if x["validation"]["is_valid_json"])
    conformant_count = sum(1 for x in items if x["validation"].get("schema_conformant", False))
    markdown_count = sum(1 for x in items if x["validation"].get("had_markdown_fences", False))
    avg_latency = sum(x["latency_sec"] for x in items) / len(items)
    print(f"--- {name} ---")
    print(f"  Total Inputs: {len(items)}")
    print(f"  Valid JSON: {valid_json_count}/{len(items)} ({valid_json_count/len(items)*100:.1f}%)")
    print(f"  Schema Conformant: {conformant_count}/{len(items)} ({conformant_count/len(items)*100:.1f}%)")
    print(f"  Had Markdown Fences: {markdown_count}/{len(items)} ({markdown_count/len(items)*100:.1f}%)")
    print(f"  Avg Latency: {avg_latency:.2f}s")
    for x in items:
        missing = x["validation"].get("missing_keys", [])
        if missing:
            print(f"    [{x['receipt_id']}] Missing Keys: {missing}")

summarize_strategy("Strategy A (Naive Natural Language)", strat_a)
summarize_strategy("Strategy B (Explicit Schema in Prompt)", strat_b)
summarize_strategy("Strategy C (API JSON Mode)", strat_c)
