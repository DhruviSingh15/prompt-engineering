"""
Prompt Engineering Rigorous Empirical Evaluation Runner
Model: gemini-3.5-flash-lite (via google-genai SDK)
Includes automated 15-RPM rate limiting, exponential backoff for 429,
and incremental persistence of actual raw model outputs.
"""

import os
import sys
import json
import time
from typing import Dict, Any, List

sys.stdout.reconfigure(encoding="utf-8")
from dotenv import load_dotenv

load_dotenv()
from google import genai
from google.genai import types
from google.genai.errors import ClientError, APIError

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_NAME = "gemini-3.5-flash-lite"
RESULTS_FILE = "results/experiments.json"

def call_with_retry(model: str, contents: Any, config: Any, max_retries: int = 5):
    """Executes generate_content with backoff on 429 Resource Exhausted."""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
            return response
        except ClientError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = 45 + (attempt * 15)
                print(f"    [Rate limit 429] Waiting {wait_time}s before retry (attempt {attempt + 1}/{max_retries})...", flush=True)
                time.sleep(wait_time)
            else:
                raise e
        except Exception as e:
            wait_time = 10 * (attempt + 1)
            print(f"    [API Error: {e}] Waiting {wait_time}s...", flush=True)
            time.sleep(wait_time)
    raise RuntimeError(f"Failed after {max_retries} retries.")

# ==============================================================================
# EXPERIMENT 1: ZERO-SHOT VS. FEW-SHOT CLASSIFICATION
# ==============================================================================

ZERO_SHOT_SYSTEM = (
    "You are an automated customer support triage engine. "
    "Classify the input text into: Category, Sentiment, Urgency, and Actionable.\n"
    "Allowed Categories: [Billing / Refund, Technical / Bug, Account Security, Feature Request, Cancellation]\n"
    "Allowed Sentiment: [Positive, Neutral, Negative, Frustrated]\n"
    "Allowed Urgency: [Low, Medium, High, Critical]\n"
    "Allowed Actionable: [YES, NO]\n"
    "Output format must be strictly a single line:\n"
    "Category: <category> | Sentiment: <sentiment> | Urgency: <urgency> | Actionable: <actionable>\n"
    "Do not include any greeting, preamble, reasoning, or markdown formatting."
)

FEW_SHOT_SYSTEM = (
    "You are an automated customer support triage engine. "
    "Classify the input text into: Category, Sentiment, Urgency, and Actionable.\n"
    "Allowed Categories: [Billing / Refund, Technical / Bug, Account Security, Feature Request, Cancellation]\n"
    "Allowed Sentiment: [Positive, Neutral, Negative, Frustrated]\n"
    "Allowed Urgency: [Low, Medium, High, Critical]\n"
    "Allowed Actionable: [YES, NO]\n"
    "Output format must be strictly a single line:\n"
    "Category: <category> | Sentiment: <sentiment> | Urgency: <urgency> | Actionable: <actionable>\n"
    "Do not include any greeting, preamble, reasoning, or markdown formatting.\n\n"
    "--- EXAMPLES ---\n"
    "Input: Wow, thank you SO much for locking my account 10 minutes before my investor demo because of a 'suspicious' login from my own laptop. Truly brilliant work.\n"
    "Output: Category: Account Security | Sentiment: Frustrated | Urgency: Critical | Actionable: YES\n\n"
    "Input: Could we get an option to export tables as CSV instead of just PDF? Would save us 15 minutes every Friday. Otherwise loving the platform!\n"
    "Output: Category: Feature Request | Sentiment: Positive | Urgency: Low | Actionable: NO\n\n"
    "Input: I was double billed for invoice #4029 on my corporate Mastercard. Please issue a credit to the card.\n"
    "Output: Category: Billing / Refund | Sentiment: Neutral | Urgency: High | Actionable: YES\n\n"
    "Input: I'm shutting down my agency so I will not be renewing our enterprise plan next month. Thanks for the support over the past two years.\n"
    "Output: Category: Cancellation | Sentiment: Neutral | Urgency: Medium | Actionable: YES\n"
    "--- END EXAMPLES ---"
)

EXP1_TEST_INPUTS = [
    {
        "id": "T1_Sarcasm",
        "text": "Oh wonderful, your latest update just wiped 4 months of client invoices right before my tax audit. Top notch engineering guys, absolutely stellar work!",
        "expected": "Category: Technical / Bug | Sentiment: Frustrated | Urgency: Critical | Actionable: YES"
    },
    {
        "id": "T2_Account_Security",
        "text": "Hi team, love the software! Just noticed two unauthorized password reset emails in Russian sent to my inbox at 3 AM. I haven't logged in from Europe. Could you verify my session security please?",
        "expected": "Category: Account Security | Sentiment: Neutral | Urgency: High | Actionable: YES"
    },
    {
        "id": "T3_Billing_Dispute",
        "text": "Why did my card get charged $89 when the signup modal explicitly promised $29/month? If this isn't reversed immediately I will initiate a chargeback with Chase.",
        "expected": "Category: Billing / Refund | Sentiment: Frustrated | Urgency: High | Actionable: YES"
    },
    {
        "id": "T4_Feature_Request",
        "text": "Would it be possible to add dark mode to the PDF export view? Current white background is a bit harsh on the eyes during late night review sessions, but otherwise everything is great!",
        "expected": "Category: Feature Request | Sentiment: Positive | Urgency: Low | Actionable: NO"
    },
    {
        "id": "T5_Cancellation",
        "text": "Please cancel my auto-renew subscription at the end of the current billing cycle. No issues with the product itself, just downsizing my consultancy.",
        "expected": "Category: Cancellation | Sentiment: Neutral | Urgency: Medium | Actionable: YES"
    },
    {
        "id": "T6_Minor_Glitch",
        "text": "Hey, is the analytics dashboard having a minor hiccup today? The graphs took about 10 seconds longer to render on Firefox than usual. Not blocking me, just thought I'd flag it.",
        "expected": "Category: Technical / Bug | Sentiment: Neutral | Urgency: Low | Actionable: NO"
    }
]

# ==============================================================================
# EXPERIMENT 2: DIRECT PROMPTING VS. CHAIN-OF-THOUGHT (CoT)
# ==============================================================================

EXP2_PROBLEMS = [
    {
        "id": "P1_Logistics_Math",
        "title": "Multi-Step Manufacturing Packaging & Downtime Problem",
        "problem": (
            "A manufacturing plant has three automated packaging lines: Line A, Line B, and Line C.\n"
            "- Line A operates for 8 hours at an unhindered rate of 150 units per hour. However, it requires a 20-minute sanitation shutdown after every 3 hours of continuous operation (the downtime starts at the 3-hour mark and at the 6-hour mark).\n"
            "- Line B operates for 10 hours. Its base speed is 20% faster than Line A, but after 6 hours of operation, a motor bearing overheats and drops its speed by 25% for the remaining 4 hours. Line B has no scheduled downtime.\n"
            "- Line C operates for 6 hours at a constant rate of 120 units per hour, but 5% of its produced units are defective and discarded.\n"
            "- Across all three lines, 2% of the final non-defective packaged units are damaged during pallet wrapping and cannot be shipped.\n"
            "What is the exact total number of undamaged units ready for shipping at the end of the shift?"
        ),
        "ground_truth": "3,335.92 (or 3,335-3,336 units)"
    },
    {
        "id": "P2_Deductive_Logic",
        "title": "5-Person 1-on-1 Performance Review Schedule Deductive Logic Puzzle",
        "problem": (
            "Five colleagues—Elena, Marcus, Priya, Tariq, and Chloe—are scheduled for 1-on-1 performance reviews between 9:00 AM and 2:00 PM on Monday. Each review lasts exactly 1 hour, starting on the hour (9:00 AM, 10:00 AM, 11:00 AM, 12:00 PM, 1:00 PM).\n"
            "Constraints:\n"
            "1. Elena's review is scheduled earlier than Priya's review, and Priya's review is scheduled earlier than Marcus's review.\n"
            "2. Tariq's review is scheduled for the last slot of the day (1:00 PM).\n"
            "3. Chloe's review is scheduled earlier than Priya's review, but she is not the first person of the day.\n"
            "4. The 11:00 AM slot is assigned to Priya.\n"
            "5. Marcus is scheduled immediately before Tariq.\n"
            "What is the exact chronological schedule (time slot and person) for all 5 reviews?"
        ),
        "ground_truth": "9:00 AM: Elena, 10:00 AM: Chloe, 11:00 AM: Priya, 12:00 PM: Marcus, 1:00 PM: Tariq"
    }
]

# ==============================================================================
# EXPERIMENT 3: STRUCTURED OUTPUT (JSON EXTRACTION)
# ==============================================================================

JSON_SCHEMA_DESCRIPTION = """
Extract expense receipts into valid JSON adhering strictly to this schema:
{
  "merchant": "string",
  "date": "string (YYYY-MM-DD)",
  "currency": "string (ISO 3-letter, e.g. USD, EUR)",
  "total_amount": float,
  "tax_amount": float or null,
  "payment_method": "string",
  "category": "string (one of: Meals & Entertainment, Travel & Lodging, Office Supplies, Technology & Software, Utilities)",
  "line_items": [
    {
      "description": "string",
      "quantity": int,
      "unit_price": float,
      "total": float
    }
  ],
  "is_reimbursable": boolean
}
"""

EXP3_TEST_INPUTS = [
    {
        "id": "Receipt_1_Dinner",
        "text": "Fwd: Dinner with client at Bistro Le Marais in NYC on Oct 24th, 2026. Billed $185.00 on my Amex ending in 4012, plus added $35.00 cash tip. We had 2 ribeye steaks at $65 each and a bottle of Pinot Noir for $55. Sales tax came out to $16.50. Mark this under Meals & Entertainment for client acquisition. Total paid: $220.00."
    },
    {
        "id": "Receipt_2_Software_Coupon",
        "text": "hey finance team, just renewed our annual team license for Linear Software Inc. today (2026-10-15). Paid via company Visa card. Base sub was 5 seats at $120/seat ($600), but we had a $50 welcome coupon so net before tax was $550.00. VAT/Tax was $44.00, so grand total charged was $594.00 USD. Definitely reimbursable under Technology & Software."
    },
    {
        "id": "Receipt_3_Airport_Ride",
        "text": "ALERT: Chase Card *8831 approved at UBER *TRIP SAN FRANCISCO CA on 10/18/2026. Amt: $62.85 USD. Details: Ride fare $54.10 (qty 1), SFO Airport surcharge $5.00 (qty 1), Bay Bridge Toll $3.75 (qty 1). Tax: $0.00. Business Travel & Lodging."
    },
    {
        "id": "Receipt_4_European_Format",
        "text": "RECHNUNG / INVOICE #89211 from CloudScale GmbH, Berlin. Datum: 05.11.2026. Zahlungsmethode: SEPA Wire Transfer. Pos 1: Dedicated Server Hosting Oct 2026, 1 Stück @ €450,00 = €450,00. Pos 2: Backup Storage Add-on 2TB, 2 Stück @ €25,00 = €50,00. Zwischensumme: €500,00. MwSt (19% Tax): €95,00. Gesamtbetrag: €595,00 EUR. Category: Technology & Software. Business expense: Yes."
    },
    {
        "id": "Receipt_5_Store_Mixed",
        "text": "Staples Store #0412 - 10/22/2026. Paid with Mastercard. Items: Ergonomic Wireless Mouse (qty 1 @ $49.99), 3 packs printer paper (qty 3 @ $8.50 = $25.50), personal chocolate bar ($2.50 - do not reimburse, personal). Subtotal eligible: $75.49. Store coupon -$10.00. Net eligible: $65.49. Sales tax: $5.24. Total charged for business items: $70.73 USD. Category: Office Supplies."
    },
    {
        "id": "Receipt_6_Credit_Adjustment",
        "text": "Notice from AWS Cloud Services dated 2026-09-30. Account *1094. Original monthly bill was $320.00 USD paid via Corporate Debit. AWS issued an SLA service outage credit of -$80.00. Revised total charge for September: $240.00 USD. Tax: $19.20. Line item: Cloud Compute EC2 instances (qty 1 @ $240.00). Approved under Technology & Software."
    }
]

def save_checkpoint(results):
    os.makedirs("results", exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def validate_json(raw_text):
    cleaned = raw_text.strip()
    had_markdown = False
    if cleaned.startswith("```"):
        had_markdown = True
        lines = cleaned.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        parsed = json.loads(cleaned)
        required_keys = ["merchant", "date", "currency", "total_amount", "tax_amount", "payment_method", "category", "line_items", "is_reimbursable"]
        missing = [k for k in required_keys if k not in parsed]
        schema_valid = len(missing) == 0
        return {
            "is_valid_json": True,
            "had_markdown_fences": had_markdown,
            "schema_conformant": schema_valid,
            "missing_keys": missing,
            "parsed_data": parsed
        }
    except Exception as e:
        return {
            "is_valid_json": False,
            "had_markdown_fences": had_markdown,
            "schema_conformant": False,
            "error": str(e)
        }

def run_all():
    print("="*70, flush=True)
    print("STARTING RESILIENT PROMPT ENGINEERING EXPERIMENTS", flush=True)
    print(f"Model: {MODEL_NAME} | Rate Limiting: 4.5s pace + 429 auto-backoff", flush=True)
    print("="*70, flush=True)

    # Load existing checkpoint if available
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r", encoding="utf-8") as f:
                results = json.load(f)
            print("Loaded existing results checkpoint.", flush=True)
        except Exception:
            results = None
    else:
        results = None

    if not results or "experiment_1_zero_vs_few_shot" not in results:
        results = {
            "metadata": {
                "model": MODEL_NAME,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "sdk": "google-genai"
            },
            "experiment_1_zero_vs_few_shot": [],
            "experiment_2_cot": [],
            "experiment_3_structured_output": {
                "strategy_a_naive": [],
                "strategy_b_explicit_schema": [],
                "strategy_c_api_json_mode": []
            }
        }

    # --------------------------------------------------------------------------
    # 1. RUN EXPERIMENT 1: ZERO-SHOT VS. FEW-SHOT
    # --------------------------------------------------------------------------
    existing_e1_ids = {x["test_id"] for x in results["experiment_1_zero_vs_few_shot"]}
    print("\n>>> Running Experiment 1: Zero-Shot vs Few-Shot Classification...", flush=True)
    for item in EXP1_TEST_INPUTS:
        if item["id"] in existing_e1_ids:
            print(f"  [Skipping already done] {item['id']}", flush=True)
            continue
        print(f"  Testing Input: {item['id']}...", flush=True)
        
        # Zero-shot
        prompt_zero = f"Input: {item['text']}\nOutput:"
        t0 = time.time()
        res_zero = call_with_retry(
            model=MODEL_NAME,
            contents=prompt_zero,
            config=types.GenerateContentConfig(
                system_instruction=ZERO_SHOT_SYSTEM,
                temperature=0.0
            )
        )
        t_zero = round(time.time() - t0, 3)
        time.sleep(4.5)

        # Few-shot
        prompt_few = f"Input: {item['text']}\nOutput:"
        t0 = time.time()
        res_few = call_with_retry(
            model=MODEL_NAME,
            contents=prompt_few,
            config=types.GenerateContentConfig(
                system_instruction=FEW_SHOT_SYSTEM,
                temperature=0.0
            )
        )
        t_few = round(time.time() - t0, 3)
        time.sleep(4.5)

        results["experiment_1_zero_vs_few_shot"].append({
            "test_id": item["id"],
            "input_text": item["text"],
            "expected": item["expected"],
            "zero_shot": {
                "system_instruction": ZERO_SHOT_SYSTEM,
                "user_prompt": prompt_zero,
                "raw_output": res_zero.text.strip(),
                "latency_sec": t_zero
            },
            "few_shot": {
                "system_instruction": FEW_SHOT_SYSTEM,
                "user_prompt": prompt_few,
                "raw_output": res_few.text.strip(),
                "latency_sec": t_few
            }
        })
        save_checkpoint(results)

    # --------------------------------------------------------------------------
    # 2. RUN EXPERIMENT 2: DIRECT VS. CHAIN-OF-THOUGHT
    # --------------------------------------------------------------------------
    existing_e2_ids = {x["problem_id"] for x in results["experiment_2_cot"]}
    print("\n>>> Running Experiment 2: Direct vs Chain-of-Thought Reasoning...", flush=True)
    for prob in EXP2_PROBLEMS:
        if prob["id"] in existing_e2_ids:
            print(f"  [Skipping already done] {prob['id']}", flush=True)
            continue
        print(f"  Testing Problem: {prob['id']}...", flush=True)

        prompt_direct = f"{prob['problem']}\n\nAnswer directly with the final numerical value or concise list without explaining your thought process."
        t0 = time.time()
        res_direct = call_with_retry(
            model=MODEL_NAME,
            contents=prompt_direct,
            config=types.GenerateContentConfig(temperature=0.0)
        )
        t_direct = round(time.time() - t0, 3)
        time.sleep(4.5)

        prompt_cot = f"{prob['problem']}\n\nThink step by step. Show your step-by-step intermediate calculations or deductions carefully, identify any constraints or potential traps, and then state the final answer clearly at the end."
        t0 = time.time()
        res_cot = call_with_retry(
            model=MODEL_NAME,
            contents=prompt_cot,
            config=types.GenerateContentConfig(temperature=0.0)
        )
        t_cot = round(time.time() - t0, 3)
        time.sleep(4.5)

        results["experiment_2_cot"].append({
            "problem_id": prob["id"],
            "title": prob["title"],
            "problem_text": prob["problem"],
            "ground_truth": prob["ground_truth"],
            "direct": {
                "prompt": prompt_direct,
                "raw_output": res_direct.text.strip(),
                "latency_sec": t_direct
            },
            "chain_of_thought": {
                "prompt": prompt_cot,
                "raw_output": res_cot.text.strip(),
                "latency_sec": t_cot
            }
        })
        save_checkpoint(results)

    # --------------------------------------------------------------------------
    # 3. RUN EXPERIMENT 3: STRUCTURED OUTPUT (3 STRATEGIES)
    # --------------------------------------------------------------------------
    existing_e3_ids = {x["receipt_id"] for x in results["experiment_3_structured_output"]["strategy_a_naive"]}
    print("\n>>> Running Experiment 3: Structured Output across 3 Strategies...", flush=True)
    for item in EXP3_TEST_INPUTS:
        if item["id"] in existing_e3_ids:
            print(f"  [Skipping already done] {item['id']}", flush=True)
            continue
        print(f"  Testing Receipt: {item['id']}...", flush=True)

        # Strategy A: Naive Natural Language Prompt
        prompt_a = f"Extract the expense receipt information from this text into JSON format:\n\n{item['text']}"
        t0 = time.time()
        res_a = call_with_retry(
            model=MODEL_NAME,
            contents=prompt_a,
            config=types.GenerateContentConfig(temperature=0.0)
        )
        t_a = round(time.time() - t0, 3)
        time.sleep(4.5)

        # Strategy B: Explicit Schema + Format Constraints in Prompt
        prompt_b = (
            f"You are a strict financial receipt parser.\n"
            f"{JSON_SCHEMA_DESCRIPTION}\n"
            f"Rules:\n"
            f"- Return ONLY the JSON object. Do not wrap in markdown ```json or include conversational preamble.\n"
            f"- Dates must be in YYYY-MM-DD format.\n"
            f"- Numeric amounts must be numbers, not strings.\n"
            f"- Ensure all required keys exist.\n\n"
            f"Text to parse:\n{item['text']}"
        )
        t0 = time.time()
        res_b = call_with_retry(
            model=MODEL_NAME,
            contents=prompt_b,
            config=types.GenerateContentConfig(temperature=0.0)
        )
        t_b = round(time.time() - t0, 3)
        time.sleep(4.5)

        # Strategy C: Explicit Schema + API-Level JSON Mode
        prompt_c = (
            f"Extract the expense receipt from the following text according to this schema:\n"
            f"{JSON_SCHEMA_DESCRIPTION}\n\n"
            f"Text:\n{item['text']}"
        )
        t0 = time.time()
        res_c = call_with_retry(
            model=MODEL_NAME,
            contents=prompt_c,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0
            )
        )
        t_c = round(time.time() - t0, 3)
        time.sleep(4.5)

        val_a = validate_json(res_a.text)
        val_b = validate_json(res_b.text)
        val_c = validate_json(res_c.text)

        results["experiment_3_structured_output"]["strategy_a_naive"].append({
            "receipt_id": item["id"],
            "raw_output": res_a.text.strip(),
            "validation": val_a,
            "latency_sec": t_a
        })
        results["experiment_3_structured_output"]["strategy_b_explicit_schema"].append({
            "receipt_id": item["id"],
            "raw_output": res_b.text.strip(),
            "validation": val_b,
            "latency_sec": t_b
        })
        results["experiment_3_structured_output"]["strategy_c_api_json_mode"].append({
            "receipt_id": item["id"],
            "raw_output": res_c.text.strip(),
            "validation": val_c,
            "latency_sec": t_c
        })
        save_checkpoint(results)

    print("\n" + "="*70, flush=True)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!", flush=True)
    print(f"Results saved to {RESULTS_FILE}", flush=True)
    print("="*70, flush=True)

if __name__ == "__main__":
    run_all()
