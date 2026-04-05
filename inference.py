import os
import requests
from openai import OpenAI


API_BASE_URL     = os.getenv("API_BASE_URL", "https://dharshansriram-ai-pr-env.hf.space")
MODEL_NAME       = os.getenv("MODEL_NAME",   "gpt-4")
HF_TOKEN         = os.getenv("HF_TOKEN", "")

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL,
)

SCENARIOS = ["easy", "medium", "hard"]


def run_scenario(scenario):
    print(f"[START] scenario={scenario}")

    res = requests.get(
        f"{API_BASE_URL}/demo",
        params={"scenario": scenario, "seed": 42},
    )

    data = res.json()
    score = data["data"]["weighted_total"]

    print(f"[STEP] scenario={scenario} score={score}")
    print(f"[END] scenario={scenario} final_score={score}")

    return score


def main():
    results = {}

    for scenario in SCENARIOS:
        score = run_scenario(scenario)
        results[scenario] = score

    print("[FINAL RESULTS]")
    print(results)


if __name__ == "__main__":
    main()
