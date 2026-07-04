"""Example: Using AIngram as a Hermes Agent memory provider.

Shows semantic recall — finding facts using different words than stored.
"""
from aingram.integrations.hermes import AIngramHermesMemory

DB = "hermes_example.db"


def main():
    with AIngramHermesMemory(DB) as mem:
        # Simulate facts accumulated across Hermes sessions
        mem.remember("The gold trading bot uses a 10-month moving average to decide when to be in the market.")
        mem.remember("Old bot version v12 lost $29,723 due to bugs — MIN_ATR blocked 99% of trades.")
        mem.remember("User is based in Langkawi and owns a Jeanneau SO40 sailboat named EFA Mar.")
        mem.remember("Risk management: max $500 per trade, hard daily stop at -2% of account.")
        mem.remember("OANDA demo account balance is approximately $60,836.")
        mem.remember("Faber TAA is the primary strategy; Clean Bot is secondary.")

        print("=" * 60)
        print("AIngram Hermes Memory — Semantic Recall Demo")
        print("=" * 60)

        queries = [
            ("what happened to the old automated trader?", "old bot failure"),
            ("tell me about the guy who owns the yacht", "user identity"),
            ("how do we limit losses?", "risk management"),
            ("what strategy is in charge?", "primary strategy"),
            ("how much cash is in the paper account?", "account balance"),
        ]

        for query, label in queries:
            results = mem.recall(query, limit=2)
            print(f"\n[{label}] Query: \"{query}\"")
            for r in results:
                print(f"  [{r['score']:.4f}] {r['content'][:80]}...")


if __name__ == "__main__":
    main()
