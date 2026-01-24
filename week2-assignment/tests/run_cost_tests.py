from types import SimpleNamespace
import time

# Basic test harness for agent helpers

def run_all_tests():
    import sys, os
    sys.path.insert(0, os.path.join(os.getcwd(), "week2-assignment"))
    import agent

    # Replace real cost_tracker with a fake that returns deterministic numbers
    class FakeCostTracker:
        def track_llm_call(self, *, user_id, model, input_tokens, output_tokens, node):
            input_cost = float(input_tokens) * 0.000001
            output_cost = float(output_tokens) * 0.000002
            call_cost = input_cost + output_cost
            return {
                "timestamp": time.time(),
                "user_id": user_id,
                "model": model,
                "node": node,
                "input_tokens": int(input_tokens),
                "output_tokens": int(output_tokens),
                "input_cost": input_cost,
                "output_cost": output_cost,
                "call_cost": call_cost,
            }

    fake = FakeCostTracker()
    agent.cost_tracker = fake

    samples = []

    # Sample 1: dict-shaped LLM response with usage
    samples.append({
        "name": "dict_usage",
        "result": {"usage": {"prompt_tokens": 20, "completion_tokens": 150}}
    })

    # Sample 2: object with llm_response attribute containing usage dict
    class ObjWithUsage:
        def __init__(self):
            self.llm_response = {"usage": {"prompt_tokens": 45, "completion_tokens": 200}}

    samples.append({"name": "attr_llm_response", "result": ObjWithUsage()})

    # Sample 3: object with direct attributes
    class ObjDirect:
        def __init__(self):
            self.prompt_tokens = 60
            self.completion_tokens = 120

    samples.append({"name": "direct_attrs", "result": ObjDirect()})

    # Sample 4: no usage information -> fallback
    samples.append({"name": "no_usage", "result": "short text response"})

    state = {"request_costs": [], "total_cost": 0.0, "cost_summary": None, "trace_id": "test-trace"}

    for s in samples:
        print("--- Testing:", s["name"])
        in_toks, out_toks = agent._extract_token_usage(s["result"], fallback_prompt_text="Testing prompt content")
        print(f"Extracted tokens -> input: {in_toks}, output: {out_toks}")

        cost_info = agent._track_and_append_cost(
            state=state,
            llm_result=s["result"],
            model="gpt-4o-mini",
            node=f"test_node_{s['name']}",
            prompt_text="Testing prompt content",
            user_tier="standard"
        )

        print("Cost info:", cost_info)
        print("State total_cost:", state.get("total_cost"))
        print("State cost_summary:", state.get("cost_summary"))
        print()

    print("Final request_costs count:", len(state.get("request_costs", [])))


if __name__ == '__main__':
    run_all_tests()
