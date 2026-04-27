from dotenv import load_dotenv
load_dotenv()

from inventory_agent import inventory_agent

from agno.eval.performance import PerformanceEval

def run_agent():
    resp1 = inventory_agent.run(" Is AirPods Pro available?")
    print(resp1)

perfo = PerformanceEval(
    name="Inventory Agent",
    func=run_agent,   
    num_iterations=1,
    warmup_runs=0
)

if __name__ == "__main__":
    perfo.run(print_results=True, print_summary=True)