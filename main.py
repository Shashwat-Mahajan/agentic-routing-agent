import logging
import os
import re
import sys
import types
from typing import Dict, List, Optional

from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools import tool

# Agno's workflow package imports optional RemoteWorkflow (fastapi). This app doesn't use it.
try:
    import fastapi as _fastapi  # type: ignore  # noqa: F401
except Exception:
    _fastapi_stub = types.ModuleType("fastapi")

    class WebSocket:  # minimal stub for agno.workflow.remote import
        pass

    _fastapi_stub.WebSocket = WebSocket
    sys.modules.setdefault("fastapi", _fastapi_stub)

from agno.workflow.router import Router
from agno.workflow.step import Step
from agno.workflow.types import StepInput
from agno.workflow.workflow import Workflow


def _load_dotenv() -> None:
    """Load key=value pairs from a .env file next to this module (no extra dependency)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


_load_dotenv()

# ----------- Setup Logging -----------
logger = logging.getLogger("CustomerSupport")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.handlers = [handler]


def create_llm():
    """Preferred LLM: Groq if GROQ_API_KEY is set, else OpenAI if OPENAI_API_KEY is set."""
    return Groq(id="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))



# ----------- Tool: Mock Product Catalog -----------
MOCK_CATALOG = {
    "iPhone 15": {"price": "$999", "availability": "In Stock", "offers": "10% off on exchange"},
    "Galaxy S23": {"price": "$899", "availability": "Limited Stock", "offers": None},
    "Pixel 8": {"price": "$799", "availability": "Out of Stock", "offers": "5% launch discount"},
}


def _product_catalog_lookup_impl(query: str) -> str:
    for prod, info in MOCK_CATALOG.items():
        if prod.lower() in query.lower():
            result = f"{prod}: Price: {info['price']}, Availability: {info['availability']}"
            if info["offers"]:
                result += f", Offer: {info['offers']}"
            return result
    return "Product not found in catalog."


@tool
def product_catalog_lookup(query: str) -> str:
    """Deterministic sales lookup for price, availability, and offers."""
    return _product_catalog_lookup_impl(query)


# ----------- Tool: Mock Troubleshooting Knowledge Base -----------
MOCK_KB = {
    "overheating": "Try closing background apps and remove phone case. If the issue persists, restart the device or contact support.",
    "won't turn on": "Charge the device for at least 30 minutes. If it still won't turn on, perform a hard reset or visit a service center.",
    "screen flickering": "Check for recent software updates. If persists, lower brightness and disable adaptive brightness.",
}


def _troubleshooting_lookup_impl(query: str) -> str:
    text = query.lower()
    for k in MOCK_KB.keys():
        if k in text:
            return f"Troubleshooting ({k}): {MOCK_KB[k]}"
    return "No troubleshooting steps found for this issue."


@tool
def troubleshooting_lookup(query: str) -> str:
    """Deterministic troubleshooting lookup for known technical issues."""
    return _troubleshooting_lookup_impl(query)


# ----------- Agents (factory) -----------
def build_sales_agent(model) -> Agent:
    return Agent(
        name="SalesAgent",
        model=model,
        instructions="SalesAgent should only format sales output when asked by the workflow.",
        markdown=False,
    )


def build_tech_agent(model) -> Agent:
    return Agent(
        name="TechSupportAgent",
        model=model,
        instructions="TechSupportAgent should only format troubleshooting output when asked by the workflow.",
        markdown=False,
    )


def build_general_agent(model) -> Agent:
    return Agent(
        name="GeneralInfoAgent",
        model=model,
        instructions=(
            "Answer policy, refund, returns, warranty, shipping, and FAQ questions only. "
            "If exact policy details are missing, provide a safe generic support-policy answer."
        ),
        markdown=False,
    )


def _contains_any(text: str, keywords: List[str]) -> bool:
    return any(k in text for k in keywords)


def _split_clauses(text: str) -> List[str]:
    chunks = re.split(
        r"\s*(?:,|;|\?|\.\s+|\band\b|\balso\b|\bplus\b)\s*",
        text,
        flags=re.IGNORECASE,
    )
    return [c.strip() for c in chunks if c and c.strip()]


def _pick_product_name(text: str) -> str:
    for product in MOCK_CATALOG.keys():
        if product.lower() in text.lower():
            return product
    return text


def _extract_product_name(text: str) -> Optional[str]:
    for product in MOCK_CATALOG.keys():
        if product.lower() in text.lower():
            return product
    return None


def build_workflow() -> Workflow:
    model = create_llm()

    sales_agent = build_sales_agent(model)
    tech_agent = build_tech_agent(model)
    general_agent = build_general_agent(model)

    def make_sales_step(subquery: str, product_name: Optional[str]) -> Step:
        def _executor(_step_input: StepInput) -> str:
            # Ensure the deterministic tool lookup always receives the product name.
            product_for_lookup = product_name or _extract_product_name(subquery) or _pick_product_name(subquery)
            return _product_catalog_lookup_impl(product_for_lookup)

        return Step(name="SalesAgent", executor=_executor, add_workflow_history=True)

    def make_tech_step(subquery: str) -> Step:
        def _executor(_step_input: StepInput) -> str:
            # Pass an extracted issue key to improve determinism.
            lower = subquery.lower()
            if "overheat" in lower or "overheating" in lower:
                issue = "overheating"
            elif "won't turn on" in lower or "wont turn on" in lower:
                issue = "won't turn on"
            elif "flicker" in lower:
                issue = "screen flickering"
            else:
                issue = subquery
            return _troubleshooting_lookup_impl(issue)

        return Step(name="TechSupportAgent", executor=_executor, add_workflow_history=True)

    def make_general_step(subquery: str) -> Step:
        def _executor(_step_input: StepInput) -> str:
            lower = subquery.lower()
            if "refund" in lower or "return" in lower or "policy" in lower:
                return (
                    "Refund Policy: You can return your purchase within 30 days of delivery for a full refund. "
                    "Items must be in their original condition with all tags and packaging included. "
                    "Refunds are processed within 5-7 business days after we receive the return."
                )
            if "warranty" in lower:
                return (
                    "Warranty: Warranty coverage depends on the product and purchase terms. "
                    "If you share your purchase details and the issue you’re experiencing, we can help next steps."
                )
            if "shipping" in lower or "delivery" in lower:
                return (
                    "Shipping & Delivery: Delivery timelines vary by location and inventory availability. "
                    "If your order is delayed, provide your order number and we’ll help track it."
                )
            if "cancel" in lower or "cancellation" in lower:
                return (
                    "Cancellation: If your order has not shipped yet, we can usually cancel it. "
                    "Please provide your order details so we can confirm the status."
                )
            return "FAQs: Please contact customer support and include the question you have so we can provide the correct policy information."

        return Step(name="GeneralInfoAgent", executor=_executor, add_workflow_history=True)

    def simple_intent_router(step_input: StepInput) -> List[Step]:
        full_query = (step_input.get_input_as_string() or "").strip()
        text = full_query.lower()

        sales_keywords = [
            "price",
            "pricing",
            "cost",
            "offer",
            "offers",
            "deal",
            "discount",
            "availability",
            "stock",
            "in stock",
            "out of stock",
        ]
        tech_keywords = [
            "overheat",
            "overheating",
            "hot",
            "won't turn on",
            "wont turn on",
            "not turning on",
            "not working",
            "screen",
            "flicker",
            "flickering",
            "bug",
            "error",
            "troubleshoot",
            "crash",
            "stuck",
            "battery drain",
            "restart",
        ]
        general_keywords = [
            "return",
            "returns",
            "refund",
            "refunds",
            "policy",
            "policies",
            "warranty",
            "shipping",
            "delivery",
            "faq",
            "faqs",
            "replacement",
            "cancel",
            "cancellation",
            "support hours",
            "business hours",
            "faq",
        ]

        sales_parts: List[str] = []
        tech_parts: List[str] = []
        general_parts: List[str] = []

        clauses = _split_clauses(full_query)
        logger.info("Original query: %s", full_query)
        logger.info("Split clauses: %s", clauses)

        product_name = _extract_product_name(full_query)

        for clause in clauses:
            c = clause.lower()
            if _contains_any(c, general_keywords):
                general_parts.append(clause)
            if _contains_any(c, tech_keywords):
                tech_parts.append(clause)
            if _contains_any(c, sales_keywords):
                sales_parts.append(clause)

        if not sales_parts and _contains_any(text, sales_keywords):
            sales_parts.append(full_query)
        if not tech_parts and _contains_any(text, tech_keywords):
            tech_parts.append(full_query)
        if not general_parts and _contains_any(text, general_keywords):
            general_parts.append(full_query)

        routed: Dict[str, str] = {}
        selected: List[Step] = []
        if general_parts:
            routed["GeneralInfoAgent"] = "; ".join(dict.fromkeys(general_parts))
            selected.append(make_general_step(routed["GeneralInfoAgent"]))
        if tech_parts:
            routed["TechSupportAgent"] = "; ".join(dict.fromkeys(tech_parts))
            selected.append(make_tech_step(routed["TechSupportAgent"]))
        if sales_parts:
            sales_subquery = "; ".join(dict.fromkeys(sales_parts))
            if product_name and product_name.lower() not in sales_subquery.lower():
                sales_subquery = f"{product_name}: {sales_subquery}"

            routed["SalesAgent"] = sales_subquery
            selected.append(make_sales_step(sales_subquery, product_name))

        if not selected:
            routed["GeneralInfoAgent"] = full_query
            selected.append(make_general_step(full_query))

        for agent_name, subquery in routed.items():
            logger.info("Route -> %s: %s", agent_name, subquery)

        # Append a final deterministic join step after all selected agents.
        selected.append(Step(name="FinalResponse", executor=combine_executor, add_workflow_history=True))

        return selected

    router = Router(
        name="CustomerSupportRouter",
        choices=[
            Step(name="SalesAgent", executor=lambda _in: "", add_workflow_history=True),
            Step(name="TechSupportAgent", executor=lambda _in: "", add_workflow_history=True),
            Step(name="GeneralInfoAgent", executor=lambda _in: "", add_workflow_history=True),
        ],
        selector=simple_intent_router,
        allow_multiple_selections=True,
    )

    def combine_executor(step_input: StepInput) -> str:
        prev = getattr(step_input, "previous_step_outputs", None) or {}
        parts: List[str] = []

        for agent_name in ["GeneralInfoAgent", "TechSupportAgent", "SalesAgent"]:
            step_out = prev.get(agent_name)
            if not step_out:
                continue
            content = getattr(step_out, "content", None)
            if content:
                parts.append(str(content).strip())
            else:
                parts.append(str(step_out).strip())

        return "\n---\n".join([p for p in parts if p])

    return Workflow(
        name="CustomerSupportWorkflow",
        steps=[router],
    )


# ----------- CLI Demo / Entrypoint -----------
def demo():
    workflow = build_workflow()
    workflow.cli_app(markdown=False, show_step_details=True, exit_on=["exit"])


if __name__ == "__main__":
    demo() 