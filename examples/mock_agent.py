import re
from typing import Any, Callable

ORDERS_DB = {
    "ORD-101": {"status": "shipped", "carrier": "FedEx", "amount": 120.0},
    "ORD-102": {"status": "delivered", "carrier": "UPS", "amount": 45.0},
    "ORD-103": {"status": "processing", "carrier": "USPS", "amount": 80.0},
}


def lookup_order(order_id: str) -> dict[str, Any]:
    if order_id not in ORDERS_DB:
        raise ValueError(f"Order {order_id} not found")
    return dict(ORDERS_DB[order_id])


def cancel_order(order_id: str) -> dict[str, str]:
    if order_id not in ORDERS_DB:
        raise ValueError(f"Order {order_id} not found")
    if ORDERS_DB[order_id]["status"] == "shipped":
        return {"status": "rejected", "reason": "Already shipped"}
    return {"status": "cancelled", "order_id": order_id}


class SupportAgent:
    def __init__(
        self,
        lookup_tool: Callable[[str], dict[str, Any]] = lookup_order,
        cancel_tool: Callable[[str], dict[str, str]] = cancel_order,
    ) -> None:
        self.lookup_tool = lookup_tool
        self.cancel_tool = cancel_tool

    def run(self, query: str) -> str:
        match = re.search(r"ORD-\d+", query.upper())
        if not match:
            return "Please provide a valid order ID (e.g., ORD-101)."

        order_id = match.group()

        if "cancel" in query.lower():
            try:
                res = self.cancel_tool(order_id)
                if res.get("status") == "cancelled":
                    return f"Order {order_id} has been successfully cancelled."
                return f"Cannot cancel order {order_id}: {res.get('reason')}."
            except Exception as e:
                return f"Unable to cancel order {order_id} due to an error: {e}"

        try:
            info = self.lookup_tool(order_id)
            return (
                f"Order {order_id} is currently {info['status']} "
                f"via {info['carrier']}."
            )
        except Exception as e:
            return f"Could not find details for order {order_id}: {e}"
