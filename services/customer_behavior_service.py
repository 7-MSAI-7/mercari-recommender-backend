from fastapi import Request

class CustomerBehaviorService:
    @staticmethod
    def store_customer_behavior(request: Request, name: str, event: str):
        if "behavior" not in request.session:
            request.session["behavior"] = []
        request.session["behavior"].append({"name": name, "event": event})
        return request.session["behavior"]

    @staticmethod
    def get_behaviors(request: Request):
        return request.session.get("behavior", []) 