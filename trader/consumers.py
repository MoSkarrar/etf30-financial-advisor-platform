import json

from asgiref.sync import async_to_sync
from channels.generic.websocket import WebsocketConsumer

from trader.services import narration_service, trading_service


class TradeConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()

    def receive(self, text_data=None, bytes_data=None):
        try:
            payload = json.loads(text_data or "{}")
        except json.JSONDecodeError as exc:
            self.send(text_data=json.dumps({"type": "terminal", "message": f"Invalid JSON payload: {exc}"}, ensure_ascii=False))
            return

        try:
            trading_service.execute_trade(self, payload)
        except Exception as exc:
            self.send(text_data=json.dumps({"type": "terminal", "message": f"Execution error: {exc}"}, ensure_ascii=False))

    def disconnect(self, close_code):
        return


class NarrationConsumer(WebsocketConsumer):
    def connect(self):
        session_id = self.scope["url_route"]["kwargs"]["session_id"]
        self.state = narration_service.create_session_state(session_id)
        self.accept()
        for payload in narration_service.connect_messages(self.state):
            self._push(payload)

    def receive(self, text_data=None, bytes_data=None):
        try:
            data = json.loads(text_data or "{}")
        except json.JSONDecodeError as exc:
            self._push({"type": "advisor_answer", "message": f"Invalid JSON payload: {exc}"})
            return

        message_type = (data.get("type") or "").strip()

        if message_type == "load_run":
            for payload in narration_service.load_run(self.state, data.get("run_id", "")):
                self._push(payload)
            return

        if message_type in {
            "ask_advisor",
            "ask_technical_xai",
            "compare_explainers",
            "show_risk_flags",
            "compare_benchmark",
            "run_scenario",
            "explain_weight_change",
        }:
            narration_service.handle_event_async(self.state, message_type, data, self._push)
            return

        self._push({"type": "advisor_answer", "message": f"Unsupported message type: {message_type}"})

    def disconnect(self, close_code):
        return

    def _push(self, payload: dict):
        async_to_sync(self.channel_layer.send)(self.channel_name, {"type": "push.json", "payload": payload})

    def push_json(self, event):
        self.send(text_data=json.dumps(event["payload"], ensure_ascii=False))
