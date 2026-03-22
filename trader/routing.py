from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/execute-model/', consumers.TradeConsumer.as_asgi()),
    re_path(r"ws/narration/(?P<session_id>[A-Za-z0-9_\-]+)/$", consumers.NarrationConsumer.as_asgi()),
]