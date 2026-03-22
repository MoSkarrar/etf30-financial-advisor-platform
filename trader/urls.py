from django.urls import path

from . import views

urlpatterns = [
    path("", views.home, name="trader-home"),
    path("narration/session/<str:session_id>/", views.narration_session, name="narration_session"),
]