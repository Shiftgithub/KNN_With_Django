# app/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('knn_predict/', views.knn_predict, name='knn_predict'),
]
