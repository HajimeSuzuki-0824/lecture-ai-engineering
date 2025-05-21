# notify.py
import requests
import os

def notify_slack(message: str):
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if webhook_url:
        requests.post(webhook_url, json={"text": message})
