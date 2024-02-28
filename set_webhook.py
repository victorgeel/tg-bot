import os
import requests
import subprocess
from dotenv import load_dotenv


def set_webhook():
    """
    Set the tg webhook.
    Some useful links:
    https://www.youtube.com/live/9ue5nFEunX0?si=Bcd4nXqVojgQgTNq,
    https://www.youtube.com/live/bM8jhOvdbGI?si=48VENirvAVFzmMCL,
    http://localhost.run/docs/
    """

    load_dotenv()
    TOKEN = os.getenv('BOT_TOKEN')
    whook = os.getenv('URL')

    req = f'https://api.telegram.org/bot{TOKEN}/setWebhook?url=https://{whook}/'
    r = requests.get(req)
    return r.json()



