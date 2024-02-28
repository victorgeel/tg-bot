import subprocess
from set_webhook import set_webhook


def main():
    webhook_res = subprocess.run(['python', 'set_webhook.py'])
    webhook_res = set_webhook()
    print(webhook_res)
    if not webhook_res['ok']:
        raise Exception(webhook_res['description'])

    bot_res = subprocess.run(['python', 'bot.py'])


if __name__ == '__main__':
    main()
