# Face Swap Telegram Bot

## Overview
This is a simple pet project: a Telegram bot that allows users to seamlessly swap faces in their images. There are two types of swapping: using classical computer vision algorithms (like face meshing) based on facial landmarks obtained from the neural network, and the more accurate method using a pretrained model to swap faces from two given photos. Additionally, it can be deployed using Docker.

## Features

- This project uses two types of swapping (all model results can be found either in the original repositories or in the `/research/` folder).
  - Face parsing (by OpenCV, [Facer](https://github.com/FacePerceiver/facer), and [FaRL](https://github.com/FacePerceiver/FaRL))
  - [SimSwap](https://github.com/neuralchen/SimSwap) model
  - The [e4s model](https://github.com/e4s2022/e4s) showed unsatisfactory results.
- Telegram integration for easy accessibility.
- User-friendly commands with inline image processing.

## Getting Started
Follow these steps to set up the bot.

### Prerequisites
- Telegram API token (get yours at [Telegram BotFather](https://t.me/botfather))

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/slava-qw/face-swap-tg-bot.git
    cd face-swap-tg-bot
    ```
2. Write down your Telegram API token in the configuration file (`.env`). Also, for [correct work](https://docs.python-telegram-bot.org/en/v20.8/telegram.bot.html#telegram.Bot.set_webhook), get the https-link to set up the webhook (e.g., https://localhost.run/docs/).

#### Local
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the code:
    ```bash
    python main.py
    ```

#### Docker
2. Build a Docker image and run a container:
    ```bash
    docker-compose up
    ```
