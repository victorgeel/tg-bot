import asyncio
import aiofiles
import io
import logging
import os
# from bot_config import *
import uvicorn
from fastapi import FastAPI, Request, Response
from http import HTTPStatus

from dotenv import load_dotenv
from faceswap_arch import *

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update, InputMediaPhoto
from telegram.ext import (
    filters,
    ConversationHandler,
    MessageHandler,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes
)
import json
from user_db import *

# from warnings import filterwarnings
# filterwarnings("ignore")

# Enable logging
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.message.from_user
    logger.info("User %s started the conversation.", user.first_name)

    user_id = user.id
    user_name = user.first_name

    conn = await create_connection(db_dir)
    await insert_data(conn, user_id, ('user_id', 'user_name'), [user_id, user_name])

    if user_id != admin_id:
        await context.bot.send_message(chat_id=update.effective_chat.id,
                                       text="Sorry, bot isn't working right now. I'm probably working on it to make it better. Come back later.")
        return ConversationHandler.END
    else:
        await context.bot.send_message(chat_id=update.effective_chat.id,
                                       text="Hello there, I'm a bot that will change your photo to a real 'master piece' ("
                                            "if you know that I mean :) ). Choose a selfie and send it to me.")

    return START_ROUTES


async def photo_choose(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler to handle ONLY the photos"""

    logger.info("User %s send photo.", update._effective_user.first_name)

    status_code = START_ROUTES

    keyboard = [
        [InlineKeyboardButton("Choose from our gallery", callback_data="1")],
        [InlineKeyboardButton("Send your own photo", callback_data="2")],
    ]

    send_data = {
        "text": 'Your photo has been sent to the neural '
                'network, in order to process it. But '
                'before to get the final result, please '
                'choose the second photo in which you want '
                'to insert your face (or just choose one '
                'from our gallery)',
        "reply_markup": InlineKeyboardMarkup(keyboard)}

    if update.callback_query:
        query = update.callback_query
        await query.answer()

        if update.callback_query.data == 'go back':
            logger.info("User %s go back.", update._effective_user.first_name)
            # send_message_func = query.edit_message_text
            await query.edit_message_text(**send_data)
    else:
        logger.info("User %s download photo.", update._effective_user.first_name)

        send_data["chat_id"] = update.effective_chat.id
        # send_message_func = context.bot.send_message

        new_file = await update.message.effective_attachment[-1].get_file()
        photo_data = await new_file.download_as_bytearray()

        # Count the number of photos
        conn = await create_connection(db_dir)
        query_text = f'SELECT COUNT(photo_1) FROM user_data WHERE user_id = ?'
        result = await execute_query(conn, query_text, (update.message.from_user.id,))

        if result == 0:  # if it's only the 1st user's photo
            logger.info("User %s: first photo.", update._effective_user.first_name)

            # Insert into the first column if it's empty
            await update_cols(conn, update.message.from_user.id, ['photo_1'], [photo_data])

            await context.bot.send_message(**send_data)

        else:  # when the user will send the 2nd photo
            logger.info("User %s: second photo.", update._effective_user.first_name)

            # Insert into the second column if the first column is not empty
            await update_cols(conn, update.message.from_user.id, ['photo_2'], [photo_data])

            send_data['text'] = 'All two photos had received. So, now, please, wait. OR you can play the mini-game ' \
                                'instead '

            ########################################################################################################################

            # make buttons for the mini-game while waiting the processing of the photos
            # keyboard = [
            #     [InlineKeyboardButton("Start.", callback_data="start the game")],
            #     [InlineKeyboardButton("Just wait.", callback_data="just wait")],
            # ]
            # send_data['reply_markup'] = InlineKeyboardMarkup(keyboard)

            send_data['reply_markup'] = None

            # status_code = QUEST_ROUTES
            await context.bot.send_message(**send_data)

            bot = context.bot
            user_id = update._effective_user.id

            if swap_type == 'mask swap':
                await mask_swap(bot, conn, update, user_id)
            elif swap_type == 'simswap':
                await sim_swap(bot, conn, update, user_id)
            else:
                logger.info("User %s: unknown swap type.", update._effective_user.first_name)
            ########################################################################################################################

        # Commit the changes and close the connection
        # await conn.commit()

    # await send_message_func(**send_data)

    return status_code


async def back_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Parses the CallbackQuery and updates the message text."""

    logger.info("User %s go back (button).", update._effective_user.first_name)

    query = update.callback_query
    await query.answer()

    if query.data == '1':
        logger.info("User %s touch back button.", update._effective_user.first_name)

        media = [InputMediaPhoto(media=open(os.path.join(gallery_dir_path, path), 'rb')) for path in
                 os.listdir(gallery_dir_path)]

        keyboard = [[InlineKeyboardButton(f"{i + 1}", callback_data=f"photo_{i + 1}") for i in
                     range(len(os.listdir(gallery_dir_path)))]]
        keyboard.append([InlineKeyboardButton("Go back", callback_data="go back")])

        await update.get_bot().send_media_group(chat_id=update.effective_chat.id, media=media)
        await asyncio.sleep(0.2)

        await update.get_bot().send_message(
            chat_id=update.effective_chat.id,
            text=f"Choose one of the given photos.",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    elif query.data == '2':
        logger.info("User %s send its own photo.", update._effective_user.first_name)

        await query.edit_message_text(text=f"Please send another photo.")

    elif query.data.startswith('photo'):  # when the user chose the 2nd photo from gallery
        photo_idx = int(query.data.split('_')[1])

        photo_data = bytearray(open(os.path.join(gallery_dir_path, f'photo_{photo_idx}.jpg'), 'rb').read())

        conn = await create_connection(db_dir)
        await update_cols(conn, update._effective_user.id, ['photo_2'], [photo_data])

        ########################################################################################################################

        # make buttons for the mini-game while waiting the processing of the photos
        # keyboard = [
        #     [InlineKeyboardButton("Start.", callback_data="start the game")],
        #     [InlineKeyboardButton("Just wait.", callback_data="just wait")],
        # ]

        await query.edit_message_text(text=f'Your photo (number {photo_idx}) has been saved. Thus, all two needed '
                                           f'photos received. So, now, please, wait. OR you can play the mini-game '
                                           f'instead',
                                      # reply_markup=InlineKeyboardMarkup(keyboard)
                                      )

        bot = context.bot
        user_id = update._effective_user.id

        if swap_type == 'mask swap':
            await mask_swap(bot, conn, update, user_id)
        elif swap_type == 'simswap':
            await sim_swap(bot, conn, update, user_id)
        else:
            logger.info("User %s: unknown swap type.", update._effective_user.first_name)

        # return QUEST_ROUTES
        return WAITING_ROUTES
        ########################################################################################################################

    elif query.data == 'go back':
        logger.info("User %s really go back.", update._effective_user.first_name)

        return await photo_choose(update, context)  # casue I need the same text and button structure


async def quest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()  # if you don't need a popup message about starting the image preprocessing

    if query.data == 'just wait' or query.data == 'go back':
        await query.edit_message_text(text=f"Please, wait until the process is done.")
        return WAITING_ROUTES
    else:
        await query.edit_message_text(text=f"Sorry, this isn't working right now :(. Please, wait until the process is done.")
        return WAITING_ROUTES

        # FIXME: lots of long pauses during mini-game because of the image processing
        conn = await create_connection(db_dir)

        # read json-file with questions
        # https://www.twilio.com/blog/working-with-files-asynchronously-in-python-using-aiofiles-and-asyncio
        # https://github.com/Tinche/aiofiles
        async with aiofiles.open(quests_path, mode='r') as f:
            result = await f.read()

        quests_dict = json.loads(result)

        if query.data != 'start the game' and query.data != 'play again':  # change the num of right-answered questions (during the answering)
            query_text = f'SELECT r_q_num FROM user_data WHERE user_id = ?'
            prev_score: int = await execute_query(conn, query_text, (query.from_user.id,))

            i, j = int(query.data.split(' ')[0]), int(query.data.split(' ')[2])

            given_ans = quests_dict['questions'][i]['options'][j]
            real_ans = quests_dict['questions'][i]['answer']

            if given_ans == real_ans:
                prev_score += 1

            await update_cols(conn, query.from_user.id, ['r_q_num'], [prev_score])

        else:  # here query.data == 'play again'
            await update_cols(conn, query.from_user.id, ['q_num', 'r_q_num'], [0, 0])

        query_text = f'SELECT q_num FROM user_data WHERE user_id = ?'
        i: int = await execute_query(conn, query_text, (query.from_user.id,))

        if i > len(quests_dict['questions']) - 1:
            query_text = f'SELECT r_q_num FROM user_data WHERE user_id = ?'
            score: int = await execute_query(conn, query_text, (query.from_user.id,))

            keyboard = [
                [InlineKeyboardButton("Go back.", callback_data="go back")],
                [InlineKeyboardButton("Play again.", callback_data="play again")]
            ]

            await query.edit_message_text(
                text=f"Your score is {score}. {'Congratulations!' if score > 5 else 'Try again.'}",
                reply_markup=InlineKeyboardMarkup(keyboard))

        else:
            quest_data = quests_dict['questions'][i]

            keyboard = [
                [InlineKeyboardButton(var, callback_data=f"{i} quest, {j} var")]
                for j, var in enumerate(quest_data['options'])
            ]

            await query.edit_message_text(text=quest_data['question'], reply_markup=InlineKeyboardMarkup(keyboard))

            # change the num of question for user in user_data table
            await update_cols(conn, query.from_user.id, ['q_num'], [i + 1])


async def animation_echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_document(chat_id=update.effective_chat.id, document=update.message.sticker.file_id)


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text=update.message.text)


async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, I didn't understand that command.")


async def end(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Returns `ConversationHandler.END`, which tells the
    ConversationHandler that the conversation is over.
    """
    query = update.callback_query
    await query.answer()

    await query.edit_message_text(text="See you next time!")
    return ConversationHandler.END


async def set_bot_webhook(application):
    whook = os.getenv('URL')
    webhook_url = f'https://api.telegram.org/bot{bot_token}/setWebhook?url=https://{whook}/'

    await application.bot.set_webhook(url=webhook_url, allowed_updates=Update.ALL_TYPES)


async def main():
    conn = await create_connection(db_dir)
    await create_table(conn)

    # Stages
    # START_ROUTES, END_ROUTES, QUEST_ROUTES, WAITING_ROUTES = range(4)
    # Callback data
    ONE, TWO, THREE, FOUR = range(4)

    application = ApplicationBuilder().token(bot_token).build()

    photo_handler = MessageHandler(filters.PHOTO & (~ filters.FORWARDED), photo_choose)
    back_handler = CallbackQueryHandler(back_button)
    quest_handler = CallbackQueryHandler(quest)
    unknown_handler = MessageHandler(filters.COMMAND, unknown)
    echo_handler = MessageHandler(filters.TEXT & (~ filters.FORWARDED), echo)

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            START_ROUTES: [
                photo_handler,
                back_handler,
                unknown_handler
            ],
            QUEST_ROUTES: [quest_handler],
            WAITING_ROUTES: [echo_handler, animation_echo],
            END_ROUTES: [
                CallbackQueryHandler(end, pattern="^" + str(TWO) + "$"),
                CallbackQueryHandler(start, pattern="^" + str(THREE) + "$"),
            ],
        },
        fallbacks=[CommandHandler("start", start)],
    )

    # Add ConversationHandler to application that will be used for handling updates
    application.add_handler(conv_handler)

    # Pass webhook settings to telegram
    await set_bot_webhook(application)

    # Set up webserver
    app = FastAPI()

    @app.post("/")
    async def telegram(request: Request) -> Response:
        """Handle incoming Telegram updates by putting them into the `update_queue`"""
        print(await request.json())
        await application.update_queue.put(Update.de_json(data=await request.json(), bot=application.bot))

        return Response(status_code=HTTPStatus.OK)

    # Run application and webserver together
    webserver = uvicorn.Server(
        config=uvicorn.Config(
            app=app,
            port=PORT,
            # use_colors=False,
            host="0.0.0.0",
            log_level="info",
            reload=True
        )
    )

    # Run application and webserver together

    # https://github.com/python-telegram-bot/python-telegram-bot/wiki/Frequently-requested-design-patterns#running-ptb-alongside-other-asyncio-frameworks
    # https://github.com/python-telegram-bot/python-telegram-bot/issues/3410
    # https://github.com/python-telegram-bot/python-telegram-bot/wiki/Webhooks
    # like here: https://github.com/python-telegram-bot/python-telegram-bot/blob/v13.x/examples/rawapibot.py
    # https://github.com/alex-sherman/unsync/

    async with application:
        # await application.initialize()
        await application.start()
        # await application.updater.start_polling()
        await webserver.serve()

        # Stop the other asyncio frameworks here
        # await application.updater.stop()
        await application.stop()
        # await application.shutdown()

        logger.info('Bot stopped.')

if __name__ == '__main__':
    """
    Some instructions (precisely more for me to don't forget the right steps):
    
    First, get the link of https webhook in the site (run the command from it: ssh -R 80:localhost:8080 localhost.run)
    and put it to the 'URL' in .env file
    Second, set the webhook in the set_webhook.py file (just run it) (optional, already write the func for it)
    And, finally, run the bot.py file (this file) 
    """
    START_ROUTES, END_ROUTES, QUEST_ROUTES, WAITING_ROUTES = range(4)

    load_dotenv()
    bot_token = os.getenv("BOT_TOKEN")
    admin_id = int(os.getenv("ADMIN_ID"))

    db_dir = os.path.normpath(os.getcwd() + os.getenv("DB_DIR"))
    gallery_dir_path = os.path.normpath(os.getcwd() + os.getenv("GALLERY_DIR_PATH"))
    quests_path = os.path.normpath(os.getcwd() + os.getenv("QUESTS_PATH"))
    swap_type = os.getenv("SWAP_TYPE")

    PORT = 8080
    logger.info('Start the bot (run main.py).')
    asyncio.run(main())
