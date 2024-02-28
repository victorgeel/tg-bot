import shutil
from pathlib import Path

from bot_config import *
from img_processing import *
import io
from telegram import InputMediaPhoto
from dotenv import load_dotenv
import os
import subprocess
from icecream import ic


async def send_final_img(img: str | np.ndarray, bot, user_id, conn, update):
    if not isinstance(img, str):
        img_pil = Image.fromarray(img.astype('uint8'), 'RGB')
    else:
        img_pil = Image.open(img)

    buf = io.BytesIO()
    img_pil.save(buf, format='JPEG')
    bimg_pil = buf.getvalue()

    await bot.send_message(chat_id=user_id, text=f'Your photo is ready.')

    media = [InputMediaPhoto(media=bimg_pil, has_spoiler=True)]
    await bot.send_media_group(chat_id=update.effective_chat.id,
                               media=media
                               )

    await update_cols(conn, user_id, ['final_photo'], [bimg_pil])


async def mask_swap(bot, conn, update, user_id):
    img_prep_task = asyncio.create_task(img_processing(user_id, conn))
    # AttributeError: 'coroutine' object has no attribute 'astype' ...
    # img_prep_task = asyncio.to_thread(img_processing, user_id, conn)

    try:
        await asyncio.sleep(2)  # for edit the message after preparing photos
        final_result = await img_prep_task
        # ic(final_result)

    except FaceNotFoundError as e:
        await bot.send_message(chat_id=user_id,
                               text=f'{e}. Please, try to download a new selfie.'
                               )
    else:
        await send_final_img(final_result, bot, user_id, conn, update)


async def sim_swap(bot, conn, update, user_id: int):
    """
    This type of face swapping is based on the SimSwap repo (https://github.com/neuralchen/SimSwap/tree/main).
    Although it doesn't have class implementation for own images usage (only by bash script), it's possible to use it
    but with some strong (mostly bad) changes :(.
    :param bot:
    :param conn:
    :param update:
    :param user_id:
    :return:
    """
    # check if SimSwap folder exists
    if not os.path.exists(os.path.normpath(os.getcwd() + '/SimSwap')):
        # Execute the bash script to setup SimSwap
        subprocess.run(['bash', os.path.normpath(os.getcwd() + '/scripts/setup_simswap.sh')])

    # create user folder named by its id
    user_fol_path = os.path.normpath(os.getcwd() + f'/imgs/user_imgs/{user_id}')
    user_folder = Path(user_fol_path)
    user_folder.mkdir(parents=True, exist_ok=True)

    # get two user photos
    bphoto_1, bphoto_2 = await get_bphotos(conn, user_id)

    img1 = Image.open(BytesIO(bphoto_1))
    img2 = Image.open(BytesIO(bphoto_2))

    img1.save(os.path.join(user_fol_path, 'image1.png'))
    img2.save(os.path.join(user_fol_path, 'image2.png'))

    try:
        # https://stackoverflow.com/questions/299446/how-do-i-change-directory-back-to-my-original-working-directory-with-python
        dir_path = os.path.normpath(os.getcwd() + '/SimSwap')
        cmd = f'python test_wholeimage_swapsingle.py --crop_size 224 --use_mask --name people --Arc_path arcface_model/arcface_checkpoint.tar --pic_a_path {user_fol_path}/image1.png --pic_b_path {user_fol_path}/image2.png --output_path {user_fol_path}/ --no_simswaplogo'
        subprocess.run(f'{cmd}', shell=True, capture_output=True, cwd=dir_path)
    except Exception as e:
        await bot.send_message(chat_id=user_id, text=f'Error in SimSwap model')
    else:
        res_path = os.path.join(user_fol_path, 'result_whole_swapsingle.jpg')

        photo_data = bytearray(open(res_path, 'rb').read())
        await update_cols(conn, user_id, ['final_photo'], [photo_data])

        await send_final_img(res_path, bot, user_id, conn, update)
    finally:
        # delete user folder after save and send final photo
        shutil.rmtree(user_fol_path)
