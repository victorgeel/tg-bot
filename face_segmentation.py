# Base code were taken from here:
# https://github.com/willyfh/farl-face-segmentation, https://github.com/FacePerceiver/facer
# and modified for my needs

from bot_config import *
import asyncio
import numpy as np
import cv2

import torch
import functools
import facer
from facer.transform import (get_face_align_matrix, make_inverted_tanh_warp_grid,
                                        make_tanh_warp_grid)

# this is to map each of the face parsing label for face segmentation label > 0: background, 1: face
face_label_mapping = {
    'background': 0,
    'neck': 0,
    'face': 1,
    'cloth': 0,
    'rr': 0,
    'lr': 0,
    'rb': 1,
    'lb': 1,
    're': 1,
    'le': 1,
    'nose': 1,
    'imouth': 1,
    'llip': 1,
    'ulip': 1,
    'hair': 0,
    'eyeg': 1,
    'hat': 0,
    'earr': 0,
    'neckl': 0
}

facer.face_parsing.farl.pretrain_settings['celebm/448'] = {
    'url': [
        'https://github.com/FacePerceiver/facer/releases/download/models-v1/face_parsing.farl.celebm.main_ema_181500_jit.pt',
    ],
    'matrix_src_tag': 'points',
    'get_matrix_fn': functools.partial(get_face_align_matrix,
                                       target_shape=(448, 448), target_face_scale=0.8),
    'get_grid_fn': functools.partial(make_tanh_warp_grid,
                                     warp_factor=0.0, warped_shape=(448, 448)),
    'get_inv_grid_fn': functools.partial(make_inverted_tanh_warp_grid,
                                         warp_factor=0.0, warped_shape=(448, 448)),
    'label_names': ['background', 'neck', 'face', 'cloth', 'rr', 'lr', 'rb', 'lb', 're',
                    'le', 'nose', 'imouth', 'llip', 'ulip', 'hair',
                    'eyeg', 'hat', 'earr', 'neckl']
}


async def crop_face(input_image: np.array, mask_image: np.array) -> np.array:
    cropped_image = cv2.multiply(np.array(input_image), (mask_image / 255).astype(np.uint8))
    return cropped_image


async def map_face_label(seg_labels: torch.Tensor, data: dict[str, torch.Tensor]):
    """
    Map the face parsing label to face segmentation label (0: background, 1: face)
    """
    for i in range(len(seg_labels)):
        for j in range(len(seg_labels[i])):
            seg_labels[i][j] = face_label_mapping[data['seg']['label_names'][seg_labels[i][j]]]


async def extract_face_mask(data: dict[str, torch.Tensor]) -> np.array:
    seg_logits = data['seg']['logits']
    seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w

    predicted_labels = seg_probs.argmax(dim=1).int()

    await map_face_label(predicted_labels[0], data)

    mask_image = (predicted_labels * 255)
    mask_image = mask_image.permute(1, 2, 0)  # c x h x w -> h x w x c
    mask_image = mask_image.repeat(1, 1, 3).permute(2, 0, 1)  # h x w x c -> c x h x w
    mask_image = mask_image.to(torch.uint8).cpu().numpy()
    return mask_image


async def segment_input_image(input_image: np.array, mask_image: np.array, face_color: list[int]) -> (np.array, np.array):
    # non_face_image = cv2.multiply(np.array(input_image), (1 - (mask_image / 255)).astype(np.uint8))
    cropped_face_image = await crop_face(input_image, mask_image)
    # seg_image = cv2.addWeighted(cropped_face_image, 0.5, ((mask_image / 255) * face_color).astype(np.uint8), 0.5, 0)
    # seg_image = seg_image + non_face_image
    return cropped_face_image


async def mask_extractor(image: torch.Tensor, img_type: str) -> tuple[np.array, torch.Tensor]:
    """
    Extract the face mask, facial landmarks and cropped image of the face from the input image
    :param img_type:
    :param image:
    :return:
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # convert the image
    # image = facer.hwc2bchw(image).to(device=device)  # image: 1 x 3 x h x w
    image = image.unsqueeze(0)

    # load the pretrained models
    face_detector = facer.face_detector("retinaface/resnet50", device=device)
    face_parser = facer.face_parser("farl/celebm/448", device=device)
    face_aligner = facer.face_aligner('farl/ibug300w/448',
                                      device=device)  # optional: "farl/wflw/448", "farl/aflw19/448"

    with torch.no_grad():
        det_faces: dict = await asyncio.to_thread(face_detector, image)
        if not bool(det_faces):
            raise FaceNotFoundError('Faces not found')

        # choose the best face based on a prediction score
        best_id = torch.argmax(det_faces['scores']).item()
        det_faces = {key: value[best_id].unsqueeze(0) for key, value in det_faces.items()}

        parser_faces_task = asyncio.to_thread(face_parser, image, det_faces)
        faces_landmarks_task = asyncio.to_thread(face_aligner, image, det_faces)

        faces, faces_landmarks = await asyncio.gather(parser_faces_task, faces_landmarks_task)

    facial_keypoints = faces_landmarks['alignment']

    if img_type == 'tgt':
        mask_image = await asyncio.create_task(extract_face_mask(faces))

        # task = asyncio.create_task(segment_input_image(np.array(image[0]), mask_image, [255, 129, 54]))
        # cropped_face_image = await task

        return mask_image, facial_keypoints

    elif img_type == 'src':
        return facial_keypoints
