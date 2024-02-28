from face_segmentation import mask_extractor
from io import BytesIO
import torchvision.transforms as transforms
import asyncio
from PIL import Image
import cv2
import numpy as np
from user_db import *


def rectContains(rect, point):
    """
    Check if a point is inside a rectangle
    """

    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[0] + rect[2]:
        return False
    elif point[1] > rect[1] + rect[3]:
        return False
    return True


def calculateDelaunayTriangles(rect, points):
    """
    Calculate the Delaunay triangles
    """
    # create subdiv
    subdiv = cv2.Subdiv2D(rect)

    # Insert points into subdiv
    for p in points:
        subdiv.insert(tuple(p))

    triangle_list = subdiv.getTriangleList()
    delaunay_tri, pt = [], []

    for t in triangle_list:
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            ind = []
            # Get face-points (from 68 face detector) by coordinates
            for j in range(0, 3):
                for k in range(0, len(points)):
                    if abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0:
                        ind.append(k)
            # Three points form a triangle. Triangle array corresponds to the file tri.txt in FaceMorph
            if len(ind) == 3:
                delaunay_tri.append((ind[0], ind[1], ind[2]))

        pt = []

    return delaunay_tri


def applyAffineTransform(src, srcTri, dstTri, size):
    """
    Apply affine transform calculated using srcTri and dstTri to src and
    output an image of size.
    """

    # Given a pair of triangles, find the affine transform.
    warp_matrix = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warp_matrix, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst


def warpTriangle(img1, img2, t1, t2):
    """
    Warps and alpha blends triangular regions from img1 and img2 to img
    """

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1_rect, t2_rect, t2_rect_int = [], [], []

    for i in range(0, 3):
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2_rect_int.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    # img2_rect = np.zeros((r2[3], r2[2]), dtype=img1_rect.dtype)

    size = (r2[2], r2[3])

    img2_rect = applyAffineTransform(img1_rect, t1_rect, t2_rect, size)
    img2_rect = img2_rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
            (1.0, 1.0, 1.0) - mask)

    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2_rect


def get_result(image_src, image_tgt, result_1, result_2):
    facial_keypoints_src = result_1
    mask_image_tgt, facial_keypoints_tgt = result_2

    # https://learnopencv.com/face-swap-using-opencv-c-python/#download
    points_tgt = facial_keypoints_tgt.detach().numpy()[0, :, :]
    points_src = facial_keypoints_src.detach().numpy()[0, :, :]

    hull_index_tgt = cv2.convexHull(points_tgt, returnPoints=False)

    points_src, points_tgt = points_src.tolist(), points_tgt.tolist()

    hull_src, hull_tgt = [], []
    img1_warped = np.copy(image_tgt.numpy().transpose(1, 2, 0))

    for i in range(0, len(hull_index_tgt)):
        hull_src.append(tuple(points_src[int(hull_index_tgt[i])]))
        hull_tgt.append(tuple(points_tgt[int(hull_index_tgt[i])]))

    # Find delanauy traingulation for convex hull points
    size_img_tgt = image_tgt.shape
    rect = (0, 0, size_img_tgt[-1], size_img_tgt[1])

    dt = calculateDelaunayTriangles(rect, hull_tgt)

    if len(dt) == 0:
        return image_tgt

    # Apply affine transformation to Delaunay triangles
    for i in range(0, len(dt)):
        t1, t2 = [], []

        # get points for image_src, image_tgt corresponding to the triangles
        for j in range(0, 3):
            t1.append(hull_src[dt[i][j]])
            t2.append(hull_tgt[dt[i][j]])

        warpTriangle(image_src.numpy().transpose(1, 2, 0), img1_warped, t1, t2)

    r = cv2.boundingRect(np.float32([hull_tgt]))
    center = (r[0] + int(r[2] / 2), r[1] + int(r[3] / 2))

    # https://learnopencv.com/face-morph-using-opencv-cpp-python/
    # https://learnopencv.com/seamless-cloning-using-opencv-python-cpp/
    # https://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf

    # Clone seamlessly.
    output = cv2.seamlessClone(np.uint8(img1_warped), image_tgt.numpy().transpose(1, 2, 0),
                               mask_image_tgt.transpose(1, 2, 0), center, cv2.NORMAL_CLONE)
    return output


async def img_preporcessing(user_id: int, connection):
    """
    Here we send the photos to the processing model and get the results (mask, facial landmarks and crop face).

    :param connection:
    :param user_id:
    :return:
    """

    # q_text_1 = f'SELECT photo_1 FROM user_data WHERE user_id = ?'
    # q_text_2 = f'SELECT photo_2 FROM user_data WHERE user_id = ?'
    #
    # await asyncio.sleep(2)
    # bphoto_1 = await execute_query(connection, q_text_1, (user_id,))
    # bphoto_2 = await execute_query(connection, q_text_2, (user_id,))

    bphoto_1, bphoto_2 = await get_bphotos(connection, user_id)

    transform = transforms.Compose([
        transforms.PILToTensor()
    ])

    await asyncio.sleep(2)

    torch_photo_1 = transform(Image.open(BytesIO(bphoto_1)))
    torch_photo_2 = transform(Image.open(BytesIO(bphoto_2)))

    # https://docs-python.ru/standart-library/modul-asyncio-python/funktsija-to-thread-modulja-asyncio/

    await asyncio.sleep(2)
    mask_ext_task_1 = asyncio.create_task(mask_extractor(torch_photo_1, 'src'))
    mask_ext_task_2 = asyncio.create_task(mask_extractor(torch_photo_2, 'tgt'))

    # https://stackoverflow.com/questions/59073556/how-to-cancel-all-remaining-tasks-in-gather-if-one-fails
    # https://plainenglish.io/blog/how-to-manage-exceptions-when-waiting-on-multiple-asyncio#exceptions-in-asynciogather
    result_1, result_2 = await asyncio.gather(mask_ext_task_1, mask_ext_task_2, return_exceptions=False)

    return result_1, result_2, torch_photo_1, torch_photo_2


async def img_processing(user_id: int, cursor):
    """
    The main function that runs the whole pipeline of user's photos processing.
    :param user_id:
    :param cursor:
    :return:
    """
    await asyncio.sleep(2)  # for edit the message after preparing photos

    img_prep_task = asyncio.create_task(img_preporcessing(user_id, cursor))
    result_1, result_2, image_src, image_tgt = await img_prep_task

    result = await asyncio.to_thread(get_result, image_src, image_tgt, result_1, result_2)

    return result

# # for debugging
# from dotenv import load_dotenv
# import os
#
# load_dotenv()
#
# db_dir = f".{os.getenv('DB_DIR')}"
# admin_id = int(os.getenv('ADMIN_ID'))
#
#
# async def main():
#     conn = await aiosqlite.connect(db_dir, check_same_thread=False)
#     cursor = await conn.cursor()
#
#     img_prep_task = asyncio.create_task(img_processing(admin_id, cursor))
#     final_result = await img_prep_task
#
#     print(final_result)
#
# asyncio.run(main())
