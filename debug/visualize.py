import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor
import random

label_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
              'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
              'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
              'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
              'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
              'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
              'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
              'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
              'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
              'toothbrush']

def draw_dataloader_output(images: list,
                           gt_loc: list,
                           gt_cls: list):
    color_map = ImageColor.colormap
    color_keys = list(color_map.keys())
    random.shuffle(color_keys)
    random_color_list = []
    for key in color_keys:
        random_color_list.append(key)

    for i in range(len(images)):
        image = images[i]
        bboxes = gt_loc[i]
        category_idxs = gt_cls[i]

        draw = ImageDraw.Draw(image)
        image_w, image_h = image.size
        draw_width = int((image_h + image_w) / 2 / 100)
        font_size = int(((image_h + image_w) / 2) / 30)

        font = ImageFont.truetype("../arial.ttf", font_size)

        for j in range(len(bboxes)):
            bbox = bboxes[j]
            x1, y1, x2, y2 = list(map(int, bbox))
            category_idx = int(category_idxs[j])
            if category_idx == -1:
                continue

            category = label_list[category_idx]
            color = random_color_list[category_idx % 148]
            offset = 1
            for k in range(0, draw_width):
                draw.rectangle((x1, y1, x2, y2), outline=color)
                x1 -= offset
                y1 -= offset
                x2 += offset
                y2 += offset
            rect_size = font_size
            draw.rectangle((x1, y1, x2, y1 + rect_size), fill='black')
            draw.text((x1, y1), category, fill='white', font=font)

        image.save('results/' + str(i) + '.jpg')
        del image
        del draw

def draw_anchor_output(images: list,
                       anchor_loc: list,
                       anchor_cls: list):
    color_map = ImageColor.colormap
    color_keys = list(color_map.keys())
    random.shuffle(color_keys)
    random_color_list = []
    for key in color_keys:
        random_color_list.append(key)

    for i in range(len(images)):
        image = images[i]
        bboxes = gt_loc[i]
        category_idxs = gt_cls[i]

        draw = ImageDraw.Draw(image)
        image_w, image_h = image.size
        draw_width = int((image_h + image_w) / 2 / 100)
        font_size = int(((image_h + image_w) / 2) / 30)

        font = ImageFont.truetype("../arial.ttf", font_size)

        for j in range(len(bboxes)):
            bbox = bboxes[j]
            x1, y1, x2, y2 = list(map(int, bbox))
            category_idx = int(category_idxs[j])
            if category_idx == -1:
                continue

            category = label_list[category_idx]
            color = random_color_list[category_idx % 148]
            offset = 1
            for k in range(0, draw_width):
                draw.rectangle((x1, y1, x2, y2), outline=color)
                x1 -= offset
                y1 -= offset
                x2 += offset
                y2 += offset
            rect_size = font_size
            draw.rectangle((x1, y1, x2, y1 + rect_size), fill='black')
            draw.text((x1, y1), category, fill='white', font=font)

        image.save('results/' + str(i) + '.jpg')
        del image
        del draw


def draw_bbox(image: Image,
              scores: list,
              categories: list,
              bboxes: list,
              label_names: list):
    color_map = ImageColor.colormap
    color_keys = list(color_map.keys())
    random.shuffle(color_keys)
    random_color_list = []
    for key in color_keys:
        random_color_list.append(key)

    draw = ImageDraw.Draw(image)
    image_w, image_h = image.size
    draw_width = int((image_h + image_w) / 2 / 100)
    font_size = int(((image_h + image_w) / 2) / 30)

    font = ImageFont.truetype("../arial.ttf", font_size)

    for i in range(len(bboxes)):
        score = scores[i]
        category_idx = int(categories[i])
        category = label_names[category_idx]
        bbox = bboxes[i]
        x1, y1, x2, y2 = list(map(int, bbox))

        color = random_color_list[category_idx % 148]
        offset = 1
        for i in range(0, draw_width):
            draw.rectangle((x1, y1, x2, y2), outline=color)
            x1 -= offset
            y1 -= offset
            x2 += offset
            y2 += offset
        rect_size = font_size
        draw.rectangle((x1, y1, x2, y1+rect_size), fill='black')
        draw.text((x1, y1), category, fill='white', font=font)

    return image


def draw_image(images: list,
               scores: list,
               categories: list,
               boxes: list):
    for i in range(len(scores)):
        image = images[i]
        score = scores[i]
        category = categories[i]
        box = boxes[i]

