import torch
import numpy as np

def calc_area(boxes :torch.Tensor):
    return torch.clamp(boxes[..., 2] - boxes[..., 0], min=0.0) \
           * torch.clamp(boxes[..., 3] - boxes[..., 1], min=0.0)

def compute_iou(anchors :torch.Tensor,
                gt :torch.Tensor):
    '''
        :param anchors: anchors in all feature map [na, 4] >> 4=(x1, y1, x2, y2)
        :param gt: all target [nt, 4] >> 4=(x1, y1, x2, y2)
        :return: shape : [na, nt]
                 each index indicate iou score [anchor, gt] pair
                 ex) [1, 2] => iou between anchor1 and target2
    '''
    anchors = anchors.unsqueeze(1)
    gt = gt.unsqueeze(0)

    ## 1) get intersection area
    left_top = torch.max(anchors[..., :2], gt[..., :2])
    right_bottom = torch.min(anchors[..., 2:], gt[..., 2:])
    intersec_box = torch.cat([left_top, right_bottom], dim=-1) #[na, 4]. [max_x1, max_y1, min_x2, min_y2]
    intersec_area = calc_area(intersec_box)

    ## 2) get union area
    anchors_area = calc_area(anchors)
    gt_area = calc_area(gt)
    union = anchors_area + gt_area - intersec_area

    ## 3) get iou
    iou = intersec_area / union

    return iou

def coco_to_conner_form(boxes: list):
    '''
    :param boxes: [n, 4] shape >> 4=(x, y, w, h)
    :return: new_boxes : [n, 4] shape >> 4=(x1, y1, x2, y2)
    '''
    boxes = np.array(boxes)
    boxes = np.concatenate([boxes[..., :2],
                            boxes[..., :2] + boxes[..., 2:]],
                            axis=-1)

    return boxes.tolist()

def center_to_conner_form(boxes :torch.Tensor):
    '''
    :param boxes: [n, 4] shape >> 4=(cx, cy, w, h)
    :return: new_boxes : [n, 4] shape >> 4=(x1, y1, x2, y2)
    '''
    return torch.cat([boxes[..., :2] - boxes[..., 2:]/2,
                     boxes[..., :2] + boxes[..., 2:]/2],
                     dim=boxes.dim() - 1)

def conner_to_center_form(boxes: torch.Tensor):
    '''
    :param boxes: [n, 4] shape >> 4=(x1, y1, x2, y2)
    :return: new_boxes : [n, 4] shape >> 4=(cx, cy, w, h)
    '''
    wh = boxes[..., 2:] - boxes[..., :2]
    center_xy = boxes[..., :2] + wh/2

    return torch.cat([center_xy, wh], dim=boxes.dim() - 1)


def rescale_bbox(bboxes,
                 input_image_size,
                 output_image_size):
    '''
    :param bboxes: [num_obj, 4] = [x1, y1, x2, y2]
    :param model_image_size: (width, height) = (int, int)
    :param origin_image_size: (width, height) = (int, int)
    :return:
    '''
    if isinstance(input_image_size, int):
        input_image_size = [input_image_size, input_image_size]
    if isinstance(output_image_size, int):
        output_image_size = [output_image_size, output_image_size]

    # x_scale = origin_image_size[0] / model_image_size[0]
    # y_scale = origin_image_size[1] / model_image_size[1]
    x_scale = output_image_size[0] / input_image_size[0]
    y_scale = output_image_size[1] / input_image_size[1]

    bboxes[..., 0] = bboxes[..., 0] * float(x_scale)
    bboxes[..., 2] = bboxes[..., 2] * float(x_scale)
    bboxes[..., 1] = bboxes[..., 1] * float(y_scale)
    bboxes[..., 3] = bboxes[..., 3] * float(y_scale)

    return bboxes
