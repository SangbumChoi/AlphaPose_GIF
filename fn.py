import torch
import re
import os
import collections
from torch._six import string_classes, int_classes
import cv2
from opt import opt
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import copy
from put_gif import put_gif
import statistics

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)

numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}

_use_shared_memory = True


def collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])

    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [collate_fn(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


def collate_fn_list(batch):
    img, inp, im_name = zip(*batch)
    img = collate_fn(img)
    im_name = collate_fn(im_name)

    return img, inp, im_name


def vis_frame_fast(frame, im_res, format='coco'):
    '''
    frame: frame image
    im_res: im_res of predictions
    format: coco or mpii

    return rendered image
    '''
    if format == 'coco':
        l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (17, 11), (17, 12),  # Body
            (11, 13), (12, 14), (13, 15), (14, 16)
        ]
        p_color = [(0, 255, 255), (0, 191, 255),(0, 255, 102),(0, 77, 255), (0, 255, 0), #Nose, LEye, REye, LEar, REar
                    (77,255,255), (77, 255, 204), (77,204,255), (191, 255, 77), (77,191,255), (191, 255, 77), #LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                    (204,77,255), (77,255,204), (191,77,255), (77,255,191), (127,77,255), (77,255,127), (0, 255, 255)] #LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
        line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50), 
                    (77,255,222), (77,196,255), (77,135,255), (191,255,77), (77,255,77), 
                    (77,222,255), (255,156,127), 
                    (0,127,255), (255,127,77), (0,77,255), (255,77,36)]
    elif format == 'mpii':
        l_pair = [
            (8, 9), (11, 12), (11, 10), (2, 1), (1, 0),
            (13, 14), (14, 15), (3, 4), (4, 5),
            (8, 7), (7, 6), (6, 2), (6, 3), (8, 12), (8, 13)
        ]
        p_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, PURPLE, RED, RED,BLUE,BLUE]
    else:
        NotImplementedError

    im_name = im_res['imgname'].split('/')[-1]
    img = frame
    for human in im_res['result']:
        part_line = {}
        kp_preds = human['keypoints']
        kp_scores = human['kp_score']
        kp_preds = torch.cat((kp_preds, torch.unsqueeze((kp_preds[5,:]+kp_preds[6,:])/2,0)))
        kp_scores = torch.cat((kp_scores, torch.unsqueeze((kp_scores[5,:]+kp_scores[6,:])/2,0)))
        # Draw keypoints
        for n in range(kp_scores.shape[0]):
            if kp_scores[n] <= 0.05:
                continue
            cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
            part_line[n] = (cor_x, cor_y)
            cv2.circle(img, (cor_x, cor_y), 4, p_color[n], -1)
        # Draw limbs
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                cv2.line(img, start_xy, end_xy, line_color[i], 2*(kp_scores[start_p] + kp_scores[end_p]) + 1)
    return img


def vis_frame(frame, im_res, format='coco'):
    '''
    frame: frame image
    im_res: im_res of predictions
    format: coco or mpii

    return rendered image
    '''
    if format == 'coco':
        l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (17, 11), (17, 12),  # Body
            (11, 13), (12, 14), (13, 15), (14, 16)
        ]

        p_color = [(0, 255, 255), (0, 191, 255),(0, 255, 102),(0, 77, 255), (0, 255, 0), #Nose, LEye, REye, LEar, REar
                    (77,255,255), (77, 255, 204), (77,204,255), (191, 255, 77), (77,191,255), (191, 255, 77), #LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                    (204,77,255), (77,255,204), (191,77,255), (77,255,191), (127,77,255), (77,255,127), (0, 255, 255)] #LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
        line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50), 
                    (77,255,222), (77,196,255), (77,135,255), (191,255,77), (77,255,77), 
                    (77,222,255), (255,156,127), 
                    (0,127,255), (255,127,77), (0,77,255), (255,77,36)]
    elif format == 'mpii':
        l_pair = [
            (8, 9), (11, 12), (11, 10), (2, 1), (1, 0),
            (13, 14), (14, 15), (3, 4), (4, 5),
            (8, 7), (7, 6), (6, 2), (6, 3), (8, 12), (8, 13)
        ]
        p_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, PURPLE, RED, RED, BLUE, BLUE]
        line_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, RED, RED, BLUE, BLUE]
    else:
        raise NotImplementedError

    im_name = im_res['imgname'].split('/')[-1]
    img = frame
    height,width = img.shape[:2]
    img = cv2.resize(img,(int(width/2), int(height/2)))
    for human in im_res['result']:
        part_line = {}
        kp_preds = human['keypoints']
        kp_scores = human['kp_score']
        kp_preds = torch.cat((kp_preds, torch.unsqueeze((kp_preds[5,:]+kp_preds[6,:])/2,0)))
        kp_scores = torch.cat((kp_scores, torch.unsqueeze((kp_scores[5,:]+kp_scores[6,:])/2,0)))
        # Draw keypoints
        for n in range(kp_scores.shape[0]):
            if kp_scores[n] <= 0.05:
                continue
            cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
            part_line[n] = (int(cor_x/2), int(cor_y/2))
            bg = img.copy()
#            print(part_line[n][1])
#            cv2.circle(bg, (int(cor_x/2), int(cor_y/2)), 2, p_color[n], -1)
            # Now create a mask of logo and create its inverse mask also
#            transparency = max(0, min(1, kp_scores[n]))
#            img = cv2.addWeighted(bg, transparency, img, 1-transparency, 0)
#            if n==1:
        try:

#	both chin 25 is left part 26 is right (Example: Face chin)
            part_line[25] = (statistics.mean([part_line[0][0],part_line[3][0]]), statistics.mean([part_line[0][1],part_line[3][1]]))
            part_line[26] = (statistics.mean([part_line[0][0],part_line[4][0]]), statistics.mean([part_line[0][1],part_line[4][1]]))

#	middle of eyes (Example: Sunglasses)
            part_line[18] = (statistics.mean([part_line[1][0], part_line[2][0]]), statistics.mean([part_line[1][1],part_line[2][1]]))

#	mean of ears
            part_line[19] = (statistics.mean([part_line[3][0], part_line[4][0]]), statistics.mean([part_line[3][1],part_line[4][1]]))

            part_line[20] = (part_line[18][0]-part_line[0][0], part_line[18][1]-part_line[0][1])
#	expected head, mouth (Example: Crowns for 21 Fire for 31)
            part_line[21] = (part_line[0][0]+5*part_line[20][0], part_line[0][1]+5*part_line[20][1])
            part_line[31] = (part_line[0][0]+6*part_line[20][0], part_line[0][1]+6*part_line[20][1])
            part_line[23] = (part_line[0][0]-2*part_line[20][0], part_line[0][1]-2*part_line[20][1])

#	hip part calulation
            part_line[40] = (part_line[11][0]-part_line[12][0], part_line[11][1]-part_line[12][1])
            part_line[41] = (part_line[11][0]+2*part_line[40][0], part_line[11][1]+2*part_line[40][1])
            part_line[42] = (part_line[11][0]+part_line[40][0], part_line[11][1]+part_line[40][1])
            part_line[43] = (part_line[12][0]-2*part_line[40][0], part_line[12][1]-2*part_line[40][1])
            part_line[44] = (part_line[12][0]-part_line[40][0], part_line[12][1]-part_line[40][1])

#	expected static
            part_line[50] = (int(3*width/7), int(height/4))
            part_line[51] = (int(width/8), int(height/4))
            part_line[52] = (int(width/4), int(height/4))
            part_line[53] = (int(3*width/8), int(height/4))

            part_line[54] = (int(width/6.2), int(height/9.157))
            part_line[55] = (int(width/3.326), int(height/58))

            part_line[56] = (int(width/6.514), int(height/6.842))
            part_line[57] = (int(width/3.406), int(height/8.533))
            part_line[58] = (int(width/2.319), int(height/9.846))

            part_line[59] = (int(width/4.24), int(height/3.052))
            part_line[60] = (int(width/3.212), int(height/2.74))

#	2~3.5
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/devil_mask.gif', 15, 2, 64, 112)
            img = put_gif(im_name, img, part_line, 0, 1, 2, scale, replay_speed, start, end, file_name)
#	3.5~5
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/money2_SA.gif', 3, 2, 112, 160)
            img = put_gif(im_name, img, part_line, 1, 1, 2, scale, replay_speed, start, end, file_name)
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/money2_SA.gif', 3, 2, 112, 160)
            img = put_gif(im_name, img, part_line, 2, 1, 2, scale, replay_speed, start, end, file_name)
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/money2_SA.gif', 3, 2, 112, 160)
            img = put_gif(im_name, img, part_line, 2, 1, 2, scale, replay_speed, start, end, file_name)
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/moneyrain_crop.gif', 130, 1, 112, 160)
            img = put_gif(im_name, img, part_line, 51, 51, 51, scale, replay_speed, start, end, file_name)
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/moneyrain_crop.gif', 130, 1, 128, 160)
            img = put_gif(im_name, img, part_line, 52, 52, 52, scale, replay_speed, start, end, file_name)
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/moneyrain_crop.gif', 130, 1, 144, 160)
            img = put_gif(im_name, img, part_line, 53, 53, 53, scale, replay_speed, start, end, file_name)


#	5~6
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/spark2.gif', 20, 1, 174, 189)
            img = put_gif(im_name, img, part_line, 10, 10, 10, scale, replay_speed, start, end, file_name)
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/flame_edited.gif', 20, 1, 189, 207)
            img = put_gif(im_name, img, part_line, 10, 10, 10, scale, replay_speed, start, end, file_name)
#	6~7
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/angel_ring.gif', 25, 1, 223, 233)
            img = put_gif(im_name, img, part_line, 31, 31, 31, scale, replay_speed, start, end, file_name)
#	7~9
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/sunglass.gif', 7, 2, 240, 257)
            img = put_gif(im_name, img, part_line, 0, 1, 2, scale, replay_speed, start, end, file_name)
#	9~11
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/skull_yellow.gif', 7, 3, 277, 329)
            img = put_gif(im_name, img, part_line, 0, 1, 2, scale, replay_speed, start, end, file_name)
#	11~13
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/bear_Line.gif', 78, 2, 351, 401)
            img = put_gif(im_name, img, part_line, 51, 51, 51, scale, replay_speed, start, end, file_name)
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/bear_Line.gif', 78, 2, 351, 401)
            img = put_gif(im_name, img, part_line, 50, 50, 50, scale, replay_speed, start, end, file_name)
#	13~14
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/spark2.gif', 20, 1, 406, 432)
            img = put_gif(im_name, img, part_line, 9, 9, 9, scale, replay_speed, start, end, file_name)
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/spark2.gif', 20, 1, 406, 432)
            img = put_gif(im_name, img, part_line, 10, 10, 10, scale, replay_speed, start, end, file_name)
#	14~15
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/mad_rabbit.gif', 15, 2, 446, 467)
            img = put_gif(im_name, img, part_line, 18, 1, 2, scale, replay_speed, start, end, file_name)
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/triple_scratch2.gif', 15, 2, 446, 467)
            img = put_gif(im_name, img, part_line, 41, 1, 2, scale, replay_speed, start, end, file_name)
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/triple_scratch2_mirrored.gif', 15, 2, 446, 467)
            img = put_gif(im_name, img, part_line, 43, 1, 2, scale, replay_speed, start, end, file_name)
#	15~16
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/smoke.gif', 10, 2, 473, 488)
            img = put_gif(im_name, img, part_line, 54, 1, 2, scale, replay_speed, start, end, file_name)
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/smoke.gif', 10, 2, 473, 488)
            img = put_gif(im_name, img, part_line, 55, 1, 2, scale, replay_speed, start, end, file_name)
#	16~17
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/wing_1.gif', 25, 1, 483, 511)
            img = put_gif(im_name, img, part_line, 31, 5, 6, scale, replay_speed, start, end, file_name)
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/money_edited.gif', 2, 2, 483, 511)
            img = put_gif(im_name, img, part_line, 1, 1, 2, scale, replay_speed, start, end, file_name)
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/money_edited.gif', 2, 2, 483, 511)
            img = put_gif(im_name, img, part_line, 2, 1, 2, scale, replay_speed, start, end, file_name)
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/spark.gif', 10, 2, 483, 511)
            img = put_gif(im_name, img, part_line, 9, 9, 9, scale, replay_speed, start, end, file_name)
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/spark.gif', 10, 2, 483, 511)
            img = put_gif(im_name, img, part_line, 10, 10, 10, scale, replay_speed, start, end, file_name)
#	17~18
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/X2.gif', 5, 2, 511, 564)
            img = put_gif(im_name, img, part_line, 1, 1, 2, scale, replay_speed, start, end, file_name)
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/X2.gif', 5, 2, 511, 564)
            img = put_gif(im_name, img, part_line, 2, 1, 2, scale, replay_speed, start, end, file_name)
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/crown.gif', 13, 2, 511, 533)
            img = put_gif(im_name, img, part_line, 31, 1, 2, scale, replay_speed, start, end, file_name)
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/spread_down.gif', 13, 2, 540, 550)
            img = put_gif(im_name, img, part_line, 10, 10, 8, scale, replay_speed, start, end, file_name)
#	18~19
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/increase.gif', 30, 2, 564, 579)
            img = put_gif(im_name, img, part_line, 59, 59, 59, scale, replay_speed, start, end, file_name)
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/increase.gif', 30, 2, 564, 579)
            img = put_gif(im_name, img, part_line, 60, 60, 60, scale, replay_speed, start, end, file_name)
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/flame2.gif', 7, 2, 564, 579)
            img = put_gif(im_name, img, part_line, 31, 31, 31, scale, replay_speed, start, end, file_name)
#	19~20
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/devil_tail.gif', 10, 2, 579, 603)
            img = put_gif(im_name, img, part_line, 42, 11, 12, scale, replay_speed, start, end, file_name)
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/fixed_monster_1.gif', 16, 2, 593, 616)
            img = put_gif(im_name, img, part_line, 56, 56, 56, scale, replay_speed, start, end, file_name)
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/fixed_monster_2.gif', 16, 2, 598, 616)
            img = put_gif(im_name, img, part_line, 57, 57, 57, scale, replay_speed, start, end, file_name)
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/fixed_monster_3.gif', 16, 2, 602, 616)
            img = put_gif(im_name, img, part_line, 58, 58, 58, scale, replay_speed, start, end, file_name)

#	20~20
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/pop_text.gif', 3, 2, 610, 626)
            img = put_gif(im_name, img, part_line, 10, 10, 10, scale, replay_speed, start, end, file_name)
#	20~22
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/spark.gif', 10, 2, 644, 653)
            img = put_gif(im_name, img, part_line, 9, 9, 9, scale, replay_speed, start, end, file_name)
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/spark.gif', 10, 2, 644, 653)
            img = put_gif(im_name, img, part_line, 10, 10, 10, scale, replay_speed, start, end, file_name)
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/X.gif', 5, 1, 625, 666)
            img = put_gif(im_name, img, part_line, 1, 1, 2, scale, replay_speed, start, end, file_name)
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/X.gif', 5, 1, 625, 666)
            img = put_gif(im_name, img, part_line, 2, 1, 2, scale, replay_speed, start, end, file_name)
#	22~23
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/flame_edited.gif', 13, 1, 678, 689)
            img = put_gif(im_name, img, part_line, 10, 10, 10, scale, replay_speed, start, end, file_name)
#	end
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/error.gif', 100, 1, 696, 736)
            img = put_gif(im_name, img, part_line, 51, 51, 51, scale, replay_speed, start, end, file_name)
            (file_name, scale, replay_speed, start, end) = ('examples/gif/YG/error.gif', 100, 1, 696, 736)
            img = put_gif(im_name, img, part_line, 53, 53, 53, scale, replay_speed, start, end, file_name)

        except KeyError:
            img = bg
    img = cv2.resize(img,(width,height),interpolation=cv2.INTER_CUBIC)
    return img


def getTime(time1=0):
    if not time1:
        return time.time()
    else:
        interval = time.time() - time1
        return time.time(), interval
