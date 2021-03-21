from ctypes import *
import math
import matplotlib.pyplot as plt
from utils import *
import numpy as np
from itertools import combinations
import random
import os
import cv2
import time
import darknet
import argparse

#=========== Only use one parser ===========#

#----------- Arguments für Tiny -----------#
def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default="./Zoo_Test_Video.mp4",
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--out_filename", type=str, default="",
                        help="inference video name. Not saved if empty")
    parser.add_argument("--weights", default="./backup/yolov4-tiny-zoo_4000.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="./cfg/yolov4-tiny-zoo.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./data/zoo.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with confidence below this value")
    parser.add_argument("--max_distance", type=int, default=200,
                        help="maximum pixel distance at what close proximity is indicated")
    return parser.parse_args()

#----------- Arguments für Fullsize -----------#

# def parser():
#     parser = argparse.ArgumentParser(description="YOLO Object Detection")
#     parser.add_argument("--input", type=str, default="./Zoo_Test_Video.mp4",
#                         help="video source. If empty, uses webcam 0 stream")
#     parser.add_argument("--out_filename", type=str, default="",
#                         help="inference video name. Not saved if empty")
#     parser.add_argument("--weights", default="./backup/yolo-zoo_7000.weights",
#                         help="yolo weights path")
#     parser.add_argument("--dont_show", action='store_true',
#                         help="windown inference display. For headless systems")
#     parser.add_argument("--ext_output", action='store_true',
#                         help="display bbox coordinates of detected objects")
#     parser.add_argument("--config_file", default="./cfg/yolo-zoo.cfg",
#                         help="path to config file")
#     parser.add_argument("--data_file", default="./data/zoo.data",
#                         help="path to data file")
#     parser.add_argument("--thresh", type=float, default=.25,
#                         help="remove detections with confidence below this value")
#     parser.add_argument("--max_distance", type=int, default=100,
#                         help="maximum pixel distance at what close proximity is indicated")
#     return parser.parse_args()


def str2int(video_path):
    try:
        return int(video_path)
    except ValueError:
        return video_path


def check_arguments_errors(args):
    assert 0 < args.thresh < 1
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(
            os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(
            os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(
            os.path.abspath(args.data_file))))
    if str2int(args.input) == str and not os.path.exists(args.input):
        raise(ValueError("Invalid video path {}".format(os.path.abspath(args.input))))


# Calculate Euclidean Distance between two points
def euclidean_distance(p1, p2):
    dst = math.sqrt(p1**2 + p2**2)
    return dst


# Converts center coordinates to rectangle coordinates
def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


# Draw boxes and log distance between objects and fps
def cvDrawBoxes(detections, img):
    if len(detections) > 0:
        centroid_dict = dict()
        objectId = 0

        img = darknet.draw_boxes(detections, frame_resized, class_colors)

        for label, confidence, bbox in detections:
            if label == 'baer' or 'mara' or 'mara1' or 'mara2':
                x, y, w, h = (bbox[0],
                              bbox[1],
                              bbox[2],
                              bbox[3])
                xmin, ymin, xmax, ymax = convertBack(
                    float(x), float(y), float(w), float(h))
                centroid_dict[objectId] = (
                    int(x), int(y), xmin, ymin, xmax, ymax, label)
                objectId += 1

        red_zone_list = []
        red_line_list = []

        for (id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2):
            dx, dy = p1[0] - p2[0], p1[1] - p2[1]
            distance = euclidean_distance(dx, dy)

            if distance < max_distance:
                if id1 not in red_zone_list:
                    red_zone_list.append(id1)
                    red_line_list.append(p1[0:2])
                if id2 not in red_zone_list:
                    red_zone_list.append(id2)
                    red_line_list.append(p2[0:2])

            if centroid_dict[id1][6] not in centroid_dict[id2][6]:
                print(
                    f'Distance between {centroid_dict[id1][6]} and {centroid_dict[id2][6]}: {distance} Pixel')

                for check in range(0, len(red_line_list)-1):
                    start_point = red_line_list[check]
                    end_point = red_line_list[check+1]

                    check_line_x = abs(end_point[0] - start_point[0])
                    check_line_y = abs(end_point[1] - start_point[1])

                    if (check_line_x < max_distance) and (check_line_y < max_distance):
                        cv2.line(img, start_point, end_point, (255, 0, 0), 2)

        text = "Animals close: %s" % str(
            len(red_zone_list))
        location = (10, 25)
        cv2.putText(img, text, location, cv2.FONT_HERSHEY_SIMPLEX,
                    1, (246, 86, 86), 2, cv2.LINE_AA)
    return img


if __name__ == '__main__':
    args = parser()
    check_arguments_errors(args)

    configPath = args.config_file
    weightPath = args.weights
    input_path = args.input
    metaPath = args.data_file
    thresh = args.thresh
    max_distance = args.max_distance

    network, class_names, class_colors = darknet.load_network(
        configPath,  metaPath, weightPath, batch_size=1)
    class_colors = {'baer': (255, 0, 0), 'mara': (0, 0, 255), 'mara1': (
        255, 0, 255), 'mara2': (255, 100, 255)}  # Classcolor override

    cap = cv2.VideoCapture(input_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()

        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (width, height),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

        detections = darknet.detect_image(
            network, class_names, darknet_image, thresh)

        image = cvDrawBoxes(detections, frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        fps = int(1/(time.time() - prev_time))
        print("FPS: {}".format(fps))
        darknet.print_detections(detections, args.ext_output)

        cv2.imshow('Demo', image)
        cv2.waitKey(1)

    cap.release()
