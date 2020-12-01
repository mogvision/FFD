
import os
import sys
import time
import cv2
from threading import Thread
from os.path import isfile, join
import numpy as np
from tempfile import TemporaryFile
from os import listdir

import string
import argparse
import subprocess
import time

from util.util import Matching_FFD_keypoints, readkp

if __name__ == '__main__':
    curr_dir = os.getcwd()
    parser = argparse.ArgumentParser(description='Feature Detection by FFD')

    parser.add_argument(
        '--NUM_show_matches', type=int, default=100,
        help='Number of matches shown in the output')

    parser.add_argument(
        '--input_pairs', type=str, default='image',
        help='Path to the images')

    parser.add_argument(
        '--max_keypoints', type=int, default=10000,
        help='Maximum number of keypoints detected by FFD'
             ' (\'-1\' keeps all keypoints)')

    parser.add_argument(
        '--num_level', type=int, default=3,
        help='Number of decomposition levels')

    parser.add_argument(
        '--contrast_threshold', type=float, default=0.05,
        help='FFD\'s contrast threshold')

    parser.add_argument(
        '--curvature_ratio', type=float, default=10.,
        help='FFD\'s curvature ratio')

    parser.add_argument(
        '--time_cost', type=int, default=0,
        help='Report running time over 25 runs'
        ' (\'-1\' doesn\' report time')

    opt = parser.parse_args()
    print(opt)


    KPTS_FFD = []
    IMGs = []

    image_formats = [".jpg", ".png", ".ppm", ".pgm"]
    for image_name in os.listdir(opt.input_pairs):
        ext = os.path.splitext(image_name)[1]
        if ext.lower() in image_formats:
            image_dir  = os.path.join(curr_dir, opt.input_pairs, image_name)
            store_dir  = os.path.join(curr_dir, opt.input_pairs)
            IMGs.append(image_dir)

            process = subprocess.Popen('./FFD '+ str(os.path.join(curr_dir, opt.input_pairs, image_name)) + ' ' \
                + str(store_dir) + ' ' \
                + str(opt.num_level) + ' ' \
                + str(opt.max_keypoints) + ' ' \
                + str(opt.contrast_threshold) + ' ' \
                + str(opt.curvature_ratio) + ' '\
                + str(opt.time_cost), 
                shell=True,
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            keypoints, num_kp = readkp(os.path.join(store_dir, 'FFD_'+image_name+'.txt'))
            KPTS_FFD.append(keypoints)

    Matching_FFD_keypoints(IMGs, KPTS_FFD, opt.NUM_show_matches)

