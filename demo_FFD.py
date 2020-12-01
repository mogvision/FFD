import os
import sys
import numpy as np
import argparse
import subprocess
import time

from util.util import readkp


if __name__ == '__main__':
    curr_dir = os.getcwd()
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

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
        '--Time_cost', type=bool, default=False,
        help='Report running time over 25 runs'
        ' (\'-1\' doesn\' report time')

    opt = parser.parse_args()
    print(opt)


    image_formats = [".jpg", ".png", ".ppm", ".pgm"]
    for image_name in os.listdir(opt.input_pairs):
        ext = os.path.splitext(image_name)[1]
        if ext.lower() in image_formats:
            image_dir  = os.path.join(curr_dir, opt.input_pairs, image_name)
            store_dir  = os.path.join(curr_dir, opt.input_pairs)

            process = subprocess.Popen('./FFD '+ str(os.path.join(curr_dir, opt.input_pairs, image_name)) + ' ' \
                + str(store_dir) + ' ' \
                + str(opt.num_level) + ' ' \
                + str(opt.max_keypoints) + ' ' \
                + str(opt.contrast_threshold) + ' ' \
                + str(opt.curvature_ratio) + ' '\
                + str(1*opt.Time_cost), 
                shell=True,
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()

            if (opt.Time_cost):
                print('Feature extraction details: ', stdout)

            keypoints, num_kp = readkp(os.path.join(store_dir, 'FFD_'+image_name+'.txt'))
            print("[+] \n  %s: #detected keypoints->%d"%(image_name,num_kp) )
            with np.printoptions(precision=3, suppress=True, threshold=5):
                print("\tx,\ty,\tscale,\tresponse:\n", keypoints )
