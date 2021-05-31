#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 00:07:53 2020
@author: cagan
"""
import argparse
from flying_mnist import FlyingMNIST
########################################################################################
# Moving MNIST generator with ground truth
# By C.S.Coban
# Inspired by: https://gist.github.com/tencia/afb129122a64bde3bd0c
########################################################################################

def prepare_parser():

    parser = argparse.ArgumentParser()

    # Input params
    parser.add_argument('--canv_height', default = 473, type= int, help = "Canvas image height")
    parser.add_argument('--canv_width', default = 473, type= int, help = "Canvas image width")
    parser.add_argument('--use_trn', default = True, help = "Use MNIST train set")
    parser.add_argument('--num_videos', default = 10, type= int, help = "Number of episodes")
    parser.add_argument('--num_frames', default = 150, type= int, help = "Number of frames in a video")

    # Digit specific params
    parser.add_argument('--use_coloring', default= True, help = "Apply coloring to digits")
    parser.add_argument('--max_digits', default = 5, type= int, help = "Max number of digits")
    parser.add_argument('--max_speed', default = 30, type= int, help = "Max speed of a digit")
    parser.add_argument('--digit_size_min', default = 50, type= int, help = "Minimum digit size")
    parser.add_argument('--digit_size_max', default = 120, type= int, help = "Maximum digit size")
    parser.add_argument('--leaving_digits', default = True, type= str, help = "Allows leaving digits")
    parser.add_argument("--digits", nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
    parser.add_argument('--target_dir', default = "./trn", type= str, help = "Target dir to save")


    return parser


if __name__ == "__main__":

    # Parse args
    parser = prepare_parser()
    opts = parser.parse_args()
    f = FlyingMNIST(opts)
    f.generate()