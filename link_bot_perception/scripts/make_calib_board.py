#!/usr/bin/python3

import argparse
import io

import pygame
from cv2 import aruco

from marker_generation import genSvg


def main():
    parser = argparse.ArgumentParser(description='Generate Aruco Markers.')
    parser.add_argument('id', type=int, help='marker id')
    parser.add_argument('--dict-number', type=int, default='7')
    args = parser.parse_args()

    paper_size = (215.9, 279.4)
    aruco_border_str = genSvg(args.id, args.dict_number, paper_size)
    pygame_surface = pygame.image.load(io.BytesIO(aruco_border_str.encode()))
    aruco_dict = aruco.Dictionary_get(args.dict_number)
    img = aruco.drawMarker(aruco_dict, args.id, int(2000))
    # pygame.draw.line(pygame_surface, color=[0, 0, 0], start_pos=[10, 10], end_pos=[100, 10])

    pygame.image.save(pygame_surface, f'calib{args.id}.png')


if __name__ == '__main__':
    main()
