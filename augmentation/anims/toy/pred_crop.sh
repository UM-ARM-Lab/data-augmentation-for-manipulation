#!/bin/bash
convert toy_no_aug.png -crop $1 toy_no_aug-cropped.png
convert toy_aug.png -crop $1 toy_aug-cropped.png

