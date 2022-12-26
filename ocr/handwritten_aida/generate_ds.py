import argparse
import os
import glob
import random

import yaml
import json

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import cv2
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('ds_path', type=str)
parser.add_argument('ds_store_path', type=str)


def parse_args(parser):
    return parser.parse_args()


def main():
    args = parser.parse_args()
    ds_path = args.ds_path
    ds_store_path = args.ds_store_path

    # Make necessary folders
    if not os.path.exists(ds_store_path):
        os.mkdir(ds_store_path)

    if not os.path.exists(f'{ds_store_path}/images'):
        os.mkdir(f'{ds_store_path}/images')

    if not os.path.exists(f'{ds_store_path}/labels'):
        os.mkdir(f'{ds_store_path}/labels')

    # Find all dataset folders in ds_path, assuming each folder is named batch_1, batch_2, etc.
    ds_search_path = ds_path
    print(f'Looking for dataset folders in {ds_search_path}')
    dir_list = list(os.walk(ds_search_path))[0]

    init_run = True
    classes = []

    for dirs in dir_list:
        if isinstance(dirs, list):  # Incorrect if d is not a list
            i = 1
            for directory in dirs:
                if 'batch' not in directory:
                    print(f'Directory {directory} does not seem to contain a dataset. Skipping.')
                    continue
                json_file_path = f'{ds_search_path + directory}/JSON/' + list(os.walk(f'{ds_search_path + directory}/JSON/'))[0][2][0]
                print(f'Grabbing data from "{directory}"')
                print(f'JSON file path: {json_file_path}')

                with open(json_file_path, 'r') as f:
                    data = json.loads(f.read())

                print(f'Number of items: {len(data)}')

                for d in data:
                    filename = d['filename']
                    img_data = d['image_data']
                    visible_latex = img_data['visible_latex_chars']

                    # Coordinates of each character
                    xmins = img_data['xmins']
                    xmaxs = img_data['xmaxs']
                    ymins = img_data['ymins']
                    ymaxs = img_data['ymaxs']

                    img_path = f'{ds_search_path + directory}/background_images/{filename}'
                    img = cv2.imread(img_path)

                    # sanity check
                    assert len(visible_latex) == len(xmins) == len(xmaxs) == len(ymins) == len(ymaxs)

                    df = pd.DataFrame(columns=['class', 'x_center', 'y_center', 'width', 'height'])

                    for lt, xmin, xmax, ymin, ymax in zip(visible_latex, xmins, xmaxs, ymins, ymaxs):
                        if lt not in classes:  # make sure one class gets one idx
                            classes.append(lt)

                        class_idx = classes.index(lt)

                        x_center = abs(xmax + xmin) / 2
                        y_center = abs(ymax + ymin) / 2
                        width = abs(xmax - xmin)
                        height = abs(ymax - ymin)

                        # Another sanity check
                        assert x_center + (width / 2) <= 1 and y_center + (height / 2) <= 1

                        # Now we write to the dataframe
                        df.loc[len(df.index)] = [class_idx, x_center, y_center, width, height]

                        # Make sure class label is int
                        df = df.astype({'class': 'int'})

                    # END CHARS --------------------------------------------------------------------------------

                    # All checks passed and we have a ready df
                    cv2.imwrite(f'{ds_store_path}/images/{filename}.png', img)

                    with open(f'{ds_store_path}/labels/{filename}.txt', 'w') as f:
                        df_str = df.to_string(header=False, index=False)
                        f.write(df_str)

                    if init_run:  # This is the first iteration, so show example images
                        init_run = False
                        xmins_raw = img_data['xmins_raw']
                        xmaxs_raw = img_data['xmaxs_raw']
                        ymins_raw = img_data['ymins_raw']
                        ymaxs_raw = img_data['ymaxs_raw']

                        full_latex = [x for x in img_data['full_latex_chars'] if x != '{' and x != '}']

                        print(f'Sample image file name: {filename}')
                        print(f'Full Latex: {full_latex}')
                        print(f'Image shape: {img.shape}')

                        for lt, xmin, xmax, ymin, ymax in zip(visible_latex, xmins_raw, xmaxs_raw, ymins_raw, ymaxs_raw):
                            img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (36, 255, 12), 1)
                            cv2.putText(img, lt, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                        plt.imshow(img)
                        plt.show()

                # END FILE -------------------------------------------------------------------------------------

                i += 1

    # Write classes
    with open('classes.txt', 'w') as f:
        f.write(str(classes))


if __name__ == '__main__':
    main()
