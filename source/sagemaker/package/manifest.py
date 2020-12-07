import json
from io import BytesIO, TextIOWrapper
from parse import parse
import boto3
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
from PIL import Image


def get_manifest_rows_from_path(manifest_path):
    if "s3://" in manifest_path:
        s3 = boto3.client('s3')
        bucket, key = parse("s3://{}/{}", manifest_path)
        bytestream = BytesIO()
        s3.download_fileobj(bucket, key, bytestream)
        bytestream.seek(0)
        manifest_stringio = TextIOWrapper(bytestream, encoding='utf-8')
        manifest_lines = manifest_stringio.readlines()
    else:
        with open(manifest_path, "r") as f:
            manifest_lines = f.readlines()
    manifest_rows = [json.loads(s) for s in manifest_lines]
    return manifest_rows


def dump_manifest_rows(manifest_rows, manifest_path):
    with open(manifest_path, "w") as f:
        for manifest_row in manifest_rows:
            manifest_row_json = json.dumps(manifest_row)
            f.write(manifest_row_json)
            f.write("\n")

def read_image(image_path):
    if "s3://" in image_path:
        s3 = boto3.client('s3')
        bucket, key = parse("s3://{}/{}", image_path)
        bytestream = BytesIO()
        s3.download_fileobj(bucket, key, bytestream)
        bytestream.seek(0)
    else:
        with open(image_path, "rb") as f:
            bytestream = f.read()
    image = np.array(Image.open(bytestream))
    return image

def visualize_manifest_images(manifest_path, max_images=10, verbose=False):
    manifest_rows = get_manifest_rows_from_path(manifest_path)

    for i, manifest_row in enumerate(manifest_rows):
        if i == max_images:
            return
        source_ref = manifest_row["source-ref"]

        image = read_image(source_ref)
        plt.figure()
        ax = plt.subplot()
        ax.axis("off")
        ax.imshow(image)
        annotations = manifest_row['label']['annotations'] if 'label' in manifest_row else []
        for annotation in annotations:
            top = int(annotation['top'])
            left = int(annotation['left'])
            width = int(annotation['width'])
            height = int(annotation['height'])
            if verbose:
                print(top, left, width, height)
            rect = Rectangle((left, top), width, height, edgecolor='r', linewidth=3, fill=False)
            ax.add_patch(rect)
        groundtruth = manifest_row['true-labels']['annotations'] if 'true-labels' in manifest_row else []
        for annotation in groundtruth:
            top = int(annotation['top'])
            left = int(annotation['left'])
            width = int(annotation['width'])
            height = int(annotation['height'])
            if verbose:
                print(top, left, width, height)
            rect = Rectangle((left, top), width, height, edgecolor='b', linewidth=3, fill=False)
            ax.add_patch(rect)
        print("Showing image for {}".format(manifest_row['source-ref']))
        plt.show()