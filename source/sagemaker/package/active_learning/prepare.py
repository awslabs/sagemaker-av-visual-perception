import json
import numpy as np
from package.manifest import read_image, dump_manifest_rows

from sagemaker.s3 import S3Uploader

def unlabeled_input():
    pass

def partially_labeled_input(s3_location, manifest_rows, ratio_unlabeled=0.8):

    with open("./artifacts/annotations_metadata.json", "r") as f:
        label_metadata = json.load(f)

    partially_labelled_manifest_rows = []
    for row in manifest_rows:
        image = read_image(row['source-ref'])
        height, width, depth = image.shape
        image_size = {
            "width": width, "height": height, "depth": depth
        }

        manifest_row = dict()
        # image_path = os.path.basename(row['source-ref'])
        # manifest_row['source-ref'] = "{}/images/{}".format(s3_location, image_path)
        manifest_row['source-ref'] = row['source-ref']
        manifest_row['label'] = {"annotations": [], "image_size": [image_size]}
        for annotation in row['true-labels']['annotations']:
            if annotation["class_id"] == 2: # get for pedestrians
                manifest_row['label']['annotations'].append({
                    "class_id": 0,
                    "top": annotation['top'],
                    "left": annotation['left'],
                    "width": annotation['width'],
                    "height": annotation['height']
                })
                manifest_row['label-metadata'] = label_metadata
        if len(manifest_row['label']['annotations']):
            partially_labelled_manifest_rows.append(manifest_row)


    n_unlabeled = int(len(partially_labelled_manifest_rows) * ratio_unlabeled)
    n_labeled = int(len(partially_labelled_manifest_rows) - n_unlabeled)

    print("{} examples will be labeled".format(n_labeled))
    print("{} examples will be unlabeled".format(n_unlabeled))

    # Destroy labels
    unlabeled_idx = np.random.choice(np.arange(len(partially_labelled_manifest_rows)), n_unlabeled, replace=False)
    for i in unlabeled_idx:
        del partially_labelled_manifest_rows[i]['label']
        del partially_labelled_manifest_rows[i]['label-metadata']

    manifest_path = "./manifests/partially_labeled_input.manifest"
    dump_manifest_rows(partially_labelled_manifest_rows, manifest_path)

    return S3Uploader.upload(manifest_path, s3_location)

def labels_config_and_template(s3_location):
    s3_labels_path = S3Uploader.upload("./artifacts/class_labels.json", s3_location)
    s3_template_path = S3Uploader.upload("./artifacts/instructions.template", s3_location)
    return s3_labels_path, s3_template_path