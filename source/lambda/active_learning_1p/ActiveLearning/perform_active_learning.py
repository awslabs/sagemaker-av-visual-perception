import json
import os
import numpy as np

from typing import List

from ActiveLearning.s3_helper import S3Ref, download_stringio, download_with_query, upload, create_ref_at_parent_key, get_uris_inside_prefix
from ActiveLearning.string_helper import generate_job_id_and_s3_path
from io import StringIO

from ActiveLearning.helper import SimpleActiveLearning, ImageActiveLearning

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_class_map_from_s3(labels_s3_uri):
    """
     fetch the list of labels from a label s3 bucket.
    """
    labels_source = S3Ref.from_uri(labels_s3_uri)
    labels_dict = json.loads(download_stringio(labels_source).read())
    class_map = labels_dict["class-map"]
    return class_map


def get_label_names_from_s3(labels_s3_uri):
    """
     fetch the list of labels from a label s3 bucket.
    """
    labels_source = S3Ref.from_uri(labels_s3_uri)
    labeled_query = """SELECT label FROM S3Object[*].labels[*].label"""
    result_inp = download_with_query(labels_source, labeled_query)
    label_names = []
    for line in result_inp:
        label_data = json.loads(line)
        label_names.append(label_data['label'])
    return label_names


def get_dicts_from_manifest_file(inference_input):
    """
     Load inference input as a python list
    """
    sources = []
    for line in inference_input:
        data = json.loads(line)
        sources.append(data)
    inference_input.seek(0)
    return sources


def get_predictions(inference_output):
    """
     Load inference output as a python list
    """
    predictions = []
    for line in inference_output:
        data = json.loads(line)
        prediction = {}
        for key, value in data.items():
            if key != "SageMakerOutput":
                prediction[key] = value
            else:
                if not isinstance(value, dict):
                    print("Error: Expected dictionary inside SageMakerOutput.")
                prediction.update(value)
        predictions.append(prediction)
    return predictions


def collect_inference_inputs(s3_input_uri):
    """
    Read the content of manifest specified by input parameter and return a 3-dimensional tuple:
    1. S3Ref corresponding to the input URI
    2. StringIO of the contents of the targeted manifest
    3. List of dicts corresponding to the lines present in the targeted manifest
    """
    inference_input_s3_ref = S3Ref.from_uri(s3_input_uri)
    inference_input = download_stringio(inference_input_s3_ref)
    manifest_dicts = get_dicts_from_manifest_file(inference_input)
    logger.info("Collected {} inference inputs.".format(len(manifest_dicts)))
    return inference_input_s3_ref, inference_input, manifest_dicts


def collect_inference_outputs(inference_output_uri):
    """
     collect information related to output of inference.
    """
    sagemaker_output_file = "unlabeled.manifest.out"
    prediction_output_uri = inference_output_uri + sagemaker_output_file
    prediction_output_s3 = S3Ref.from_uri(prediction_output_uri)
    prediction_output = download_stringio(prediction_output_s3)
    predictions = get_predictions(prediction_output)
    logger.info("Collected {} inference outputs.".format(len(predictions)))
    return predictions


def collect_inference_outputs_from_prefix(inference_output_uri):
    """
    Input parameter specifies the prefix where *.out files are generated. Query for the *.out files and return
    a 2-dimensional tuple:
    1. List of S3Refs corresponding to each .out file
    2. List of string corresponding to the content of each .out file
    """
    inference_output_s3_ref = S3Ref.from_uri(inference_output_uri)
    inference_output_keys = get_uris_inside_prefix(inference_output_s3_ref)
    inference_output_tuples = []
    for inference_output_key in inference_output_keys:
        inference_output_prefix = 's3://{}/{}'.format(inference_output_s3_ref.bucket, inference_output_key)
        # only include .out files
        _, extension = os.path.splitext(inference_output_prefix)
        if extension == ".out":
            inference_output_s3_ref = S3Ref.from_uri(inference_output_prefix)
            inference_output_string = download_stringio(inference_output_s3_ref).read()
            inference_output_dict = json.loads(inference_output_string)
            inference_output_tuples.append((inference_output_s3_ref, inference_output_dict))
    logger.info("Collected {} inference outputs.".format(len(inference_output_tuples)))
    return zip(*inference_output_tuples)  # converts list of tuples into tuple of lists


def write_auto_annotations(active_learning_strategy, sources, predictions, inference_input_s3_ref):
    """
     write auto annotations to s3
    """
    logger.info("Generating auto annotations where confidence is high.")
    auto_annotation_stream = StringIO()
    auto_annotations = active_learning_strategy.autoannotate(predictions, sources)
    for auto_annotation in auto_annotations:
        auto_annotation_stream.write(json.dumps(auto_annotation) + "\n")

    # Auto annotation.
    auto_dest = create_ref_at_parent_key(inference_input_s3_ref, "autoannotated.manifest")
    upload(auto_annotation_stream, auto_dest)
    logger.info("Uploaded autoannotations to {}.".format(auto_dest.get_uri()))
    return auto_dest.get_uri(), auto_annotations


def write_selector_file(active_learning_strategy, sources, inference_input_s3_ref, inference_input, auto_annotations):
    """
     write selector file to s3. This file is used to decide which records should be labeled by humans next.
    """
    logger.info("Selecting input for next manual annotation")
    selection_data = StringIO()
    selections = active_learning_strategy.select_for_labeling(sources, auto_annotations)
    selections_set = set(selections)
    for line in inference_input:
        data = json.loads(line)
        if data["id"] in selections_set:
            selection_data.write(json.dumps(data) + "\n")
    inference_input.seek(0)
    selection_dest = create_ref_at_parent_key(
        inference_input_s3_ref, "selection.manifest")
    upload(selection_data, selection_dest)
    logger.info("Uploaded selections to {}.".format(selection_dest.get_uri()))
    return selection_dest.get_uri(), selections


def align_manifest_and_inference_output_dicts(manifest_dicts: List[str], inference_output_s3_refs: List[S3Ref],
                                              inference_output_dicts: List[str]):
    """
    Look at filenames and sort in lexicographical order.
    """
    manifest_dicts_sorted = sorted(manifest_dicts, key=lambda x: str.lower(x['source-ref']))
    inference_output_s3_refs_sorted, inference_output_dicts_sorted = \
        zip(*sorted(zip(inference_output_s3_refs, inference_output_dicts), key=lambda x: str.lower(x[0].key)))

    return manifest_dicts_sorted, inference_output_s3_refs_sorted, inference_output_dicts_sorted


def lambda_handler(event, context):
    """
    This function generates auto annotatations and performs active learning.
    - auto annotations generates machine labels for confident examples.
    - active learning selects for examples to be labeled by humans next.
    """
    labels_s3_uri = event['LabelCategoryConfigS3Uri']
    job_name_prefix = event['LabelingJobNamePrefix']
    job_name = "labeling-job/{}".format(job_name_prefix)
    label_attribute_name = event['LabelAttributeName']
    meta_data = event['meta_data']
    intermediate_folder_uri = meta_data["IntermediateFolderUri"]
    input_total = int(meta_data['counts']['input_total'])
    # Select for next round of manual labeling.
    # max_selections = int(input_total * 0.005)
    max_selections = 16
    # Handle corner case where integer division can lead us to 0 selections.
    if max_selections == 0:
        max_selections = input_total

    inference_input_s3_ref, inference_input, manifest_dicts = \
        collect_inference_inputs(meta_data['UnlabeledManifestS3Uri'])
    inference_output_s3_refs, inference_output_dicts = \
        collect_inference_outputs_from_prefix(meta_data['transform_config']['S3OutputPath'])

    # Align manifest lines and inference output lines so that we can populate predictions in the manifest
    manifest_dicts_aligned, inference_output_s3_refs_aligned, inference_output_dicts_aligned = \
        align_manifest_and_inference_output_dicts(manifest_dicts, inference_output_s3_refs, inference_output_dicts)

    class_map = get_class_map_from_s3(labels_s3_uri)
    logger.info("Retrieved class map: {}".format(json.dumps(class_map)))
    # label_names = get_label_names_from_s3(labels_s3_uri)
    # logger.info("Collected {} label names.".format(len(label_names)))

    image_al = ImageActiveLearning(job_name, label_attribute_name, class_map, max_selections)
    meta_data['autoannotations'], auto_annotations = write_auto_annotations(
        image_al, manifest_dicts_aligned, inference_output_dicts_aligned, inference_input_s3_ref)
    meta_data['selections_s3_uri'], selections = write_selector_file(
        image_al, manifest_dicts_aligned, inference_input_s3_ref, inference_input,
        auto_annotations)
    meta_data['selected_job_name'], meta_data['selected_job_output_uri'] = generate_job_id_and_s3_path(
        job_name_prefix, intermediate_folder_uri)
    meta_data['counts']['autoannotated'] = len(auto_annotations)
    meta_data['counts']['selected'] = len(selections)
    return meta_data


# if __name__ == "__main__":
#     with open("input.json", "r") as f:
#         event = json.load(f)
#     context = {}
#     lambda_handler(event, context)
