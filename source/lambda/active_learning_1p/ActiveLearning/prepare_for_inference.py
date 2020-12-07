import json
import os

from io import StringIO
from pathlib import Path

from ActiveLearning.s3_helper import S3Ref, copy_with_query_and_transform, create_ref_at_parent_key, download_stringio, copy

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def augment_inference_input(inference_raw):
    """
     The inference manifest needs to be augmented with a value 'k' so that blazing text
     produces all probabilities instead of just the top match.
    """
    augmented_inference = StringIO()
    for line in inference_raw:
        infer_dict = json.loads(line)
        # Note: This number should ideally be equal to the number of classes.
        # But using a big number, produces the same result.
        infer_dict['k'] = 1000000
        augmented_inference.write(json.dumps(infer_dict) + "\n")
    logger.info("Augmented inference data by adding 'k' to each line.")
    return augmented_inference


def create_tranform_config(training_config):
    """
     Transform config specifies input parameters for the transform job.
    """
    return {
        # We reuse the training job name for the model name and corresponding
        # transform job name.
        'TransformJobName': training_config['TrainingJobName'],
        'ModelName': training_config['TrainingJobName'],
        'S3OutputPath': training_config['S3OutputPath']
    }


def lambda_handler(event, context):
    """
    This function generates auto annotations and performs active learning.
    """
    label_attribute_name = event['LabelAttributeName']
    meta_data = event['meta_data']
    s3_input_uri = meta_data['IntermediateManifestS3Uri']

    transform_config = create_tranform_config(meta_data['training_config'])

    source = S3Ref.from_uri(s3_input_uri)
    unlabeled_manifest_s3_ref = S3Ref.from_uri(transform_config['S3OutputPath'] + "unlabeled.manifest")

    logger.info("Creating inference output from unlabeled subset of input {}.".format(
        s3_input_uri))
    sql_unlabeled = """select * from s3object[*] s where s."{}" is missing """
    unlabeled_query = sql_unlabeled.format(label_attribute_name)
    copy_with_query_and_transform(
        source, unlabeled_manifest_s3_ref, unlabeled_query, augment_inference_input)

    # Make S3 prefix for images to be labeled by active learning process
    unlabeled_directory_pref_s3_ref = S3Ref.from_uri(transform_config['S3OutputPath'] + "labeled_by_active_learning")
    unlabeled_manifest_string_io = download_stringio(unlabeled_manifest_s3_ref)
    unlabeled_manifest_string = unlabeled_manifest_string_io.read()
    unlabeled_manifest_json_strings = unlabeled_manifest_string.strip().split("\n")
    for unlabeled_manifest_json_string in unlabeled_manifest_json_strings:
        unlabeled_manifest_row = json.loads(unlabeled_manifest_json_string)
        unlabeled_image_s3_ref_string = unlabeled_manifest_row['source-ref']
        unlabeled_image_s3_ref = S3Ref.from_uri(unlabeled_image_s3_ref_string)
        image_basename = os.path.basename(unlabeled_image_s3_ref[-1])  # e.g. 1234.jpg, no prefix
        new_pref_for_image = "{}/{}".format(unlabeled_directory_pref_s3_ref.get_uri(), image_basename)
        new_pref_for_image_s3_ref = S3Ref.from_uri(new_pref_for_image)
        copy(unlabeled_image_s3_ref, new_pref_for_image_s3_ref)
        logger.info("Uploaded unlabeled image for inference to {}.".format(
            new_pref_for_image_s3_ref.get_uri()))

    meta_data['UnlabeledPrefixS3Uri'] = unlabeled_directory_pref_s3_ref.get_uri()
    meta_data['UnlabeledManifestS3Uri'] = unlabeled_manifest_s3_ref.get_uri()
    logger.info("Uploaded unlabeled manifest for inference to {}.".format(
        unlabeled_manifest_s3_ref.get_uri()))
    logger.info("Uploaded unlabeled image directory for inference to {}.".format(
        unlabeled_directory_pref_s3_ref.get_uri()))

    meta_data['transform_config'] = transform_config
    return meta_data
