from ActiveLearning.s3_helper import S3Ref, copy_with_query, create_ref_at_parent_key, download_stringio

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    """
    This method selects 10% of the input manifest as validation and creates an s3 file containing the validation objects.
    """
    label_attribute_name = event['LabelAttributeName']
    meta_data = event['meta_data']
    s3_input_uri = meta_data['IntermediateManifestS3Uri']

    input_total = int(meta_data['counts']['input_total'])
    # 10% of the total input should be used for validation.
    validation_set_size = input_total // 10

    source = S3Ref.from_uri(s3_input_uri)

    validation_labeled_query = """select * from s3object[*] s where s."{}-metadata"."human-annotated" IN ('yes') LIMIT {}""".format(
        label_attribute_name, validation_set_size)
    dest = create_ref_at_parent_key(source, "validation_input.manifest")
    copy_with_query(source, dest, validation_labeled_query)
    logger.info("Uploaded validation set of size {} to {}.".format(
        validation_set_size, dest.get_uri()))

    meta_data['counts']['validation'] = validation_set_size
    meta_data['ValidationS3Uri'] = dest.get_uri()
    return meta_data

# def lambda_handler(event, context):
#     """
#     Hardcode the S3 reference to the validation set of NightOwls
#     """
#     meta_data = event['meta_data']
#     validation_s3 = "replace_with_custom"
#     meta_data['ValidationS3Uri'] = validation_s3
#     validation_s3_ref = S3Ref.from_uri(validation_s3)
#     logging.info("Fetched validation set manifest from {}".format(validation_s3))
#     validation_rows = download_stringio(validation_s3_ref).read().strip().split("\n")
#     logging.info("{:d} rows are present in validation set manifest".format(len(validation_rows)))
#     meta_data['counts']['validation'] = len(validation_rows)
#
#     return meta_data
