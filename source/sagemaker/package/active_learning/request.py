import json

ac_arn_map = {'us-west-2': '081040173940',
              'us-east-1': '432418664414',
              'us-east-2': '266458841044',
              'ca-central-1': '918755190332',
              'eu-west-1': '568282634449',
              'eu-west-2': '487402164563',
              'eu-central-1': '203001061592',
              'ap-northeast-1': '477331159723',
              'ap-northeast-2': '845288260483',
              'ap-south-1': '565803892007',
              'ap-southeast-1': '377565633583',
              'ap-southeast-2': '454466003867',
              }

def create_ground_truth_request(manifest_file_uri, label_file_uri, template_file_uri, region, role, job_name_prefix, s3_output_uri, workteam_arn=None):

    public_workteam_arn = 'arn:aws:sagemaker:{}:394669845002:workteam/public-crowd/default'.format(region)

    prehuman_arn = 'arn:aws:lambda:{}:{}:function:PRE-BoundingBox'.format(region, ac_arn_map[region])
    acs_arn = 'arn:aws:lambda:{}:{}:function:ACS-BoundingBox'.format(region, ac_arn_map[region])

    task_description = 'Make a bounding box around each pedestrian'
    task_keywords = ['image', 'object', 'detection', 'bounding', 'box']
    task_title = task_description

    human_task_config = {
        "AnnotationConsolidationConfig": {
            "AnnotationConsolidationLambdaArn": acs_arn,
        },
        "PreHumanTaskLambdaArn": prehuman_arn,
        "MaxConcurrentTaskCount": 200,  # 200 tasks will be sent at a time to the workteam.
        "NumberOfHumanWorkersPerDataObject": 1,  # 1 workers will be enough to label each text.
        "TaskAvailabilityLifetimeInSeconds": 21600,  # Your work team has 6 hours to complete all pending tasks.
        "TaskDescription": task_description,
        "TaskKeywords": task_keywords,
        "TaskTimeLimitInSeconds": 300,  # Each text must be labeled within 5 minutes.
        "TaskTitle": task_title,
        "UiConfig": {
            "UiTemplateS3Uri": template_file_uri,
        }
    }

    use_private_workforce = workteam_arn is not None
    if not use_private_workforce:
        human_task_config["PublicWorkforceTaskPrice"] = {
            "AmountInUsd": {
                "Dollars": 0,
                "Cents": 1,
                "TenthFractionsOfACent": 2,
            }
        }
        human_task_config["WorkteamArn"] = public_workteam_arn
    else:
        human_task_config["WorkteamArn"] = workteam_arn

    ground_truth_request = {
        "InputConfig": {
            "DataSource": {
                "S3DataSource": {
                    "ManifestS3Uri": manifest_file_uri,
                }
            },
            "DataAttributes": {
                "ContentClassifiers": [
                    "FreeOfPersonallyIdentifiableInformation",
                    "FreeOfAdultContent"
                ]
            },
        },
        "OutputConfig": {
            "S3OutputPath": s3_output_uri,
        },
        "HumanTaskConfig": human_task_config,
        "LabelingJobNamePrefix": job_name_prefix,
        "RoleArn": role,
        "LabelAttributeName": "label",
        "LabelCategoryConfigS3Uri": label_file_uri,
    }
    with open("./requests/ground_truth.request", "w") as f:
        json.dump(ground_truth_request, f, indent=2)

    return json.dumps(ground_truth_request)