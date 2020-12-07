import boto3
cognito = boto3.client('cognito-idp')
sm = boto3.client('sagemaker')
region = boto3.session.Session().region_name

def get_cognito_configs():
    sm_client = boto3.client('sagemaker')
    workforces = sm_client.list_workforces()
    if len(workforces["Workforces"]) > 0:
        return workforces["Workforces"][0]["CognitoConfig"]
    else:
        return None

def create_groundtruth_workteam(workteam_name, config):
    if type(config) == dict:
        cognito.create_group(
            GroupName=workteam_name,
            UserPoolId=config['UserPool'],
        )
        sm.create_workteam(WorkteamName=workteam_name,
                           Description="Labelling team for " + workteam_name,
                           MemberDefinitions=[{
                               "CognitoMemberDefinition": {
                                   'UserPool': config['UserPool'],
                                   'UserGroup': workteam_name,
                                   'ClientId': config['ClientId']}
                           }]
                           )

    else:
        sm.create_workteam(WorkteamName=workteam_name,
              Description="Labelling team for " + workteam_name,
              MemberDefinitions=[{
                "CognitoMemberDefinition":{
                    'UserPool': config.cognito_user_pool,
                    'UserGroup':config.cognito_user_pool_group,
                    'ClientId':config.cognito_clientId}
              }]
             )

    workteam_arn = sm.describe_workteam(WorkteamName=workteam_name)["Workteam"]["WorkteamArn"]
    return workteam_arn


def get_signup_domain(workteam_name):
    sm_client = boto3.client('sagemaker')
    workteams = sm_client.list_workteams()

    for workteam in workteams["Workteams"]:
        if workteam["WorkteamName"] == workteam_name:
            subdomain = workteam["SubDomain"]
            return "https://{}/logout".format(subdomain)
    return None

def update_user_pool_with_invite(userpool, signup_domain):
    create_user_config = {'InviteMessageTemplate':
            {"EmailMessage":
             "Hi there, \n\nYou are invited to work on a labelling project:\n\nSign up here: {}\n\n".format(signup_domain) +
             "Your username is '<b>{username}</b>' and your temporary password is '<b>{####}</b>'."}}
    cognito.update_user_pool(
        UserPoolId=userpool,
        AdminCreateUserConfig=create_user_config
    )
    print("Users will get email with the following message: ")
    print()
    print(create_user_config['InviteMessageTemplate']["EmailMessage"])
