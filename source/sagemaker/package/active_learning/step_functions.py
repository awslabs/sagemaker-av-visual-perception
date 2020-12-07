import boto3

class ActiveLearningPipeline:
    def __init__(self, stateMachineArn):
        self.stateMachineArn = stateMachineArn
        self.sfn = boto3.client('stepfunctions')

    def start_execution(self, input_data):
        response = self.sfn.start_execution(
            stateMachineArn=self.stateMachineArn,
            input=input_data
        )

        return response['executionArn']
