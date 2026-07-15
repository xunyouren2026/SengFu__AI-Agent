"""
Cloud Tools Module - 云服务工具模块
提供AWS和Azure云服务的统一封装接口
"""

from .aws_sdk_wrapper import (
    AWSSDKWrapper,
    AWSConfig,
    AWSService,
    AWSRegion,
    S3Client,
    EC2Client,
    LambdaClient,
    DynamoDBClient,
    S3Object,
    EC2Instance,
    LambdaFunction,
    DynamoDBItem,
    create_aws_client
)

from .azure_sdk_wrapper import (
    AzureSDKWrapper,
    AzureConfig,
    AzureService,
    AzureRegion,
    BlobClient,
    VMClient,
    FunctionsClient,
    BlobContainer,
    BlobObject,
    VirtualMachine,
    AzureFunction,
    create_azure_client
)

__all__ = [
    # AWS
    "AWSSDKWrapper",
    "AWSConfig",
    "AWSService",
    "AWSRegion",
    "S3Client",
    "EC2Client",
    "LambdaClient",
    "DynamoDBClient",
    "S3Object",
    "EC2Instance",
    "LambdaFunction",
    "DynamoDBItem",
    "create_aws_client",
    # Azure
    "AzureSDKWrapper",
    "AzureConfig",
    "AzureService",
    "AzureRegion",
    "BlobClient",
    "VMClient",
    "FunctionsClient",
    "BlobContainer",
    "BlobObject",
    "VirtualMachine",
    "AzureFunction",
    "create_azure_client"
]
