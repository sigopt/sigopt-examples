# SigOpt Orchestrate AWS IAM Permissions

## Full Access

SigOpt Orchestrate requires that AWS accounts creating clusters have full access to the following services:
 * iam
 * ecr
 * ec2
 * cloudformation
 * autoscaling
 * eks
 * ssm

For your convenience, we have created a JSON policy document, [orchestrate_full_access_policy_document.json](orchestrate_full_access_policy_document.json).

You can use this document to [Create an IAM Policy](https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies_create.html) on AWS for your orchestrate users via the console, the command line, or one of the clients.

## Run Only

"Run only" users can connect to existing clusters and run experiments, but cannot create clusters.

These users require permission to assume the orchestrate-generated <cluster-name>-k8s-access-role role (available in your AWS console).

Run only users also require full acccess to ECR. You can use the document [orchestrate_run_only_policy_document.json](orchestrate_run_only_policy_document.json) to create an IAM policy for run only users.
