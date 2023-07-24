import ai21

from utils.init import init

init()

# The bedrock API is available only to us-east-1 scoped clients right now
ai21.aws_region = "us-east-1"

response = ai21.Completion.execute(
    destination=ai21.BedrockDestination(
        model_id=ai21.BedrockModelID.J2_JUMBO_INSTRUCT
    ),
    prompt="Write a news release in the voice of a global banking conglomerate announcing an unprecedented building campaign to expand and rebuild their corporate headquarters in Alpha Centauri.",
    maxTokens=200,
)

print(response)
