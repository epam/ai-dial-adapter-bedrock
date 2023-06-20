import json

from bedrock import BedrockModel
from utils.init import init

init()


model = BedrockModel()

print(json.dumps(model.available_models(), indent=2))

prompt = """In this sentence "there are many countries in Eurpoe, for example, France, India and Poland", what countries are mentioned and which are in Eurpoe? """

print(json.dumps(model.predict("amazon.titan-tg1-large", prompt), indent=2))
