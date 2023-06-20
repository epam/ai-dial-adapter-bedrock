python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install ./bedrock/botocore-1.29.142-py3-none-any.whl
pip install ./bedrock/boto3-1.26.142-py3-none-any.whl
pip install ./bedrock/awscli-1.27.142-py3-none-any.whl