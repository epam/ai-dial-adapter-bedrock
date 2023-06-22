# OpenAI adapter for Bedrock models

The server provides `/chat/completions` and `/completions` endpoint compatible with those of OpenAI API.

## Installation

```sh
./install.sh
```

## Configuration

Create `.env` file and enter the AWS credentials as environment variables:

```
AWS_ACCESS_KEY_ID=<key>
AWS_SECRET_ACCESS_KEY=<key>
```

The variables are required for the server to work, since the Bedrock models are hosted in AWS.

## Test server locally

Run the server:

```sh
python ./app.py
```

Open `localhost:8080/docs` to make sure the server is up and running.

Run the client:

```sh
python ./client.py
```

First select the Bedrock model and chat emulation mode (zero-memory or meta-chat).
Then you will be able to converse with the model.

## Configuration

## Docker

Build the image:

```sh
./build.sh
```

Run the image:

```sh
./run.sh
```

Open `localhost:8080/docs` to make sure the server is up and running.
