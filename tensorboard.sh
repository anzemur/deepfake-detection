#!/bin/bash

PORT=8081

tensorboard --logdir=./logs --port $PORT
ngrok http $PORT &

