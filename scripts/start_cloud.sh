#!/bin/bash

status=`gcloud compute instances describe gpu-instance | grep status | cut -d ' ' -f2`
if [ "$status" == "TERMINATED" ]; then
    echo "Starting GPU instance"
    gcloud compute instances start gpu-instance
fi

echo "Connecting to an instance"
gcloud compute ssh donatasrep@gpu-instance
