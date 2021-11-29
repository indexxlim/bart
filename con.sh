gcloud compute tpus describe tpu-env --zone=europe-west4-a
export TPU_IP_ADDRESS=10.45.66.122
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
export TPU_NAME=tpu-env
