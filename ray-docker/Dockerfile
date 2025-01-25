# Custom docker images are not necessarily required for single instance trainings.
# Just use requirements.txt with your training script.
FROM 462105765813.dkr.ecr.ap-northeast-1.amazonaws.com/sagemaker-rl-ray-container:ray-1.6.0-torch-cpu-py36
ENV PATH="/opt/ml/code:${PATH}"
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code
COPY train.py /opt/ml/code/train.py
ENV SAGEMAKER_PROGRAM train.py
