FROM tensorflow/tensorflow:2.6.0-gpu

# Bad keys in base image https://github.com/NVIDIA/nvidia-docker/issues/1631#issuecomment-1112828208
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

ENV TZ=Australia/Sydney
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Add some args for creating user. The UID should match your UID. To check, do echo $UID in terminal
ARG USERNAME=AcT_container
ARG UID=1005
ARG HOME_DIR=/home/$USERNAME

# Add user
# Best practise: don't run as root in container and ensure USERNAME appears in prompt
# uid has to match id of repo owner on host. Ensures you don't get locked out of your own folder while inside the container
# Has to correspond to the id of the user that owns the folder being mounted from the host into the container
RUN useradd --uid $UID --home-dir $HOME_DIR --create-home $USERNAME

# Create a default password for the user and make him a sudoer. Variables won't be interpretad inside '', so use "" instead.
RUN echo "$USERNAME:password" | chpasswd && adduser $USERNAME sudo

# Obtain all relevant packages
RUN apt-get update && \
    apt-get install -y sudo cmake wget bash zip git rsync build-essential software-properties-common ca-certificates xvfb geany

# Get Python3.8 and relevant libraries
RUN apt-get install -y python3.8-venv python3.8-dev python3-pip
RUN apt-get install -y libsm6 libxrender1 libfontconfig1 libpython3.8-dev libopenblas-dev
RUN python3.8 -m pip install -U --force-reinstall pip

# Copy requirements.txt from the forked AcT repo in the /tmp directory and install all requirements
WORKDIR $HOME_DIR
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
RUN mkdir AcT

# Make the user the owner of their home directory to allow read/write access
RUN chown -R $USERNAME $HOME_DIR
# RUN chown -R $UID $HOME_DIR/AcT # Might need to run this inside the container itself

# ==================================================================
# ROS from official ROS page

# ==================================================================
# config & cleanup
# ------------------------------------------------------------------
RUN ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

# Start an interactive container as $USER instead of ROOT
USER $USERNAME

ENTRYPOINT ["/bin/sh", "-c"]
CMD ["bash"]
