
# **Day 23: Exploring SyftBox Setup**

## **Overview**

Yesterday, I explored the concept of **Ring Computation** (Day 22). Today, I focused on setting up the **SyftBox** tool from OpenMined. This tutorial guides you through configuring a single SyftBox node, which will later be expanded to emulate three nodes (“syft”, “drift”, and “thrift”) in the public SyftBox network. The goal is to create a reproducible environment for exploring **Federated Learning (FL)** concepts.

---

## **Step-by-Step Setup**

### **Step 1: Setting Up the Environment**

#### **1.1 Create the Node Directory**

```sh
mkdir thrift
cd thrift
```

#### **1.2 Create an `.env` File**

Create a file named `.env` with the following content:

```sh
EMAIL=myemail+thrift1@gmail.com
DATA_DIR=/build/data
CONFIG_PATH=/build/config.json
PORT=8888
```

Replace `myemail+thrift1@gmail.com` with your own email alias.

---

### **Step 2: Create a Dummy Python File**

Create a file named `app/main.py` to ensure the container runs persistently during the SyftBox setup process:

```python
import time
import random

def main():
    print("Docker container started. Running infinite loop...")
    while True:
        sleep_time = random.randint(5, 15)
        print(f"Sleeping for {sleep_time} seconds...")
        time.sleep(sleep_time)
        print("Still running! Your Docker container is alive and well.")

if __name__ == "__main__":
    main()
```

---

### **Step 3: Create the Dockerfile**

The Dockerfile defines the container environment for SyftBox. Create a file named `Dockerfile` with the following content:

```Dockerfile
# Base Image
FROM python:3.12-slim

# Install Dependencies
RUN apt-get update && \
    apt-get install -y wget gnupg curl build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev git jq unzip vim

# Set Environment Variables for pyenv
ENV PYENV_ROOT /root/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

# Install pyenv and Python
RUN curl https://pyenv.run | bash && \
    pyenv install 3.12 && \
    pyenv global 3.12

# Work Directory
WORKDIR /build

# Environment Variables
ENV PYTHONPATH=/build

# Copy Application Code
COPY . .
RUN curl -LsSf https://syftbox.openmined.org/install.sh | sed 's/ASK_RUN_CLIENT=1/ASK_RUN_CLIENT=0/' | sh

# Command
CMD ["python", "app/main.py"]
```

---

### **Step 4: Create the Docker-Compose File**

Create a `docker-compose.yml` file to manage the container:

```yaml
version: '3.8'

services:
  syftbox:
    build: .
    container_name: thriftbox
    ports:
      - "${PORT}:${PORT}"
    volumes:
      - .:/build
    environment:
      - EMAIL=${EMAIL}
      - DATA_DIR=${DATA_DIR}
      - CONFIG_PATH=${CONFIG_PATH}
#    command: >
#      sh -c ". ~/.zshrc && syftbox client --email $EMAIL --data-dir $DATA_DIR --config_path $CONFIG_PATH --port $PORT"
```

Uncomment the `command` section after completing the initial setup.

---

### **Step 5: Setting Up SyftBox Credentials**

#### **5.1 Start the Container**

Start the container:

```sh
docker-compose up -d
```

Verify the container is running:

```sh
docker ps
```

#### **5.2 Access the Container**

Enter the running container:

```sh
docker exec -it thriftbox bash
```

Load the environment variables and run the SyftBox client:

```sh
source .env
sh -c ". ~/.zshrc && syftbox client --email $EMAIL --data-dir $DATA_DIR --config_path $CONFIG_PATH --port $PORT"
```

Follow the prompts to set up SyftBox, including verifying the token sent to your email.

---

### **Step 6: Finalizing the Setup**

#### **6.1 Stop and Edit the Docker Compose File**

Stop the container:

```sh
ctrl + c
exit
```

Uncomment the `command` section in the `docker-compose.yml` file and restart the container:

```sh
docker-compose stop
docker-compose up -d
```

Verify the container is running:

```sh
docker ps
```

Check the SyftBox network status here:
[SyftBox Network Stats](https://syftbox.openmined.org/datasites/aggregator@openmined.org/syft_stats.html).

---

### **Step 7: Testing the Node with CPU Tracker Tutorial**

Clone the `cpu_tracker_member` repository and test the node:

```sh
docker exec -it thriftbox bash
cd /build/data/apis
git clone https://github.com/openmined/cpu_tracker_member.git
```

Verify the CPU tracker is working:

```sh

cat /build/data/datasites/myemail+thrift1\@gmail.com/api_data/cpu_tracker/cpu_tracker.json
```

Example output:

```json
{
    "cpu": 34.3,
    "timestamp": "2024-12-14 06:38:53"
}
```

Check logs for details:

```sh
tail -f cpu_tracker_member.log
```

Check the SyftBox CPU Tracker contributions here:
[SyftBox CPU Tracker](https://syftbox.openmined.org/datasites/aggregator@openmined.org/index.html).

---

### **Next Steps**

With the first SyftBox node set up, the next steps include:

1. Configuring the additional nodes (“syft” and “drift”).
2. Expanding the environment to implement **Ring Computation**.

Stay tuned for the next steps in building a fully distributed **Federated Learning** workflow!

