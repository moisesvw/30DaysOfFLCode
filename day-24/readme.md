## Day 24: Exploring Ring Computation with SyftBox - A Multi-Node Setup

On [Day 23](https://github.com/moisesvw/30DaysOfFLCode/tree/main/day-23), I set up **SyftBox** using Docker to create a node named `thrift` running the CPU Tracker member aggregator. Today, I expanded the setup to create two additional nodes, named `drift` and `syft`, and explored OpenMined’s [Ring Computation tutorial](https://github.com/OpenMined/ring). 

This tutorial demonstrates how to perform **ring computation** by sequentially aggregating values from `secret.json` files across nodes until completing the ring. Each node contributes its unique value to the computation, showcasing how this model can reduce communication overhead and avoid reliance on a central server.

---

### **Setting Up Additional Nodes**

Using the `Dockerfile`, `docker-compose.yml`, and `.env` files from Day 23, I created two new directories, `driftbox` and `syftbox`, and replicated the files inside them:

```sh
mkdir driftbox
mkdir syftbox
cp thriftbox/Dockerfile thriftbox/docker-compose.yml thriftbox/.env driftbox/.
cp thriftbox/Dockerfile thriftbox/docker-compose.yml thriftbox/.env syftbox/.
```

After copying the files, I modified the `.env` file in each directory to assign unique `EMAIL` and `PORT` values for each node. Then, I followed the steps from Day 23 to set up SyftBox on each container.

Here’s the output of `docker ps` after the setup:

```sh
docker ps

CONTAINER ID   IMAGE               COMMAND                  CREATED         STATUS        PORTS                                       NAMES
07c16e1f09b4   driftbox-syftbox    "sh -c '. ~/.zshrc &…"   4 seconds ago   Up 1 second   0.0.0.0:8989->8989/tcp, :::8989->8989/tcp   driftbox
88be31616524   thriftbox-syftbox   "sh -c '. ~/.zshrc &…"   22 hours ago    Up 22 hours   0.0.0.0:8888->8888/tcp, :::8888->8888/tcp   thriftbox
1dd04bd4a966   syftbox-syftbox     "sh -c '. ~/.zshrc &…"   About a minute ago      Up About a minute   0.0.0.0:9090->9090/tcp, :::9090->9090/tcp   syftbox
```

---

### **Cloning and Modifying the Ring Project**

Next, I cloned OpenMined’s **Ring Computation example** into the first node using the following steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/OpenMined/ring.git
   ```
2. Modify the `ring/data.json` file to include the participants in the ring. The first node must also appear as the last node to complete the cycle. The edited `data.json` looks like this:
   ```json
   {
       "participants": [
           "myemail+drift1@gmail.com",
           "myemail+syft1@gmail.com",
           "myemail+thrift1@gmail.com",
           "myemail+drift1@gmail.com"
       ],
       "data": 0,
       "current_index": 0
   }
   ```
3. Review the `ring/secret.json` file, which contains the initial data contributed by each node. For example:
   ```json
   {"data": 1}
   ```
4. Copy the modified `ring` directory into the first node’s file system:
   ```sh
   docker cp ring driftbox:/build/data/apis/.
   ```

---

### **Understanding the Ring Computation Process**

Each node contributes a value from its `secret.json` file, and the `data.json` file is passed sequentially through the ring. The final result aggregates contributions from all nodes. The core logic is implemented in two key files:

1. **`main.py`:** Manages the computation logic.
2. **`run.sh`:** Automates the execution of the ring computation process.

Here’s an overview of the key functionality in `main.py`:

#### **Extracts of `main.py` Explained**

- **Loading and Updating Ring Data:**
  ```python
  def load_ring_data(file_path: Path):
      ring_data = load_json(file_path)
      return ring_data["participants"], ring_data["data"], ring_data["current_index"]

  def create_ring_data(participants: List[str], data: int, current_index: int):
      return {"participants": participants, "data": data, "current_index": current_index}
  ```
  These functions handle reading and updating the ring’s data, including participants, the current index, and the aggregated value.

- **Processing Each Node’s Contribution:**
  ```python
  data += my_secret
  next_index = current_index + 1

  if next_index < len(ring_participants):
      next_person = ring_participants[next_index]
      new_ring_data = create_ring_data(ring_participants, data, next_index)
      receiver_ring_data = client.api_data("ring", datasite=next_person) / "running" / "data.json"
      write_json(receiver_ring_data, new_ring_data)
  else:
      print(f"Terminating ring, writing back to {DONE_FOLDER}")
      final_ring_data = create_ring_data(ring_participants, data, current_index)
      write_json(DONE_FOLDER / "data.json", final_ring_data)
  ```
  - Each node increments the `data` value using its secret.
  - If there are more participants, the updated data is passed to the next node.
  - Once all participants contribute, the computation terminates, and the final result is saved in the `done` folder.

- **Cleaning Up Processed Data:**
  ```python
  file_path.unlink()
  print(f"Done processing {file_path}, removed from pending inputs")
  ```
  Removes the processed `data.json` file to prevent duplicate processing.

---

### **Provisioning Additional Nodes**

It is time for the other ring collaborators node. I'll prepare the ring code needed to provision those nodes.

I will create a ring folder and copy the necessary files so nodes can compute the task to complete the ring computation:

```sh
mkdir /tmp/ring
cp ring/run.sh ring/main.py ring/utils.py ring/secret.json /tmp/ring/.
# Modify the secret data for the syftbox node, before sending the data to this node
vim /tmp/ring/secret.json
# {"data": 0.4}
# Copy the data to the syftbox node

docker cp ring syftbox:/build/data/apis/.
```

Provision the last node (thriftbox):
```sh
vim /tmp/ring/secret.json
# {"data": 1}
# Copy the data to the thriftbox node
docker cp ring thriftbox:/build/data/apis/.
```

For each new node (`syftbox` and `thriftbox`), I:

1. Modified their `secret.json` files with unique contributions:
   - For `syftbox`: `{"data": 0.4}`
   - For `thriftbox`: `{"data": 1}`
2. Copied the modified `ring` directory into their respective containers:
   ```sh
   docker cp ring syftbox:/build/data/apis/.
   docker cp ring thriftbox:/build/data/apis/.
   ```
---

### **Starting the Ring Computation**

To begin the computation, I moved `data.json` into the `running` folder of the first node (`driftbox`):

```sh
cd /build/data/datasites/myemail+drift1@gmail.com/api_data/ring
mv data.json running/.
```

This triggered the computation process, where logs from each node confirmed their participation:

```
Found input /build/data/datasites/myemail+syft1@gmail.com/api_data/ring/running/data.json! Let's get to work.
Writing to /build/data/datasites/myemail+thrift1@gmail.com/api_data/ring/running/data.json.
Done processing /build/data/datasites/myemail+syft1@gmail.com/api_data/ring/running/data.json, removed from pending inputs
```

---

### **Final Result**

After completing the ring computation, the aggregated result was:

```json
{"participants": ["myemail+drift1@gmail", "myemail+syft1@gmail.com", "myemail+thrift1@gmail.com", "myemail+drift1@gmail.com"], "data": 3.4, "current_index": 3}
```

This demonstrates the effectiveness of ring computation in reducing server communication while securely aggregating data across distributed nodes.
