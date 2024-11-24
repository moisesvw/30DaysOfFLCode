
### **Day 3: Exploring Differential Privacy (DP) and its Role in Mitigating Privacy Risks**

Today, I delved into the foundational concepts of **Differential Privacy (DP)** and explored its importance in addressing **privacy risks** in data analysis and Federated Learning (FL). To make this exploration practical, I conducted an experiment to demonstrate how the absence of DP mechanisms can lead to the reidentification of a single processed record by an adversary, linking it back to the original dataset. This highlights the critical role DP plays in protecting sensitive data and ensuring that individual contributions remain private, even in collaborative machine learning systems like FL.

## **Key Learnings**

1. **Sensitivity**:
   - Measures how much a query result (e.g., average) can change when a single individual’s data is added or removed from a dataset.
   - High sensitivity increases the risk of exposing sensitive information, making it critical to control or obscure such impacts.

2. **Differential Privacy (DP)**:
   - A formal framework ensuring that the inclusion or exclusion of a single data point does not significantly affect the outcome of analysis.
   - Achieved by adding mathematically calibrated noise to results or updates.

---

## **The Experiment**

I worked with subsets created in **Day 2**, simulating private datasets, and attempted to reidentify a processed record in the original dataset.

### **Step 1: Load a Processed Record**
A processed record contains only derived features such as `distance`, `amt`, and `category`. Here’s how I extracted a random record:

```python
import pandas as pd

# Load processed data
subset = pd.read_csv("subset_1.csv")

# Select a random record
registro_procesado = subset.sample(n=1).iloc[0]
print("Processed record:", registro_procesado)
```

---

### **Step 2: Calculate Distances for Reidentification**

Using attributes like `distance` (calculated in preprocessing), `amt`, and `city_pop`, I calculated the **Euclidean distance** between the processed record and all records in the original dataset.

#### **Code:**
```python
import numpy as np
from geopy.distance import geodesic

# Load the original dataset
original = pd.read_csv("fraudTrain.csv")

# Add calculated distances back into the original dataset
def calcular_distancia_vectorizado(lat1, long1, lat2, long2):
    return np.vectorize(
        lambda lat1, long1, lat2, long2: geodesic((lat1, long1), (lat2, long2)).km
    )(lat1, long1, lat2, long2)

original['distance'] = calcular_distancia_vectorizado(
    original['lat'], original['long'], original['merch_lat'], original['merch_long']
)

# Calculate Euclidean distance for each record
def distancia_euclidiana(row, registro):
    return np.sqrt((row['amt'] - registro['amt'])**2 + 
                   (row['distance'] - registro['distance'])**2 + 
                   (row['city_pop'] - registro['city_pop'])**2)

original['euclidean_distance'] = original.apply(
    lambda row: distancia_euclidiana(row, registro_procesado),
    axis=1
)

# Find records with a small Euclidean distance
umbral_distancia = 0.1
candidatos_df = original[original['euclidean_distance'] < umbral_distancia]

print("Matching records:")
print(candidatos_df)
```

---

### **Step 3: Visualizing the Compromised Record**

I created a visualization to emphasize the compromised record, highlighting sensitive details like `merchant`, `cc_num`, and `trans_num`.

#### **Code for Visualization:**
```python
from PIL import Image, ImageDraw, ImageFont

def generate_detected_image(record, output_file="detected_info.png"):
    image = Image.new("RGB", (1000, 900), "black")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    
    draw.text((50, 50), "Privacy Compromised - Match Found", fill="red", font=font_large)
    
    y_offset = 100
    for key, value in record.items():
        draw.text((50, y_offset), f"{key}: {value}", fill="white", font=font_large)
        y_offset += 30
    
    image.save(output_file)

record_detected = candidatos_df.iloc[0].to_dict()
generate_detected_image(record_detected)
```

---

### **Step 4: Mitigation Using Differential Privacy**

To mitigate the risk, I applied **Laplace noise** to sensitive attributes like `amt` and `distance`. This noise prevents exact matches during reidentification attempts.

#### **Code with Noise:**
```python
epsilon = 1.0  # Privacy parameter
original['amt_noisy'] = original['amt'] + np.random.laplace(scale=1/epsilon, size=len(original))
original['distance_noisy'] = original['distance'] + np.random.laplace(scale=1/epsilon, size=len(original))

def distancia_euclidiana_noisy(row, registro):
    return np.sqrt((row['amt_noisy'] - registro['amt'])**2 + 
                   (row['distance_noisy'] - registro['distance'])**2)

original['euclidean_distance_noisy'] = original.apply(
    lambda row: distancia_euclidiana_noisy(row, registro_procesado),
    axis=1
)

# Find matching records with noisy distances
candidatos_df_noisy = original[original['euclidean_distance_noisy'] < umbral_distancia]
print("Matching records after noise:")
print(candidatos_df_noisy)
```

---

## **Results**

1. **Before Applying Noise**:
   - The adversary successfully identified a match and extracted sensitive details like the `merchant` and `cc_num`.

2. **After Applying Noise**:
   - The adversary could no longer find an exact match, protecting individual privacy.

---

## **Relation to Differential Privacy Principles**

1. **Parallel Databases**:
   - I measured the impact of including or excluding a record (sensitivity) on query results like distances.

2. **Empirical Sensitivity**:
   - The Euclidean distance calculation showed the significant influence of certain attributes, guiding noise addition.

3. **Data Utility and Privacy Trade-off**:
   - Noise addition effectively mitigated reidentification risk while maintaining dataset utility for aggregated analyses.

---

## **Conclusion**

This experiment illustrates how sensitivity directly impacts privacy risks in Federated Learning. By integrating Differential Privacy techniques, we can protect sensitive data during collaborative model training while ensuring fairness and robustness in **FedAvg**.

