# Day 12: Unpacking PATE: Fine-Tuning Privacy and Accuracy

Continuing my exploration of **Private Aggregation of Teacher Ensembles (PATE)** from Day 11, today I evaluated the **Student Model** and extended the analysis to understand the nuances of privacy, performance trade-offs, and scalability. PATE has demonstrated its potential for privacy-preserving machine learning, and these experiments provided valuable insights into optimizing its implementation.

---

## **Objectives for Day 12**

1. **Evaluate the Student Model's performance** on unseen data (test set).  
2. **Analyze the impact of privacy parameters (ϵ)** on model accuracy.  
3. Explore advanced scenarios, such as varying the number of teachers and measuring privacy costs.

---

## **Step 1: Evaluate the Student Model on Test Data**

To evaluate the Student Model's generalization ability:  
- Split the original dataset into training and test sets.  
- Measure accuracy on the test set.  

### Implementation

```python
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Split original data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluate the Student Model
y_pred_student = student_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_student)
print(f"Student Model Accuracy on Test Set: {accuracy:.2f}")
```

---

## **Step 2: Analyze the Impact of Privacy Parameter (ϵ)**

The privacy parameter **ϵ (epsilon)** controls the trade-off between privacy and model accuracy:  
- Lower ϵ: More noise, better privacy, lower accuracy.  
- Higher ϵ: Less noise, weaker privacy, higher accuracy.  

### Implementation

```python
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def aggregate_labels_with_noise(teachers, data, epsilon):
    """Aggregate labels with Laplacian noise for differential privacy."""
    noisy_labels = []
    for x in data:
        predictions = [teacher.predict(x.reshape(1, -1))[0] for teacher in teachers]
        counts = np.bincount(predictions, minlength=2)
        noise = np.random.laplace(0, 1/epsilon, counts.shape)
        noisy_labels.append(np.argmax(counts + noise))
    return noisy_labels

# Define epsilon values and analyze accuracy
epsilons = [0.1, 0.5, 1.0, 2.0, 5.0]
accuracies = []

for epsilon in epsilons:
    # Generate private labels with the current epsilon
    private_labels = aggregate_labels_with_noise(teacher_models, X_unlabeled, epsilon)
    
    # Train a new Student Model
    student_model = LogisticRegression(random_state=42)
    student_model.fit(X_unlabeled, private_labels)
    
    # Evaluate accuracy on the test set
    y_pred_student = student_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_student)
    accuracies.append(accuracy)
    print(f"Epsilon: {epsilon}, Accuracy: {accuracy:.2f}")

# Plot the impact of epsilon on accuracy
plt.figure(figsize=(8, 5))
plt.plot(epsilons, accuracies, marker='o')
plt.title("Impact of Epsilon on Student Model Accuracy")
plt.xlabel("Epsilon (Privacy)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()
```

---

## **Advanced Analyses**

### 1. **Effect of the Number of Teachers**

Experimenting with different numbers of teachers helps balance **labeling quality** and **system complexity**.  

```python
teacher_counts = [3, 5, 10]
accuracies = []

for count in teacher_counts:
    # Split data among the specified number of teachers
    teacher_datasets = np.array_split(np.column_stack((X_train, y_train)), count)
    teacher_models = [RandomForestClassifier(n_estimators=10, random_state=42).fit(data[:, :-1], data[:, -1]) for data in teacher_datasets]
    
    # Generate private labels
    private_labels = aggregate_labels_with_noise(teacher_models, X_unlabeled, epsilon=1.0)
    
    # Train and evaluate the Student Model
    student_model = LogisticRegression(random_state=42)
    student_model.fit(X_unlabeled, private_labels)
    y_pred_student = student_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_student)
    accuracies.append(accuracy)
    print(f"Number of Teachers: {count}, Accuracy: {accuracy:.2f}")
```

### 2. **Privacy Cost Analysis**

Understanding how much privacy budget is consumed over multiple queries is critical.  

```python
def calculate_privacy_cost(epsilon, num_queries):
    """Calculate cumulative privacy cost."""
    return epsilon * num_queries

# Example: Calculate privacy costs for varying numbers of queries
num_queries = [10, 50, 100, 200]
epsilon = 0.5
privacy_costs = [calculate_privacy_cost(epsilon, q) for q in num_queries]

# Display results
for q, cost in zip(num_queries, privacy_costs):
    print(f"Queries: {q}, Privacy Cost (epsilon): {cost:.2f}")
```

---

## **Findings**

### Key Insights
1. **Privacy-Accuracy Trade-off**: Lower ϵ ensures better privacy but can degrade model performance.  
2. **Number of Teachers**: Increasing teachers improves label quality but adds computational overhead.  
3. **Dynamic Privacy Management**: Monitoring privacy budget is crucial to maintain robust differential privacy.  
