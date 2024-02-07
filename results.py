import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pandas.plotting import parallel_coordinates

# ROC Curve data
roc_data = {
    "Ridge": {"TP": 7, "FP": 7, "TN": 66, "FN": 1},
    "Local": {"TP": 8, "FP": 7, "TN": 65, "FN": 0},
    "Global": {"TP": 6, "FP": 17, "TN": 55, "FN": 2},
    "Local-Global F": {"TP": 6, "FP": 14, "TN": 58, "FN": 2},
    "Ridge-Local F": {"TP": 8, "FP": 4, "TN": 68, "FN": 0}
}

# Calculate TPR and FPR
tpr_fpr_data = {
    algo: {
        "TPR": data["TP"] / (data["TP"] + data["FN"]), 
        "FPR": data["FP"] / (data["FP"] + data["TN"])
    } for algo, data in roc_data.items()
}

# Define the original data
data = {
    "Algorithm": ["Ridge", "Local", "Global", "Local-Global F", "Ridge-Local F"],
    "Accuracy": [0.9012, 0.9125, 0.7625, 0.8, 0.95],
    "Precision": [0.5, 0.5333, 0.2609, 0.3, 0.6667],
    "Recall": [0.875, 1, 0.75, 0.75, 1],
    "F1 Score": [0.6316, 0.6957, 0.3871, 0.4286, 0.8],
    "FMR": [0.0959, 0.0972, 0.2361, 0.1944, 0.0556],
    "FNMR": [0.125, 0, 0.25, 0.25, 0]
}

# Create DataFrame and add TPR, FPR
df = pd.DataFrame(data)
for algo in df['Algorithm']:
    df.loc[df['Algorithm'] == algo, 'TPR'] = tpr_fpr_data[algo]['TPR']
    df.loc[df['Algorithm'] == algo, 'FPR'] = tpr_fpr_data[algo]['FPR']

# Display the DataFrame
print(df)

# Heat Map
plt.figure(figsize=(10, 6))
sns.heatmap(df.set_index('Algorithm'), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap of Fingerprint Algorithm Metrics')
plt.show()

# Parallel Coordinates Plot
plt.figure(figsize=(10, 6))
parallel_coordinates(df.set_index('Algorithm').reset_index(), 'Algorithm', colormap=plt.get_cmap("Set3"))
plt.title('Parallel Coordinates Plot for all Algorithms')
plt.ylabel('Metrics Value')
plt.show()

# Prepare ROC plot
plt.figure(figsize=(10, 6))
for algo, rates in tpr_fpr_data.items():
    plt.plot([0, rates["FPR"], 1], [0, rates["TPR"], 1], label=f'{algo} (area = {auc([0, rates["FPR"], 1], [0, rates["TPR"], 1]):.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
