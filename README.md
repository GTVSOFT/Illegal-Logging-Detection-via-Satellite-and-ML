Illegal Logging Detection via Satellite and Machine Learning

The approach is adaptable to real-world satellite datasets (e.g., Sentinel-2, Landsat-8/9) for forest monitoring, deforestation mapping, and environmental enforcement.
🚀 Features

    Generates >100 synthetic observation points within a defined AOI

    Simulates satellite spectral bands (Red, NIR, SWIR)

    Calculates vegetation health index (NDVI)

    Simulates surface texture features for detecting canopy disturbance

    Computes a logging probability score and classifies pixels

    Ready for adaptation with real-world remote sensing imagery

    Dataset exported in Excel format for reproducibility

📂 Dataset

The synthetic dataset used in this example can be downloaded here:
📥 synthetic_illegal_logging_points.xlsx

Columns included:
Column	Description
id	Unique identifier for each observation
latitude	Latitude coordinate
longitude	Longitude coordinate
band1	Simulated Red band reflectance
band2	Simulated NIR band reflectance
band3	Simulated SWIR band reflectance
ndvi	Normalized Difference Vegetation Index
texture	Simulated surface texture
logging_prob	Probability of illegal logging (0–1)
is_logged	Binary classification (1 = Logged, 0 = Non-logged)
🛠 Requirements
Python environment (for data generation & ML experiments)

    Python 3.8+

    pandas

    numpy

pip install pandas numpy

Optional (for visualization)

    matplotlib

    geopandas

    scikit-learn

pip install matplotlib geopandas scikit-learn

📌 Usage
1. Generate the dataset

Run the provided Python script to generate synthetic illegal logging points:

import pandas as pd
import numpy as np

# Example: Dataset generation code here...

This will create synthetic_illegal_logging_points.xlsx.
2. Train an ML Model (Optional)

You can use scikit-learn to train a classifier:

from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load dataset
df = pd.read_excel("synthetic_illegal_logging_points.xlsx")

# Prepare features & labels
X = df[["band1", "band2", "band3", "ndvi", "texture"]]
y = df["is_logged"]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Check accuracy
print("Training Accuracy:", model.score(X, y))

3. Adapt to Real Satellite Data

Replace the synthetic dataset with:

    Sentinel-2 MSI data (Bands 4, 8, 11)

    Landsat-8/9 OLI data (Bands 4, 5, 6)

    Apply similar preprocessing (NDVI, texture) before training

📊 Potential Applications

    Forest monitoring and enforcement

    Near real-time illegal logging alerts

    Environmental conservation and policy-making

    Supporting REDD+ and carbon credit monitoring projects

👨‍💻 Author

Name: Amos Meremu Dogiye
GitHub: https://github.com/GTVSOFT
