# **VF Population Model - Afferent Response Simulation**

## **📌 Project Overview**
The **VF Population Model** is a Python-based simulation framework designed to model the firing responses of afferents (SA & RA) to vibratory force (VF) stimuli. The model integrates **spatial** and **radial stress distributions** to compute afferent recruitment, firing patterns, and response dynamics.

This tool provides a structured approach to understanding **afferent population coding**, helping to analyze how mechanoreceptors respond to varying levels of tactile stress in the skin.

---

## **🛠 Features**
- **Spatial Stress Modeling** 🗺️  
  - Simulates afferent activation based on stress data at different spatial coordinates.
  - Computes spike timing, mean firing frequency, and peak firing frequency.
  - Visualizes firing locations and intensity using spatial plots.

- **Radial Stress Modeling** 🎯  
  - Examines afferent response at increasing radial distances from a stimulus center.
  - Extracts spike trains and calculates firing frequency based on stress propagation.
  - Analyzes how afferent response changes with distance from the stimulation site.

- **Cumulative Firing & Recruitment Analysis** 📊  
  - Tracks how afferents are recruited over time.
  - Compares firing frequency across different VF tip sizes.
  - Generates cumulative firing and afferent recruitment plots.

- **Sensitivity Analysis** 🔍  
  - Tests how different model parameters (e.g., τ-values, k-values, scaling factors) impact afferent firing.
  - Helps in optimizing and refining the model for realistic tactile response simulations.

- **Visualization Tools** 🎨  
  - Generates **heatmaps, scatter plots, radial plots, and grid-based** representations of afferent activity.
  - Allows comparative analysis of different VF tip sizes and afferent types.

---

## **📂 Project Structure**
├── data/                # Contains stress data for different VF sizes and densities
│   ├── P2/              # Processed stress data categorized by VF size and density
│   ├── spatial/         # Spatial stress data
│   ├── radial/          # Radial stress data
│
├── vf_popul_model.py    # Main class for simulating afferent population responses
├── readme.md            # Project documentation
├── requirements.txt     # Dependencies required for running the model
└── vf_graphs/           # Output directory for visualizations

---

## **🚀 Getting Started**
### **1️⃣ Installation**
Clone this repository and install the required dependencies:
```bash
git clone https://github.com/your-repo/VF_Population_Model.git
cd VF_Population_Model
pip install -r requirements.txt
from vf_popul_model import VF_Population_Model

# Initialize the model with VF tip size, afferent type, and scaling factor
vf_model = VF_Population_Model(vf_tip_size=3.61, aff_type="SA", scaling_factor=1.0)

# Run radial stress model
vf_model.radial_stress_vf_model()

# Run spatial stress model
vf_results = vf_model.spatial_stress_vf_model()

# Visualize spatial afferent activity
vf_model.plot_spatial_coords()

# Perform cumulative afferent recruitment analysis
VF_Population_Model.cumulative_afferent_over_time("SA")
