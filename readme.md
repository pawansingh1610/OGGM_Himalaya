# Drang Drung Glacier — Multi‑SSP Simulation & 3‑D Visualization Pipeline

This repository contains a complete, reproducible workflow for **glacier evolution modelling, 2‑D redistribution, and advanced 3‑D visualisation** using **OGGM**, **ISIMIP3b climate data**, and **Glacier3DViz**. The pipeline simulates glacier response under multiple future climate scenarios and produces high‑quality static figures and animations.

---

## 📌 Overview

**Glacier:** Drang Drung (RGI60‑14.18948), Karakoram, India  
**Simulation Period:** 2000–2100  
**Scenarios:** SSP1‑2.6, SSP2‑4.5, SSP3‑7.0, SSP5‑8.5  
**GCM:** MRI‑ESM2‑0 (ISIMIP3b)

This workflow:

- Runs OGGM glacier simulations
- Generates gridded ice thickness
- Produces standardised NetCDF outputs
- Creates 3‑D animations and satellite visualisations
- Generates comparison plots and multi‑panel videos

The pipeline is designed to be:

- ✔ Reproducible
- ✔ Scalable
- ✔ Modular
- ✔ Fast (optional bias mode)
- ✔ Publication‑ready

---

## 🚀 Features

### Climate Scenarios

Supports multiple future emissions pathways:

- SSP1‑2.6 (low emissions)
- SSP2‑4.5 (intermediate)
- SSP3‑7.0 (high)
- SSP5‑8.5 (very high)

### Fast vs Full Simulation Modes

Two simulation strategies:

#### ⚡ Fast Mode

- Downloads only SSP3‑7.0 climate
- Other SSPs generated via temperature bias
- Minimal data download (~500 MB)
- Runtime ~10 minutes

#### 🔬 Full Mode

- Independent ISIMIP3b climate per SSP
- Higher scientific rigor
- Larger download (~2 GB)

---

## 🧭 Workflow Structure

The pipeline consists of three major phases:

### Phase 1 — OGGM Simulation

- Downloads Level‑3 glacier directory
- Runs historical baseline
- Processes climate forcing
- Simulates future glacier evolution
- Distributes 1‑D results to 2‑D grid
- Saves NetCDF output

### Phase 2 — 3‑D Visualisation

For each SSP scenario:

- Static snapshots (2000, 2050, 2100)
- Full animation (terrain)
- Animation with globe inset
- Satellite‑texture visualisation

### Phase 3 — Multi‑Scenario Comparison

- 4‑panel animation
- Rotating camera views
- Satellite comparison
- Area and thickness time‑series

---

## 📂 Output Products

### Static

- Glacier evolution maps
- Scenario comparison figures
- Time‑series plots

### Animations

- Terrain animations
- Satellite visualisations
- Rotating camera views
- Multi‑scenario videos

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

### 2. Create environment

```bash
conda create -n glacier3dviz python=3.10
conda activate glacier3dviz
```

### 3. Install dependencies

```bash
pip install oggm glacier3dviz pyvista tqdm
conda install -c conda-forge ffmpeg
```

Optional:

```bash
pip install pillow
```

---

## ▶️ Usage

Run the main pipeline:

```bash
python durung_drong_pipeline.py
```

First run:

```python
RESET_GDIR = True
```

Later runs:

```python
RESET_GDIR = False
```

---

## 🔧 Configuration

The script is highly configurable. Key parameters:

| Parameter | Description |
|------|------|
| `RGI_ID` | Glacier identifier |
| `ISIMIP_MEMBER` | Climate model |
| `SSP_SCENARIOS` | Scenario list |
| `USE_BIAS_OFFSETS` | Fast vs full mode |
| `SIM_START`, `SIM_END` | Simulation years |
| `PLOTTER_ARGS` | Rendering resolution |


### Visualisation

You can modify:

- Camera angle
- Colour maps
- Annotation labels
- Animation speed
- Video resolution


### Satellite imagery

Toggle:

```python
RENDER_SATELLITE = True
```

---

## 🎥 Customisation

### Camera

Adjust:

- Rotation arc
- Viewing direction
- Zoom


### Annotations

Includes:

- Glacier outline
- Summit markers
- North arrow
- Labels

Add custom annotations by modifying the script.

---

## 📊 Scientific Notes

- Uses OGGM v1.6
- ISIMIP3b glacier‑point climate
- 2‑D redistribution from flowline
- Bias‑offset scenario scaling

The fast bias approach is suitable for:

- Sensitivity experiments
- Communication
- Scenario exploration

Full mode is recommended for:

- Publication
- Detailed climate impact studies

---

## 📌 Reproducibility

The pipeline:

- Caches climate data
- Stores processed NetCDF
- Ensures consistent grid and CRS

Deleting output files will trigger re‑simulation.

---

## 🤝 Contributions

Contributions are welcome. Suggested improvements:

- New glaciers
- Regional studies
- Coupling with remote sensing
- Uncertainty quantification


---

## 📜 License

Add your preferred license here.

---

## 📧 Contact

Pawan Singh  
Indian Institute of Technology Roorkee

For questions, collaborations, or feedback, feel free to reach out.

---

## ⭐ Citation

If you use this workflow, please cite:

- OGGM
- Glacier3DViz
- ISIMIP3b

And reference this repository.

---

## 🌍 Future Extensions

Potential future developments:

- Ensemble GCMs
- Dynamic calving
- Coupling with hydrology
- Glacier hazard modelling
- Regional Himalayan glacier projections


---

**This repository provides a complete end‑to‑end solution for glacier modelling and visualisation under climate change.**

