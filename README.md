# Disaster Impact Assessment (DIA) Challenge
### Multimodal Deep Learning for Building Damage Classification
#### VIA Madrid Summer School in Vision and Artificial Intelligence, 2nd Edition - 15th-17th June 2026

A deep learning framework for building damage classification using satellite imagery and climate data, built for the VIA Summer School Challenge.

## Overview

VIA-DIA fuses pre/post-disaster satellite image pairs with ERA-5 climate variables to classify building damage into four levels: **no damage**, **minor**, **major**, and **destroyed**. It is built on [PyTorch Lightning](https://lightning.ai/) and trains on the [xBD dataset](https://xview2.org/).

## Repository Structure

```
VIA-DIA/
├── main.py                          # Training / testing / interpretation entry point
├── lightning_module.py              # PyTorch Lightning module (loss, metrics, logging)
├── preprocess_xbd.py                # xBD dataset preprocessing → memory-mapped cache
├── configs/
│   ├── config_baseline.yaml         # Baseline model config
│   └── config_interpretation.yaml   # Interpretation / saliency config
├── models/
│   ├── baseline_model.py            # CNN + MLP multimodal baseline
│   ├── image_difference_model.py    # Image-only CNN difference model
│   ├── image_weather_gru_model.py   # Image-difference + weather-GRU model
│   ├── weather_gru_model.py         # Weather-only GRU temporal model
│   └── shared_components.py         # Shared building blocks
└── databases/
    └── xBDClimate_database.py       # Memory-mapped xBD + ERA-5 PyTorch dataset
```


## Train the Baseline Model

Use the baseline notebook or 

```bash
python main.py -c configs/config_baseline.yaml -a BaselineMultimodalModel -d xBDClimate
```

## Train the Image-Only Difference Model

Use the same config and switch only the architecture argument:

```bash
python main.py -c configs/config_baseline.yaml -a ImageDifferenceCNNModel -d xBDClimate
```

## Train the Weather-Only GRU Model

Use the same config and switch only the architecture argument:

```bash
python main.py -c configs/config_baseline.yaml -a WeatherGRUModel -d xBDClimate
```

## Train the Image + Weather GRU Model

Use the same config and switch only the architecture argument:

```bash
python main.py -c configs/config_baseline.yaml -a ImageDifferenceWeatherGRUModel -d xBDClimate
```

#### Author

Miguel Ángel Fernández-Torres

Code implementation assisted by [Claude](https://claude.ai) (Anthropic).
