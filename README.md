# Virtual Stroke CERNN

This repository contains the code and results for **Virtual Stroke: Investigating How Neural Damage Affects Motor Tasks**, an MSc thesis project.  
The project extends the **Cortically Embedded Recurrent Neural Network (CERNN)** framework to simulate virtual lesions in cortical areas and study their effect on visuomotor performance.

---

## Requirements

Install the dependencies using:

```bash
pip install -r requirements.txt
You may need to separately install PyTorch with the correct CUDA version for your machine:
PyTorch Installation Guide

## Training
To train the model (default configuration):

bash
Copy code
python src/train_rnn_multitask.py
Training is controlled by Hydra configuration files inside src/hydraconfigs/.
You can adjust parameters such as epochs, batch_size_train, etc. in local.yaml.

## Evaluation
Mean Squared Error (MSE) lesioning
To evaluate MSE differences between healthy and lesioned networks:

bash
Copy code
python eval_mse.py --areas L_V1 L_V2 L_V3 L_V4
Accuracy comparison
To evaluate angular accuracy (± degrees):

bash
Copy code
python eval_acc.py --areas L_FEF
You can also list available cortical areas with:

bash
Copy code
python eval_mse.py --list-areas


## Results
All analysis results are stored in CSV format for easy inspection:

results_300.csv → performance after 300 epochs

results_500.csv → performance after 500 epochs

results_300_dorsal.csv, results_500_dorsal.csv → lesion results for Dorsal Attention Network

results_300_fpn.csv, results_500_fpn.csv → lesion results for Frontoparietal Network

*_per_rule.csv → breakdown per individual task (e.g. fdgo, delayanti, etc.)

## Project Overview
Model: Cortically Embedded RNN (CERNN)

Inputs: Sensory (visual L_V1, somatosensory L_3b)

Outputs: Motor (L_FEF)

Lesioning method: zeroing out hidden states of selected cortical areas during forward pass

Metrics: Mean Squared Error (MSE), angular accuracy

This framework allows systematic investigation of how lesioning specific cortical areas (Visual cortex, Dorsal Attention Network, Frontoparietal Network, etc.) alters task performance.
