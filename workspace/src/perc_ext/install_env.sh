#!/bin/bash

# 1. Création de l'environnement virtuel
echo "Création de l'environnement pcdet_env..."
virtualenv ~/pcdet_env

# 2. Activation
source ~/pcdet_env/bin/activate

# 3. Installation des piliers (Torch et Spconv) pour CUDA 12.1
echo "Installation de PyTorch et Spconv..."
pip install --upgrade pip
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 
pip install cumm-cu121==0.7.11 spconv-cu121==2.3.8 

# 4. Installation de toutes les autres dépendances
echo "Installation des dépendances du fichier requirements.txt..."
pip install -r requirements.txt 

echo "Environnement pcdet_env prêt !"
