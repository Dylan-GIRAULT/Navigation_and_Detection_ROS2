#!/bin/bash

# --- CONFIGURATION DES CHEMINS ---
WS_PATH="$HOME/Documents/sy27-zoe/workspace"
VENV_PATH="$HOME/pcdet_env"
RC_FILE="$HOME/.bashrc"

echo "Configuration des alias..."

# --- 1. AJOUT DES ALIAS DANS LE .BASHRC ---
# On vérifie si l alias existe déjà avant de l'ajouter pour éviter les doublons

add_alias() {
    if ! grep -q "alias $1=" "$RC_FILE"; then
        echo "alias $1='$2'" >> "$RC_FILE"
        echo "Alias $1 ajouté."
    fi
}

add_alias "cb" "colcon build --symlink-install"
add_alias "sws" "source install/setup.bash"
add_alias "sy27" "cd $WS_PATH"
add_alias "perc" "cd $WS_PATH && source $VENV_PATH/bin/activate && source install/setup.bash"

# --- 2. GESTION DES GIT CREDENTIALS  ---
git config --global credential.helper 'cache --timeout=3600'
echo " Git credentials : Le cache est activé pour 1 heure."

# --- 1. INSTALLATION DES OUTILS ---
echo " Installation de Rviz2, RQT et PlotJuggler"
sudo apt update
# Installation de Rviz2 et RQT (standard ROS 2 Jazzy)
sudo apt install -y ros-jazzy-rviz2 ros-jazzy-rqt ros-jazzy-rqt-common-plugins
# Installation de PlotJuggler (version ROS 2)
sudo apt install -y ros-jazzy-plotjuggler-ros

echo "--------------------------------------------------------"
echo ":) Configuration terminée !"
echo " Tape 'source ~/.bashrc' pour activer les changements."
echo "Commandes disponibles :"
echo "   sy27  : Va direct dans le workspace"
echo "   cb    : Compile colcon build"
echo "   sws   : Source le workspace"
echo "   perc  : Prépare TOUT (Venv + sourcer) pour lancer ton node"
echo "--------------------------------------------------------"
