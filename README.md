# Construction de modèles de Deep learning pour SCA 

## 🎯 Objectif du projet
Ce projet vise à développer et expérimenter une architecture hybride combinant **Convolutional Neural Networks (CNN)** et **Transformers** pour l’analyse de traces issues d’attaques par canaux auxiliaires (Side-Channel Attacks, SCA).
L’objectif est d’améliorer les performances de classification et de récupération de clés sur des jeux de données tels que **ASCAD** et **AES_HD**.



## 📂 Structure du dépôt
- `architecture_CNN.py` : définition d’une première architecture CNN simple. Cette architecture est associée avec la base de donnée 'AES_HD_dataset/'.
- `architecture_Transformer.py`  : définition de l’encodeur Transformer. On a utilise la base de données 'ASCAD_dataset/'
- `ASCAD_dataset/` : dataset ASCAD (stocké via Git LFS).
- `AES_HD_dataset/` : dataset AES-HD (stocké via Git LFS).
- `notebooks/` : notebooks Jupyter utilisés pour les expérimentations.



## ⚙️ Installation
1. **Cloner le dépôt :**
   ```bash
   git clone https://github.com/kolon-lab/Architecture_hybride_CNN-Transformer.git
   cd Architecture_hybride_CNN-Transformer

