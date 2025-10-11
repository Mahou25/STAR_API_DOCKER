# Étape 1 : Utiliser une image Python officielle
FROM python:3.11-slim

# Étape 2 : Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Étape 3 : Copier les fichiers de ton projet dans le conteneur
COPY . /app

# Étape 4 : Installer les dépendances système (pour scipy, numpy, Pillow, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Étape 5 : Mettre à jour pip et installer les dépendances Python
COPY Requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Étape 6 : Exposer le port de Flask
EXPOSE 5000

# Étape 7 : Définir les variables d'environnement Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=production

# Étape 8 : Démarrer l’application
CMD ["flask", "run"]
