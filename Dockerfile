# Obraz bazowy z Jupyter i Pythonem
FROM jupyter/scipy-notebook:latest

# Ustaw katalog roboczy
WORKDIR /workspaces/MLOPS_PZ

# Skopiuj pliki do kontenera
COPY notebooks/ ./notebooks/
COPY README.md ./
COPY requirements.txt ./
COPY start.sh ./

# Instalacja pakietów
RUN pip install --no-cache-dir -r requirements.txt

# Wystawienie portów Jupyter i MLflow
EXPOSE 8888
EXPOSE 5000
