gdown https://drive.google.com/uc?id=1eShRtrCNUuxNtRzsQpR1ERpYrMFa22dj
unzip -q archive.zip
rm Data/genres_original/jazz/jazz.00054.wav
python -m extract_embeddings.py