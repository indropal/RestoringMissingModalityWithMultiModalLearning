import os

# the absolute path of the project directory -> where object_encapsulate.py. paths.py, config.py & data directory are
projectDir = "C:\\Users\\palin\\Desktop\\Profile_Projects\\Generative_Multimodel_Learning_for_Reconstructing_Missing_Modality"
#projectDir = os.getwcd()

dataDir = os.path.join(projectDir, "data")
recDataDir = os.path.join(dataDir, "recordings")
mnistDataDir = os.path.join(dataDir, "mnist")
speechDataDir = os.path.join(dataDir, "speech")
lookupEmbeddingDir = os.path.join(dataDir, "lookupEmbedding")
OUT_DIR = os.path.join(projectDir, "models")