import os

TARGET_IMAGES_DIR = "TargetImages"
TARGET_ENCODINGS_DIR = "TargetEncodings"
DATABASE_DIR = "Database"

def makeDirs():
    os.makedirs(TARGET_ENCODINGS_DIR, exist_ok=True)
    os.makedirs(TARGET_ENCODINGS_DIR,exist_ok=True)
    os.makedirs(DATABASE_DIR,exist_ok=True)

