import face_recognition
import cv2
import numpy as np
import os
from folder_setup import makeDirs, DATABASE_DIR
from encoder import faceEncoder, encodeDatabase
from decoder import compareFaces_match, compareFaces_distance
import pandas as pd
from gemini_locator import gemini_fileWriter

def matches_to_df(all_matches, all_distances, TARGET_PERSON, output_dir = "ReportFindings", TOP_N = 3):

    df = pd.DataFrame({
        "match_filename":   [m["filename"] for m in all_matches],
        "match_path":       [m["path"] for m in all_matches],

        "distance_filename": [d["filename"] for d in all_distances],
        "distance_path":     [d["path"] for d in all_distances],
        "distance":          [d["distance"] for d in all_distances],
    })
    
    top_df = df.sort_values(by = "distance").head(TOP_N)
    top_matches = top_df["distance_path"].tolist()

    output_dir = "ReportFindings"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"{TARGET_PERSON}.csv")

    df.to_csv(output_file, index=False)

    print("Saved results to face_results.csv")

    return top_matches

def main(TARGET_PERSON = "Person_1.png"):
    makeDirs()
    Person_1_face_encoding = faceEncoder(TARGET_PERSON)
    UNIQUE_FACE_IDENTIFIER = "_face"
    results = encodeDatabase(DATABASE_DIR, unique_face_identifier=UNIQUE_FACE_IDENTIFIER)
    DISTANCE_THRESHOLD = 0.6
    file_list_match = []
    all_matches = []
    file_list_distance = []
    all_distances = []

    for name, encoding in results.items():
        file_list_match = compareFaces_match([Person_1_face_encoding], face_encoding= encoding)
        file_list_distance = compareFaces_distance([Person_1_face_encoding], face_encoding= encoding, threshold = DISTANCE_THRESHOLD)

        all_matches.extend(file_list_match)
        all_distances.extend(file_list_distance)
    
    TARGET_SAVE = os.path.splitext(TARGET_PERSON)[0]
    top_matches = matches_to_df(all_matches, all_distances, TARGET_SAVE)
    gemini_fileWriter(top_matches, TARGET_SAVE)



main("Person1.jpg")