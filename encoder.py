import face_recognition
import os
import numpy as np

def faceEncoder(image_target = "", directory = "TargetImages"):
    root, extension = os.path.splitext(image_target)
    Person_image = face_recognition.load_image_file(f"{directory}/{image_target}")

    Person_numpy_data = f'TargetEncodings/{root}.npy'
    if not os.path.isfile(Person_numpy_data):
        print(f"Numpy file for {root} does not exist, encoding information")
        Person_face_encoding = face_recognition.face_encodings(Person_image)[0]
        np.save(Person_numpy_data, Person_face_encoding)
    else:
        print(f"Numpy file for {root} exists, loading data")
        Person_face_encoding = np.load(Person_numpy_data)
    
    return Person_face_encoding

def encodeDatabase(directory="Dataset", unique_face_identifier="_face", save_dir="DatabaseEncodings"):
    os.makedirs(save_dir, exist_ok=True)

    all_encodings = {}

    for file in os.listdir(directory):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        base_name, ext = os.path.splitext(file)
        image_path = os.path.join(directory, file)

        print(f"\nProcessing: {image_path}")

        cached_files = [
            f for f in os.listdir(save_dir)
            if f.startswith(base_name + unique_face_identifier)
        ]

        if cached_files:
            print("Cached encodings found")

            for idx, npy_file in enumerate(sorted(cached_files)):
                npy_path = os.path.join(save_dir, npy_file)
                enc = np.load(npy_path)

                key = f"{base_name}{unique_face_identifier}{idx}"
                all_encodings[key] = {
                    "encoding": enc,
                    "filename": file,
                    "path": image_path,
                    "face_index": idx,
                }

            continue 
        
        image = face_recognition.load_image_file(image_path)
        database_face_encodings = face_recognition.face_encodings(image)

        print(f" Found {len(database_face_encodings)} faces in {file}")

        for i, enc in enumerate(database_face_encodings):
            npy_path = os.path.join(save_dir, f"{base_name}{unique_face_identifier}{i}.npy")
            print(f"    Saving encoding -> {npy_path}")
            np.save(npy_path, enc)

            key = f"{base_name}{unique_face_identifier}{i}"
            all_encodings[key] = {
                "encoding": enc,
                "filename": file,
                "path": image_path,
                "face_index": i,
            }

    return all_encodings

