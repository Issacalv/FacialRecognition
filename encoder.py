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

def encodeDatabase(directory="Database", unique_face_identifier= "_face", save_dir="DatabaseEncodings"):
    os.makedirs(save_dir, exist_ok=True)

    all_encodings = {}

    for file in os.listdir(directory):
        image_path = os.path.join(directory, file)

        # Skip non-images
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        base_name, ext = os.path.splitext(file)   # ext includes the dot, e.g. ".jpg"
        full_filename = base_name + ext           # reconstructs "image.jpg"
        full_path = os.path.join(directory, full_filename)  # Database/image.jpg


        print(f"\nProcessing: {image_path}")

        # Load image
        image = face_recognition.load_image_file(image_path)
        database_face_encodings = face_recognition.face_encodings(image)

        print(f"  Found {len(database_face_encodings)} faces in {file}")

        # Process each face encoding
        for i, enc in enumerate(database_face_encodings):
            npy_path = os.path.join(save_dir, f"{base_name}{unique_face_identifier}{i}.npy")

            if not os.path.isfile(npy_path):
                print(f"    Saving new encoding -> {npy_path}")
                np.save(npy_path, enc)
            else:
                print(f"    Loading cached encoding -> {npy_path}")
                enc = np.load(npy_path)

            key = f"{base_name}{unique_face_identifier}{i}"
            all_encodings[key] = {
                "encoding": enc,
                "filename": full_filename,     # e.g. "group.jpg"
                "path": full_path,             # e.g. "Database/group.jpg"
                "face_index": i,
            }

    return all_encodings
