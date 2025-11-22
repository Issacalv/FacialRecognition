import face_recognition

def compareFaces_match(Target_Encoding=None, face_encoding=None):
    file_list = []
    recognition = face_recognition.compare_faces(Target_Encoding, face_encoding["encoding"])

    if recognition[0] == True:
        file_list.append({
            "filename": face_encoding["filename"],
            "path": face_encoding["path"]
        })

    return file_list

    

def compareFaces_distance(Target_Encoding=None, face_encoding=None, threshold=None):
    file_list = []
    face_distances = face_recognition.face_distance(Target_Encoding, face_encoding["encoding"])

    for face_distance in face_distances:
        if face_distance < threshold:
            file_list.append({
                "filename": face_encoding["filename"],
                "path": face_encoding["path"],
                "distance": float(face_distance)
            })

    return file_list
