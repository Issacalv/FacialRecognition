import face_recognition

def compareFaces_match(Target_Encoding = None, face_encoding = None):
    file_list = []
    recognition = face_recognition.compare_faces(Target_Encoding, face_encoding["encoding"])
    if recognition[0] == True:
        file_name = face_encoding["filename"]
        print(f"Facial Recognition Match in file name: '{file_name}'")
        file_list.append(file_name)
    return file_list
    

def compareFaces_distance(Target_Encoding = None, face_encoding = None, threshold = None):
    file_list = []
    face_distances = face_recognition.face_distance(Target_Encoding, face_encoding["encoding"])
    for i, face_distance in enumerate(face_distances):
        if face_distance < threshold:
            filename = face_encoding["filename"]
            print(f"Person match found in {filename}")
            print("The test image has a distance of {:.2} from known image".format(face_distance, i))
            file_list.append({
                "filename": filename,
                "distance": float(face_distance)
            })


    return file_list