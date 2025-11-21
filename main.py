import face_recognition
import cv2
import numpy as np
import os
from folder_setup import makeDirs
from encoder import faceEncoder, encodeDatabase
from decoder import compareFaces_match, compareFaces_distance
import pandas as pd


def main(TARGET_PERSON = "Person_1.png"):
    makeDirs()
    Person_1_face_encoding = faceEncoder(TARGET_PERSON)
    UNIQUE_FACE_IDENTIFIER = "_face"
    results = encodeDatabase("Dataset", unique_face_identifier=UNIQUE_FACE_IDENTIFIER)
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

    df = pd.DataFrame({
        "match_filenames": all_matches,
        "distance_filenames": [d["filename"] for d in all_distances],
        "distance": [d["distance"] for d in all_distances]
    })


    df.to_csv("face_results.csv", index=False)

    print("Saved results to face_results.csv")



main("Person1.jpg")



# while True:
#     # Grab a single frame of video
#     ret, frame = video_capture.read()

#     # Only process every other frame of video to save time
#     if process_this_frame:
#         # Resize frame of video to 1/4 size for faster face recognition processing
#         small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

#         rgb_small_frame = small_frame[:, :, ::-1]
#         rgb_small_frame = np.ascontiguousarray(rgb_small_frame[:, :, :3], dtype=np.uint8)

#         face_locations = face_recognition.face_locations(rgb_small_frame)
#         face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

#         face_names = []
#         for face_encoding in face_encodings:
#             # See if the face is a match for the known face(s)
#             matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
#             name = "Unknown"

#             # # If a match was found in known_face_encodings, just use the first one.
#             # if True in matches:
#             #     first_match_index = matches.index(True)
#             #     name = known_face_names[first_match_index]

#             # Or instead, use the known face with the smallest distance to the new face
#             face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
#             best_match_index = np.argmin(face_distances)
#             if matches[best_match_index]:
#                 name = known_face_names[best_match_index]

#             face_names.append(name)

#     process_this_frame = not process_this_frame


#     # Display the results
#     for (top, right, bottom, left), name in zip(face_locations, face_names):
#         # Scale back up face locations since the frame we detected in was scaled to 1/4 size
#         top *= 4
#         right *= 4
#         bottom *= 4
#         left *= 4

#         # Draw a box around the face
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

#         # Draw a label with a name below the face
#         cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
#         font = cv2.FONT_HERSHEY_DUPLEX
#         cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

#     # Display the resulting image
#     cv2.imshow('Video', frame)

#     # Hit 'q' on the keyboard to quit!
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release handle to the webcam
# video_capture.release()
# cv2.destroyAllWindows()