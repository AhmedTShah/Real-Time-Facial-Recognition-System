from deepface import DeepFace
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

# List of available models and distance metrics
models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
metrics = ["cosine", "euclidean", "euclidean_l2"]

# Path to the image for face recognition
img = "TrainingData/hanan1.jpg"

def face_recognition(img):
    # Perform face recognition on a static image
    people = DeepFace.find(
        img_path=img,
        db_path="TrainingData",
        detector_backend='opencv',
        model_name=models[2],  # Facenet512
        distance_metric=metrics[1],  # euclidean
        enforce_detection=True
    )

    # Display the original image
    img_data = cv2.imread(img)
    img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
    plt.imshow(img_data)
    plt.axis('off')
    plt.show()

    # Print the identities of recognized people
    if isinstance(people, list):
        df = people[0]
    else:
        df = people

    for _, row in df.iterrows():
        name = os.path.splitext(os.path.basename(row['identity']))[0]
        print("Identified:", name)

def realtime_face_recognition():
    vid = cv2.VideoCapture(0)

    while True:
        ret, frame = vid.read()
        if not ret:
            break

        temp_path = "temp.jpg"
        cv2.imwrite(temp_path, frame)

        try:
            people = DeepFace.find(
                img_path=temp_path,
                db_path="TrainingData",
                detector_backend='opencv',
                model_name=models[2],  # Facenet512
                distance_metric=metrics[2],  # euclidean_l2
                enforce_detection=False
            )

            if isinstance(people, list):
                df = people[0]
            else:
                df = people

            names_displayed = set()
            for _, row in df.iterrows():
                name = os.path.splitext(os.path.basename(row['identity']))[0]
                names_displayed.add(name)

            y_offset = 30
            for name in names_displayed:
                cv2.putText(frame, name, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                y_offset += 40

        except Exception as e:
            print("Recognition error:", e)

        # Display the resulting frame
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', 960, 720)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

    if os.path.exists(temp_path):
        os.remove(temp_path)

# Uncomment to test static image recognition
# face_recognition(img)

# Run real-time face recognition
realtime_face_recognition()
