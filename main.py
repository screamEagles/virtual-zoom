import cv2
from cvzone.HandTrackingModule import HandDetector  # cvzone version: 1.5.0

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8)
start_distance = None
scale = 0
centre_x, centre_y = 500, 500

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    img1 = cv2.imread("ainsley.jpg")

    if len(hands) == 2:
        if detector.fingersUp(hands[0]) == [1, 1, 0, 0, 0] and detector.fingersUp(hands[1]) == [1, 1, 0, 0, 0]:
            landmark_list_1 = hands[0]["lmList"]
            landmark_list_2 = hands[1]["lmList"]

            if start_distance is None:
                # 8 is the index finger's tip
                length, info, img = detector.findDistance(landmark_list_1[8], landmark_list_2[8], img)
                start_distance = length
        
            length, info, img = detector.findDistance(landmark_list_1[8], landmark_list_2[8], img)
            scale = int((length - start_distance) // 2)
            centre_x, centre_y = info[4:]
            # print(scale)
    else:
        start_distance = None

    try:
        height1, width_1, _ = img1.shape
        new_height, new_width = max(2, ((height1 + scale) // 2) * 2), max(2, ((width_1 + scale) // 2) * 2)
        img1 = cv2.resize(img1, (new_width, new_height))
        img[
            max(0, centre_y - new_height // 2):min(img.shape[0], centre_y + new_height // 2),
            max(0, centre_x - new_width // 2):min(img.shape[1], centre_x + new_width // 2)
        ] = img1
    except:
        pass

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
