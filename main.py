
import cv2
import numpy as np
import mediapipe as mp
import argparse
import os

TEMPLATE_PATH = "templates/templates.npy"

mp_hands = mp.solutions.hands

def record_templates():
    print("Recording templates...")
    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    data = {}
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(static_image_mode=False,max_num_hands=1,min_detection_confidence=0.5) as hands:
        for label in labels:
            samples=[]
            print(f"Show gesture for: {label} – press SPACE to capture")
            while True:
                ret,frame = cap.read()
                if not ret: break
                img = cv2.flip(frame,1)
                rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                res = hands.process(rgb)

                cv2.putText(img,f"Show: {label}",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

                cv2.imshow("Setup",img)
                key=cv2.waitKey(1)
                if key==32:
                    if res.multi_hand_landmarks:
                        lm=[]
                        for lmset in res.multi_hand_landmarks:
                            for p in lmset.landmark:
                                lm.append([p.x,p.y,p.z])
                        samples.append(np.array(lm).flatten())
                        print("Captured sample")
                        break
                if key==ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
            data[label]=samples

    cap.release()
    cv2.destroyAllWindows()
    np.save(TEMPLATE_PATH, data)
    print("Templates saved.")

def run_detector():
    if not os.path.exists(TEMPLATE_PATH):
        print("Templates file missing. Run --setup first.")
        return

    data = np.load(TEMPLATE_PATH,allow_pickle=True).item()
    templates = {k: np.mean(v,axis=0) for k,v in data.items()}

    cap=cv2.VideoCapture(0)
    with mp_hands.Hands(max_num_hands=1) as hands:
        while True:
            ret,frame=cap.read()
            if not ret: break
            img=cv2.flip(frame,1)
            rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            res=hands.process(rgb)

            text="No Hand"

            if res.multi_hand_landmarks:
                lm=[]
                for lmset in res.multi_hand_landmarks:
                    for p in lmset.landmark:
                        lm.append([p.x,p.y,p.z])
                arr=np.array(lm).flatten()

                best=None
                best_dist=1e9
                for label,temp in templates.items():
                    d=np.linalg.norm(arr-temp)
                    if d<best_dist:
                        best_dist=d
                        best=label
                text=best

            cv2.putText(img,f"Detected: {text}",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            cv2.imshow("Detector",img)
            if cv2.waitKey(1)==ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup",action="store_true")
    parser.add_argument("--run",action="store_true")
    args = parser.parse_args()

    if args.setup: record_templates()
    elif args.run: run_detector()
    else:
        print("Use --setup or --run")
