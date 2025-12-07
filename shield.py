# libraries
import cv2
import time
import numpy as np
import os
from argparse import ArgumentParser
import pickle
from datetime import datetime, timedelta
import pyvirtualcam
from pyvirtualcam import PixelFormat
import signal
import sys

# ✅ Use cvzone for hand detection
from cvzone.HandTrackingModule import HandDetector

# - INPUT PARAMETERS ------------------------------- #
parser = ArgumentParser()
parser.add_argument("-m", "--model", dest="ML_model", default='models/model_svm.sav',
                    help="PATH of model FILE.", metavar="FILE")
parser.add_argument("-t", "--threshold", dest="threshold_prediction", default=0.9, type=float,
                    help="Threshold for prediction. A number between 0 and 1. default is 0.5")
parser.add_argument("-c", "--camera_id", dest="camera", default=0, type=int,
                    help="ID of the camera. An integer between 0 and N. Default is 1")
parser.add_argument("-s", "--shield", dest="shield_video", default='effects/shield.mp4',
                    help="PATH of the video FILE.", metavar="FILE")
parser.add_argument("-o", "--output", dest="output_mode", default='both',
                    choices=['window', 'virtual', 'both'],
                    help="Output mode: 'window', 'virtual', or 'both'. Default is 'both'")
args = parser.parse_args()
# -------------------------------------------------- #

# Global variables
cap = None
cam = None
show_window = False

def signal_handler(sig, frame):
    """Handle Ctrl+C clean exit"""
    print("\n\n" + "="*60)
    print("\n🛑 Interruption received (Ctrl+C)")
    print("🧹 Cleaning up resources...")

    if cap:
        cap.release()
        print("  ✅ Camera released")

    if show_window:
        cv2.destroyAllWindows()
        print("  ✅ OpenCV windows closed")

    if cam:
        cam.close()
        print("  ✅ Virtual camera closed")

    print("\n🏁 Application exited successfully\n")
    print("="*60)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

current_directory = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/")

# --- start camera ---
cap = cv2.VideoCapture(args.camera)
time.sleep(2)

# --- load SVM model ---
model = pickle.load(open(current_directory + '/' + args.ML_model, 'rb'))
labels = np.array(model.classes_)

KEY_1 = False
KEY_2 = False
KEY_3 = False
SHIELDS = False
scale = 1.5

# --- Hand Detector ---
detector = HandDetector(maxHands=2, detectionCon=0.7, trackingCon=0.6)

# --- load shield video ---
shield = cv2.VideoCapture(current_directory + '/' + args.shield_video)

# --- get camera dimensions ---
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

black_screen = np.array([0,0,0])

# Output mode
show_window = args.output_mode in ['window', 'both']
use_virtual_cam = args.output_mode in ['virtual', 'both']

print("\n" + "="*60)
print("🛡️  DR. STRANGE SHIELDS - GESTURE CONTROL SYSTEM 🛡️")
print("="*60)
print(f"\n📹 Camera ID: {args.camera}")
print(f"🤖 ML Model: {args.ML_model}")
print(f"🎬 Shield Video: {args.shield_video}\n")
print("-" * 60)
print(f"\n📺 Output Mode: {args.output_mode.upper()}\n")
if show_window: print("  ✅ OpenCV Window: ENABLED")
else: print("  ❌ OpenCV Window: DISABLED")
if use_virtual_cam: print("  ✅ Virtual Camera: ENABLED\n")
else: print("  ❌ Virtual Camera: DISABLED\n")
print("-" * 60)

# --- Main loop ---
try:
    # Initialize virtual camera if needed
    if use_virtual_cam:
        cam = pyvirtualcam.Camera(width, height, 30, fmt=PixelFormat.BGR)
        print(f"\n🎥 Virtual Camera Device: {cam.device}")
        print("🚀 System Ready! Starting gesture detection...")
        print("📋 Gesture Sequence: KEY_1 → KEY_2 → KEY_3 (activate shields)")
        print("📋 Shield Deactivation: KEY_4")
        print("⌨️  Press 'q' to quit" + (" (OpenCV window)" if show_window else "") + "\n")
        print("="*60 + "\n")
    else:
        cam = None
        print("🚀 System Ready! Starting gesture detection...")

    t1 = t2 = t3 = None  # Initialize time variables

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        # Detect hands
        hands, frame = detector.findHands(frame, flipType=False)

        # Status display
        status_shields = "🛡️ ON " if SHIELDS else "🛡️ OFF"
        status_k1 = "🔑1✅" if KEY_1 else "🔑1❌"
        status_k2 = "🔑2✅" if KEY_2 else "🔑2❌"
        status_k3 = "🔑3✅" if KEY_3 else "🔑3❌"
        print(f"\r{status_shields} | {status_k1} {status_k2} {status_k3}", end="", flush=True)

        # Read shield video frame
        ret_shield, frame_shield = shield.read()
        if not ret_shield:
            shield = cv2.VideoCapture(current_directory + '/' + args.shield_video)
            ret_shield, frame_shield = shield.read()

        # Create mask
        mask = cv2.inRange(frame_shield, black_screen, black_screen)
        res = cv2.bitwise_and(frame_shield, frame_shield, mask=mask)
        res = frame_shield - res
        alpha = 1

        # Overlay shields if hands detected
        if SHIELDS and hands:
            for hand in hands:
                x, y, w, h = hand['bbox']
                xc = x + w // 2
                yc = y + h // 2
                width_shield = int(w * 3.5 * scale)
                height_shield = int(h * 2 * scale)

                res2 = cv2.resize(res, (width_shield*2, height_shield*2))
                f_start_h = max(0, yc - height_shield)
                f_stop_h = min(height, yc + height_shield)
                f_start_w = max(0, xc - width_shield)
                f_stop_w = min(width, xc + width_shield)
                res2 = res2[0:f_stop_h-f_start_h, 0:f_stop_w-f_start_w, :]
                frame[f_start_h:f_stop_h, f_start_w:f_stop_w] = cv2.addWeighted(
                    frame[f_start_h:f_stop_h, f_start_w:f_stop_w], alpha, res2, 1, 1
                )

        # Gesture recognition using your SVM model
        if hands and len(hands) == 2:
            # Prepare data for prediction: landmarks normalized
            hand_points = []
            for hand in hands:
                hand_points += hand['lmList'][0:21]  # 21 points per hand

            hand_points = np.array(hand_points).flatten().reshape(1, -1)
            prediction = model.predict(hand_points)[0]
            pred_prob = np.max(model.predict_proba(hand_points))

            # Activate shields sequence
            if prediction == 'key_1' and pred_prob > 0.85:
                t1 = datetime.now()
                KEY_1 = True
            elif prediction == 'key_2' and pred_prob > 0.85 and KEY_1:
                t2 = datetime.now()
                if t1 + timedelta(seconds=2) > t2:
                    KEY_2 = True
                else:
                    KEY_1 = KEY_2 = False
            elif prediction == 'key_3' and pred_prob > 0.85 and KEY_1 and KEY_2:
                t3 = datetime.now()
                if t2 + timedelta(seconds=2) > t3:
                    KEY_3 = True
                    SHIELDS = True
                else:
                    KEY_1 = KEY_2 = False
            elif prediction == 'key_4' and pred_prob > 0.85:
                KEY_1 = KEY_2 = KEY_3 = SHIELDS = False

        # Display window if enabled
        if show_window:
            cv2.imshow('Dr. Strange shields', frame)

        # Virtual camera output
        if use_virtual_cam and cam:
            cam.send(frame)
            cam.sleep_until_next_frame()

        # Quit handling
        if show_window and cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif not show_window:
            time.sleep(0.033)  # ~30 FPS

except KeyboardInterrupt:
    print("\n🛑 Interrupted - cleaning up...")

finally:
    print("\n🧹 Final cleanup...")
    if cap: cap.release()
    if show_window: cv2.destroyAllWindows()
    if cam: cam.close()
    print("🏁 Application terminated\n")
