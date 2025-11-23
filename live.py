#!/usr/bin/env python3
"""
Merged Moondream Navigator
- Background inference thread (fast UI)
- Robust Ollama client usage and flexible response parsing
- Safe defaults and graceful shutdown
"""
import cv2
import time
import threading
import signal
import argparse
import re
import sys

try:
    import ollama
except Exception:
    raise RuntimeError("Install the Ollama Python client (pip install ollama) and ensure Ollama is running.")

# ---------- Config ----------
DEFAULT_MODEL = "moondream"
RESIZE_WH = (512, 512)
JPEG_QUALITY = 70
SLEEP_BETWEEN_INFER = 0.05  # small gap when no frame available
# ----------------------------

# Allowed commands
ALLOWED = ("LEFT", "RIGHT", "FORWARD", "STOP")

def extract_direction(text):
    """Robust extraction of single token direction from model text."""
    if not text:
        return None
    # unify
    txt = str(text).upper()
    # Prefer direct token
    for token in ALLOWED:
        if re.search(rf"\b{token}\b", txt):
            return token
    # last-resort heuristics
    if "LEFT" in txt: return "LEFT"
    if "RIGHT" in txt: return "RIGHT"
    if "FORWARD" in txt: return "FORWARD"
    if "STOP" in txt: return "STOP"
    return None

class MoondreamNavigator:
    def __init__(self, model_name=DEFAULT_MODEL, camera_index=0, show=True):
        self.model_name = model_name
        self.camera_index = camera_index
        self.show = show

        self.client = ollama.Client()  # robust client usage

        # shared state
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.current_instruction = "WAITING..."
        self.last_latency = 0.0

        self.running = False
        self._stop_event = threading.Event()

        # Prompt: deterministic single-token output
        self.prompt = (
            "You are a movement planner for a small wheeled robot. View the attached image."
            " Decide exactly one action from {LEFT, RIGHT, FORWARD, STOP} and output only that single token."
            " Prefer STOP when uncertain."
        )

    def move_robot(self, direction):
        # Replace with actual motor/ROS/serial commands for real robot.
        print(f"[MOTOR] -> {direction}")

    def _ask_model(self, image_bytes):
        """
        Send image + prompt to Ollama model and return raw textual response.
        Handles different possible response shapes from versions of the SDK.
        """
        messages = [{"role": "user", "content": self.prompt, "images": [image_bytes]}]
        try:
            # prefer client.chat (object-oriented)
            resp = self.client.chat(model=self.model_name, messages=messages, stream=False)
        except TypeError:
            # fallback if signature differs
            resp = ollama.chat(model=self.model_name, messages=messages)
        except Exception as e:
            raise

        # Try to robustly extract text from resp
        # resp may be str, dict, or an object with nested fields.
        text = None
        if isinstance(resp, str):
            text = resp
        elif isinstance(resp, dict):
            # common shapes
            # 1) {'message': {'content': '...'}}
            if 'message' in resp and isinstance(resp['message'], dict):
                text = resp['message'].get('content') or resp['message'].get('text')
            # 2) {'response': '...'}
            if not text:
                text = resp.get('response') or resp.get('text')
            # 3) sometimes nested choices
            if not text and 'choices' in resp and isinstance(resp['choices'], list):
                # try first choice
                c = resp['choices'][0]
                if isinstance(c, dict):
                    text = c.get('message', {}).get('content') or c.get('text') or c.get('delta', {}).get('content')
        else:
            # fallback to str()
            text = str(resp)

        if text is None:
            text = str(resp)
        return text

    def inference_loop(self):
        """Background thread: read latest frame, send to model, update instruction."""
        print("Inference thread started.")
        while not self._stop_event.is_set():
            frame = None
            with self.frame_lock:
                if self.current_frame is not None:
                    frame = self.current_frame.copy()
            if frame is None:
                time.sleep(SLEEP_BETWEEN_INFER)
                continue

            try:
                start = time.time()
                resized = cv2.resize(frame, RESIZE_WH, interpolation=cv2.INTER_AREA)
                ret, buf = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                if not ret:
                    print("Warning: failed to encode frame to JPEG.")
                    continue
                image_bytes = buf.tobytes()

                raw = self._ask_model(image_bytes)
                latency = time.time() - start
                self.last_latency = latency

                direction = extract_direction(raw)
                if direction is None:
                    # safety fallback
                    direction = "STOP"

                # update shared state
                self.current_instruction = direction
                self.move_robot(direction)

            except Exception as e:
                print(f"Inference error: {e}")
                self.current_instruction = "ERROR"
                time.sleep(0.5)

        print("Inference thread stopping.")

    def run(self):
        # handle signals for graceful shutdown
        def _sigterm(sig, frame):
            print("Signal received, shutting down...")
            self._stop_event.set()
        signal.signal(signal.SIGINT, _sigterm)
        signal.signal(signal.SIGTERM, _sigterm)

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print("Error: cannot open camera.")
            return

        self.running = True
        t = threading.Thread(target=self.inference_loop, daemon=True)
        t.start()

        print("Camera started. Press 'q' in the window to quit.")
        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    print("Warning: failed to read frame.")
                    time.sleep(0.1)
                    continue

                # publish frame to inference thread
                with self.frame_lock:
                    self.current_frame = frame

                # visualization overlay
                if self.show:
                    disp = frame.copy()
                    h, w = disp.shape[:2]

                    # translucent box
                    overlay = disp.copy()
                    cv2.rectangle(overlay, (10, 10), (380, 110), (0,0,0), -1)
                    cv2.addWeighted(overlay, 0.5, disp, 0.5, 0, disp)

                    # status color
                    col = (0,255,0)  # forward green
                    if "STOP" in self.current_instruction: col = (0,0,255)
                    elif "LEFT" in self.current_instruction or "RIGHT" in self.current_instruction: col = (0,255,255)

                    cv2.putText(disp, f"CMD: {self.current_instruction}", (30, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2, cv2.LINE_AA)
                    cv2.putText(disp, f"Latency: {self.last_latency:.2f}s", (30, 95),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)

                    cv2.imshow("Moondream Navigator", disp)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self._stop_event.set()
                        break
                else:
                    # small sleep to prevent tight loop if no UI
                    time.sleep(0.01)

        finally:
            self._stop_event.set()
            t.join(timeout=2.0)
            cap.release()
            if self.show:
                cv2.destroyAllWindows()
            print("Shutdown complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--no-show", dest="show", action="store_false", help="disable preview window")
    args = parser.parse_args()

    bot = MoondreamNavigator(model_name=args.model, camera_index=args.camera, show=args.show)
    bot.run()
