# app.py - FocusCam Flask Web Version
# Made by Godfred Bio (transformed to Flask with love)

import cv2
import mediapipe as mp
import math
import time
import random
from datetime import datetime
import csv
import os
import json
import pyttsx3
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify, send_file

# --- Config ---
SNAPSHOT_FOLDER = "snapshots"
QUOTE_FILE = "data/quotes.json"
LOG_FILE = "logs/focuscam_session_log.csv"
SETTINGS_FILE = "data/settings.json"

os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)

app = Flask(__name__)

# --- Global session state ---
session_active = False
paused = False
start_time = None
last_time = None
focused_seconds = 0.0
distracted_seconds = 0.0
last_alert = None
distraction_count = 0
duration = 0  # in seconds
username = ""
goal = ""
cap = None
face_mesh = None

# --- Utils ---
def play_alert_sound():
    try:
        import winsound
        winsound.Beep(1000, 500)
    except:
        pass

def speak(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except:
        pass

def get_quote(focus_percent):
    quotes = load_quotes()
    if focus_percent >= 90:
        return random.choice(quotes["motivational"])
    elif focus_percent >= 70:
        return "Good job! You can do even better!"
    else:
        return random.choice(quotes["punishing"])

def calculate_head_pitch(nose, chin):
    return math.degrees(math.atan2(chin.y - nose.y, 0.1))

def estimate_gaze(left_eye, right_eye):
    gaze = (left_eye.y + right_eye.y) / 2
    return gaze < 0.5

def save_snapshot(frame):
    filename = os.path.join(SNAPSHOT_FOLDER, f"distraction_{int(time.time())}.jpg")
    cv2.imwrite(filename, frame)

# --- Settings & Quotes ---
def load_settings():
    default = {"duration": 30, "goal": "", "username": "User"}
    if not os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(default, f)
    with open(SETTINGS_FILE) as f:
        return json.load(f)

def save_settings(s):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(s, f, indent=4)

def load_quotes():
    default_quotes = {
        "motivational": ["Stay focused.", "You're investing in your future."],
        "punishing": ["Discipline matters.", "Each distraction counts."]
    }
    if not os.path.exists(QUOTE_FILE):
        with open(QUOTE_FILE, 'w') as f:
            json.dump(default_quotes, f)
    with open(QUOTE_FILE) as f:
        return json.load(f)

def save_quotes(quotes):
    with open(QUOTE_FILE, 'w') as f:
        json.dump(quotes, f, indent=4)

# --- Session Management ---
def end_session():
    global session_active, cap, focused_seconds, distracted_seconds
    session_active = False

    total_seconds = focused_seconds + distracted_seconds
    focus_percent = int((focused_seconds / total_seconds) * 100) if total_seconds > 0 else 0
    focus_min = round(focused_seconds / 60, 2)
    distract_min = round(distracted_seconds / 60, 2)

    # Save to log
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if os.stat(LOG_FILE).st_size == 0:
            writer.writerow(["Date", "Username", "Focused (min)", "Distracted (min)"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), username, focus_min, distract_min])

    if cap:
        cap.release()

# --- Video Frame Generator ---
def gen_frames():
    global paused, start_time, last_time, focused_seconds, distracted_seconds, \
           last_alert, distraction_count, session_active

    while session_active:
        now = time.time()

        # Check if session time is over
        if now - start_time >= duration:
            end_session()
            break

        ret, frame = cap.read()
        if not ret:
            break

        if paused:
            cv2.putText(frame, "PAUSED", (frame.shape[1]//2 - 100, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            is_focused = False
            if results.multi_face_landmarks:
                face = results.multi_face_landmarks[0]
                nose = face.landmark[1]
                chin = face.landmark[152]
                left_eye = face.landmark[33]
                right_eye = face.landmark[263]

                pitch = calculate_head_pitch(nose, chin)
                gaze = estimate_gaze(left_eye, right_eye)
                is_focused = (-10 <= pitch <= 70) and gaze

            elapsed = now - last_time
            last_time = now

            if is_focused:
                focused_seconds += elapsed
            else:
                distracted_seconds += elapsed
                if now - last_alert > 5:
                    distraction_count += 1
                    play_alert_sound()
                    speak(f"{username}, please focus.")
                    save_snapshot(frame)
                    if distraction_count == 5 and goal:
                        speak(f"Remember your goal: {goal}")
                    elif distraction_count == 10:
                        speak(random.choice(load_quotes()["punishing"]))
                    last_alert = now

        # Overlay stats
        total_seconds = focused_seconds + distracted_seconds
        focus_percent = int((focused_seconds / total_seconds) * 100) if total_seconds > 0 else 0
        remaining = max(0, duration - (now - start_time))
        mins = int(remaining) // 60
        secs = int(remaining) % 60

        color = (0, 255, 0) if not paused else (0, 0, 255)
        cv2.putText(frame, f"{username} | Focus: {focus_percent}%",
                    (frame.shape[1] - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Time Left: {mins:02}:{secs:02}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- Routes ---
@app.route('/')
def index():
    settings = load_settings()
    return render_template('index.html', settings=settings)
@app.route('/stop_session', methods=['POST'])
def stop_session():
    global session_active
    if session_active:
        end_session()  # This already calculates and saves results
    return jsonify({'stopped': True})

@app.route('/start', methods=['POST'])
def start():
    global session_active, paused, start_time, last_time, focused_seconds, distracted_seconds, \
           last_alert, distraction_count, duration, username, goal, cap, face_mesh

    username = request.form.get('username', 'User').strip()
    try:
        duration_min = int(request.form.get('duration', 30))
    except:
        duration_min = 30
    goal = request.form.get('goal', '').strip()

    duration = duration_min * 60

    save_settings({"username": username, "duration": duration_min, "goal": goal})

    # Reset and start session
    session_active = True
    paused = False
    start_time = time.time()
    last_time = time.time()
    focused_seconds = 0.0
    distracted_seconds = 0.0
    last_alert = time.time()
    distraction_count = 0

    cap = cv2.VideoCapture(0)
    face_mesh = mp.solutions.face_mesh.FaceMesh()

    return redirect(url_for('session'))

@app.route('/session')
def session():
    if not session_active:
        return redirect(url_for('index'))
    return render_template('session.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_pause', methods=['POST'])
def toggle_pause():
    global paused
    paused = not paused
    return jsonify({'paused': paused})

@app.route('/status')
def status():
    if not session_active:
        total_seconds = focused_seconds + distracted_seconds
        focus_percent = int((focused_seconds / total_seconds) * 100) if total_seconds > 0 else 0
        focus_min = round(focused_seconds / 60, 2)
        distract_min = round(distracted_seconds / 60, 2)
        final_quote = get_quote(focus_percent)
        goal_display = f"\nGoal: {goal}" if goal else ""

        message = (f"Session Complete!\n\n"
                   f"Focus: {focus_min} min\n"
                   f"Distracted: {distract_min} min\n"
                   f"Focus %: {focus_percent}%\n\n"
                   f"{final_quote}{goal_display}")

        return jsonify({'ended': True, 'message': message})
    return jsonify({'ended': False})

@app.route('/edit_quotes')
def edit_quotes():
    category = request.args.get('category', 'motivational')
    quotes = load_quotes()
    return render_template('edit_quotes.html', quotes=quotes[category], category=category, all_quotes=quotes)

@app.route('/add_quote', methods=['POST'])
def add_quote():
    category = request.form['category']
    quote = request.form['quote'].strip()
    if quote:
        quotes = load_quotes()
        quotes[category].append(quote)
        save_quotes(quotes)
    return redirect(url_for('edit_quotes', category=category))

@app.route('/delete_quote', methods=['POST'])
def delete_quote():
    category = request.form['category']
    index = int(request.form['index'])
    quotes = load_quotes()
    if 0 <= index < len(quotes[category]):
        quotes[category].pop(index)
        save_quotes(quotes)
    return redirect(url_for('edit_quotes', category=category))

@app.route('/export_csv')
def export_csv():
    if os.path.exists(LOG_FILE):
        return send_file(LOG_FILE, as_attachment=True, download_name='focuscam_session_log.csv')
    return "No log file yet.", 404

if __name__ == '__main__':
    app.run(debug=True, threaded=True)