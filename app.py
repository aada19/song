import streamlit as st
import cv2
import mediapipe as mp
import av
import time
import os
import wave
import numpy as np

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


# ============================================================
# 🎸 GESTURE GUITAR AI
# Webcam + MediaPipe + Streamlit WebRTC + Guitar Audio
# ============================================================

st.set_page_config(
    page_title="Gesture Guitar AI",
    page_icon="🎸",
    layout="wide"
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 48px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 25px;
}

.chord-box {
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    border: 2px solid #555;
    margin-top: 10px;
}

.chord-name {
    font-size: 60px;
    font-weight: bold;
}

.info-box {
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #555;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🎸 GESTURE GUITAR AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Play guitar chords using your hand gestures'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# CREATE GUITAR-LIKE CHORD AUDIO
# ============================================================

AUDIO_FOLDER = "guitar_chords"

os.makedirs(AUDIO_FOLDER, exist_ok=True)


# Guitar chord frequencies
CHORDS = {

    "C": [
        130.81,   # C3
        164.81,   # E3
        196.00,   # G3
        261.63,   # C4
        329.63    # E4
    ],

    "D": [
        146.83,   # D3
        220.00,   # A3
        293.66,   # D4
        369.99,   # F#4
        440.00    # A4
    ],

    "E": [
        82.41,    # E2
        123.47,   # B2
        164.81,   # E3
        207.65,   # G#3
        246.94,   # B3
        329.63    # E4
    ],

    "G": [
        98.00,    # G2
        146.83,   # D3
        196.00,   # G3
        246.94,   # B3
        293.66,   # D4
        392.00    # G4
    ],

    "A": [
        110.00,   # A2
        164.81,   # E3
        220.00,   # A3
        277.18,   # C#4
        329.63,   # E4
        440.00    # A4
    ],

    "Em": [
        82.41,    # E2
        123.47,   # B2
        164.81,   # E3
        196.00,   # G3
        246.94,   # B3
        329.63    # E4
    ]
}


def create_guitar_chord(frequencies, filename):

    """
    Creates a short guitar-like plucked chord.
    """

    sample_rate = 44100
    duration = 1.6

    total_samples = int(sample_rate * duration)

    audio = np.zeros(total_samples)

    # Slight delay between strings to imitate strumming
    delays = np.linspace(
        0,
        0.12,
        len(frequencies)
    )

    for frequency, delay in zip(frequencies, delays):

        start = int(delay * sample_rate)

        t = np.arange(
            total_samples - start
        ) / sample_rate

        # Fundamental + harmonics
        signal = (
            1.00 * np.sin(2 * np.pi * frequency * t)
            + 0.35 * np.sin(2 * np.pi * frequency * 2 * t)
            + 0.18 * np.sin(2 * np.pi * frequency * 3 * t)
            + 0.08 * np.sin(2 * np.pi * frequency * 4 * t)
        )

        # Guitar-like attack and decay
        attack_time = 0.015
        attack_samples = int(
            attack_time * sample_rate
        )

        envelope = np.exp(
            -2.8 * np.arange(len(t)) / sample_rate
        )

        attack = np.ones(len(t))

        if attack_samples > 0:
            attack[:attack_samples] = np.linspace(
                0,
                1,
                attack_samples
            )

        signal *= envelope
        signal *= attack

        audio[start:] += signal

    # Normalize
    maximum = np.max(np.abs(audio))

    if maximum > 0:
        audio = audio / maximum

    # Slight volume reduction
    audio *= 0.65

    # Convert to 16-bit PCM
    audio_int16 = np.int16(
        audio * 32767
    )

    # Save WAV
    with wave.open(filename, "wb") as wav:

        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)

        wav.writeframes(
            audio_int16.tobytes()
        )


# ============================================================
# GENERATE CHORD FILES
# ============================================================

for chord_name, frequencies in CHORDS.items():

    chord_file = os.path.join(
        AUDIO_FOLDER,
        chord_name + ".wav"
    )

    if not os.path.exists(chord_file):

        create_guitar_chord(
            frequencies,
            chord_file
        )


# ============================================================
# FINGER DETECTION
# ============================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def count_fingers(hand_landmarks, handedness):

    landmarks = hand_landmarks.landmark

    fingers = []

    # --------------------------------------------------------
    # THUMB
    # --------------------------------------------------------

    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]

    if handedness == "Right":

        if thumb_tip.x < thumb_ip.x:
            fingers.append(1)
        else:
            fingers.append(0)

    else:

        if thumb_tip.x > thumb_ip.x:
            fingers.append(1)
        else:
            fingers.append(0)

    # --------------------------------------------------------
    # INDEX
    # --------------------------------------------------------

    fingers.append(
        1 if landmarks[8].y < landmarks[6].y else 0
    )

    # --------------------------------------------------------
    # MIDDLE
    # --------------------------------------------------------

    fingers.append(
        1 if landmarks[12].y < landmarks[10].y else 0
    )

    # --------------------------------------------------------
    # RING
    # --------------------------------------------------------

    fingers.append(
        1 if landmarks[16].y < landmarks[14].y else 0
    )

    # --------------------------------------------------------
    # LITTLE
    # --------------------------------------------------------

    fingers.append(
        1 if landmarks[20].y < landmarks[18].y else 0
    )

    return sum(fingers)


# ============================================================
# GESTURE → CHORD
# ============================================================

def get_chord(finger_count):

    if finger_count == 0:
        return "Em", "✊ Fist"

    elif finger_count == 1:
        return "C", "☝️ One Finger"

    elif finger_count == 2:
        return "D", "✌️ Two Fingers"

    elif finger_count == 3:
        return "E", "🤟 Three Fingers"

    elif finger_count == 4:
        return "G", "🖐️ Four Fingers"

    elif finger_count == 5:
        return "A", "✋ Five Fingers"

    return "Waiting...", "No gesture"


# ============================================================
# VIDEO PROCESSOR
# ============================================================

class GestureProcessor(VideoProcessorBase):

    def __init__(self):

        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.chord = "Waiting..."
        self.gesture = "No hand detected"

        self.last_chord = "Waiting..."

    def recv(self, frame):

        image = frame.to_ndarray(
            format="bgr24"
        )

        # Mirror webcam
        image = cv2.flip(image, 1)

        # BGR → RGB
        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        results = self.hands.process(
            rgb_image
        )

        detected_chord = "Waiting..."
        detected_gesture = "No hand detected"

        # ----------------------------------------------------
        # HAND DETECTED
        # ----------------------------------------------------

        if results.multi_hand_landmarks:

            hand_landmarks = (
                results.multi_hand_landmarks[0]
            )

            handedness = (
                results.multi_handedness[0]
                .classification[0]
                .label
            )

            finger_count = count_fingers(
                hand_landmarks,
                handedness
            )

            detected_chord, detected_gesture = (
                get_chord(finger_count)
            )

            # Draw hand
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Camera text
            cv2.putText(
                image,
                detected_gesture,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            cv2.putText(
                image,
                "CHORD: " + detected_chord,
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 150),
                3
            )

        else:

            cv2.putText(
                image,
                "Show your hand",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

        # Store result
        self.chord = detected_chord
        self.gesture = detected_gesture

        # Return frame
        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )


# ============================================================
# SESSION STATE
# ============================================================

if "last_played_chord" not in st.session_state:

    st.session_state.last_played_chord = ""


# ============================================================
# LAYOUT
# ============================================================

left, right = st.columns(
    [2, 1]
)


# ============================================================
# WEBCAM
# ============================================================

with left:

    st.subheader("📷 Webcam")

    ctx = webrtc_streamer(

        key="gesture-guitar",

        video_processor_factory=GestureProcessor,

        media_stream_constraints={
            "video": True,
            "audio": False
        },

        async_processing=True,

        rtc_configuration={
            "iceServers": [
                {
                    "urls": [
                        "stun:stun.l.google.com:19302"
                    ]
                }
            ]
        }
    )


# ============================================================
# CHORD PANEL
# ============================================================

with right:

    st.subheader("🎵 Detected Chord")

    if ctx.video_processor:

        chord = ctx.video_processor.chord
        gesture = ctx.video_processor.gesture

        st.markdown(
            f"""
            <div class="chord-box">
                <div>Current Chord</div>
                <div class="chord-name">
                    {chord}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write(
            f"**Gesture:** {gesture}"
        )

        # ----------------------------------------------------
        # AUTOMATIC AUDIO
        # ----------------------------------------------------

        if (
            chord != "Waiting..."
            and chord != st.session_state.last_played_chord
        ):

            audio_file = os.path.join(
                AUDIO_FOLDER,
                chord + ".wav"
            )

            if os.path.exists(audio_file):

                st.audio(
                    audio_file,
                    format="audio/wav",
                    autoplay=True
                )

                st.session_state.last_played_chord = chord

                st.success(
                    f"🎸 Playing {chord} chord!"
                )

    else:

        st.markdown(
            """
            <div class="chord-box">
                <div>Current Chord</div>
                <div class="chord-name">🎸</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.info(
            "Click START and allow webcam access."
        )


# ============================================================
# GESTURE GUIDE
# ============================================================

st.divider()

st.subheader("🎸 Gesture → Guitar Chord")


col1, col2, col3 = st.columns(3)

with col1:

    st.markdown("### ☝️")
    st.write("**1 Finger → C Chord**")

with col2:

    st.markdown("### ✌️")
    st.write("**2 Fingers → D Chord**")

with col3:

    st.markdown("### 🤟")
    st.write("**3 Fingers → E Chord**")


col4, col5, col6 = st.columns(3)

with col4:

    st.markdown("### 🖐️")
    st.write("**4 Fingers → G Chord**")

with col5:

    st.markdown("### ✋")
    st.write("**5 Fingers → A Chord**")

with col6:

    st.markdown("### ✊")
    st.write("**Fist → Em Chord**")


# ============================================================
# HOW IT WORKS
# ============================================================

st.divider()

st.subheader("🎮 How to Play")

st.write("""
1. Click **START**.
2. Allow access to your webcam.
3. Show your hand to the camera.
4. Make one of the six gestures.
5. The AI identifies your gesture.
6. The corresponding guitar chord is detected.
7. The chord sound automatically plays. 🎸
""")


st.info(
    "💡 Change your hand gesture to another gesture "
    "to play another chord."
)
