
import io
import time
import wave
import threading

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    VideoProcessorBase,
)

# ==========================================================
# GESTURE GUITAR AI
# Webcam hand gestures + guitar-like chord audio
# ==========================================================

st.set_page_config(
    page_title="Gesture Guitar AI",
    page_icon="🎸",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        color: #7C5CFF;
    }
    .subtitle {
        text-align: center;
        font-size: 17px;
        color: #888888;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">🎸 GESTURE GUITAR AI</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">'
    'Play guitar-like chords using your hand gestures'
    '</div>',
    unsafe_allow_html=True,
)

SAMPLE_RATE = 44100

# Map the number of extended fingers to a chord.
GESTURE_CHORDS = {
    0: "Em",  # Closed fist
    1: "C",   # One finger
    2: "D",   # Two fingers
    3: "G",   # Three fingers
    4: "A",   # Four fingers
    5: "E",   # Open hand
}

CHORD_NOTES = {
    "C":  [130.81, 164.81, 196.00, 261.63, 329.63],
    "D":  [146.83, 220.00, 293.66, 369.99],
    "E":  [82.41, 123.47, 164.81, 207.65, 246.94],
    "G":  [98.00, 123.47, 146.83, 196.00, 246.94],
    "A":  [110.00, 164.81, 220.00, 277.18, 329.63],
    "Em": [82.41, 123.47, 164.81, 196.00, 246.94],
}


def make_guitar_audio(chord, duration=1.8):
    """Synthesize a plucked-string-like chord as WAV audio."""
    notes = CHORD_NOTES[chord]
    t = np.arange(int(SAMPLE_RATE * duration)) / SAMPLE_RATE
    audio = np.zeros_like(t, dtype=np.float64)

    for index, frequency in enumerate(notes):
        string = np.zeros_like(t)

        # Fundamental plus harmonics for a string-like timbre.
        for harmonic in range(1, 5):
            string += (
                np.sin(2 * np.pi * frequency * harmonic * t)
                / harmonic
            )

        # Small variations between strings.
        delay = index * 0.018
        shifted_t = np.maximum(t - delay, 0)
        envelope = np.exp(-2.4 * shifted_t)
        envelope[t < delay] = 0

        audio += string * envelope / len(notes)

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.85

    samples = (audio * 32767).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(samples.tobytes())

    return buffer.getvalue()


def count_fingers(hand_landmarks, handedness):
    """Estimate how many fingers are extended."""
    lm = hand_landmarks.landmark
    count = 0

    # Thumb: compare horizontal positions, accounting for hand side.
    if handedness == "Right":
        if lm[4].x < lm[3].x:
            count += 1
    else:
        if lm[4].x > lm[3].x:
            count += 1

    # Index, middle, ring and little fingers.
    finger_pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]

    for tip, pip in finger_pairs:
        if lm[tip].y < lm[pip].y:
            count += 1

    return count


class GuitarProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.detected_chord = None
        self.finger_count = None

        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.60,
        )

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        chord = None
        fingers = None

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]

            handedness = "Right"
            if results.multi_handedness:
                handedness = (
                    results.multi_handedness[0]
                    .classification[0].label
                )

            fingers = count_fingers(hand, handedness)
            chord = GESTURE_CHORDS[fingers]

            mp.solutions.drawing_utils.draw_landmarks(
                image,
                hand,
                mp.solutions.hands.HAND_CONNECTIONS,
            )

            cv2.putText(
                image,
                f"{chord} chord | {fingers} fingers",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (80, 255, 130),
                2,
            )
        else:
            cv2.putText(
                image,
                "Show one hand to the camera",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        with self.lock:
            self.detected_chord = chord
            self.finger_count = fingers

        return av.VideoFrame.from_ndarray(
            image, format="bgr24"
        )


st.markdown("### 🎵 Gesture guide")

st.markdown(
    """
    | Hand gesture | Chord |
    |---|---|
    | ✊ Closed fist | Em |
    | ☝️ One finger | C |
    | Two fingers | D |
    | Three fingers | G |
    | Four fingers | A |
    | ✋ Open hand | E |
    """
)

st.info(
    "Start the webcam, allow camera access, and show one hand. "
    "The number of extended fingers selects the chord."
)

# The browser may require a user interaction before audio can play.
if "sound_enabled" not in st.session_state:
    st.session_state.sound_enabled = False

if "last_played_chord" not in st.session_state:
    st.session_state.last_played_chord = None

if st.button("🔊 Enable sound", type="primary"):
    st.session_state.sound_enabled = True
    st.success("Sound enabled. Use the Play chord button to test audio.")

ctx = webrtc_streamer(
    key="gesture-guitar",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=GuitarProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    async_processing=True,
)

st.markdown("### 🎸 Detected chord")

@st.fragment(run_every="500ms")
def show_chord():
    if not ctx.state.playing or ctx.video_processor is None:
        st.write("Start the webcam to detect a chord.")
        return

    processor = ctx.video_processor

    with processor.lock:
        chord = processor.detected_chord
        fingers = processor.finger_count

    if chord is None:
        st.write("No hand detected. Hold your hand clearly in view.")
        return

    st.subheader(f"Current chord: {chord}")
    st.write(f"Extended fingers detected: {fingers}")

    if st.session_state.sound_enabled:
        if chord != st.session_state.last_played_chord:
            audio_data = make_guitar_audio(chord)
            st.audio(audio_data, format="audio/wav", autoplay=True)
            st.session_state.last_played_chord = chord
    else:
        st.caption("Click Enable sound to activate audio playback.")

show_chord()

st.markdown("---")
st.markdown("### 🎹 Test audio manually")

manual_chord = st.selectbox(
    "Choose a chord to test",
    list(CHORD_NOTES.keys()),
)

if st.button("▶️ Play chord"):
    st.audio(
        make_guitar_audio(manual_chord),
        format="audio/wav",
    )

st.caption(
    "Audio is synthesized in Python and resembles a plucked string; "
    "it is not a recording of a real guitar. Gesture recognition "
    "selects chords by finger count, so it is a simple demonstration."
)
