
import threading
import numpy as np
import cv2
import mediapipe as mp
import av
import streamlit as st

from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    VideoProcessorBase,
    AudioProcessorBase,
)

# =========================================================
# GESTURE GUITAR AI
# Hand gestures control continuously playing guitar chords
# =========================================================

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
        color: #8055ff;
        font-size: 42px;
        font-weight: 800;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 17px;
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
    '<div class="subtitle">Show a hand gesture and play guitar chords!</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

SAMPLE_RATE = 48000
STRUM_LENGTH = 2.0

GESTURE_CHORDS = {
    0: "Em",  # Fist
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


# ---------------------------------------------------------
# SHARED STATE: connects webcam detection to audio playback
# ---------------------------------------------------------

class GuitarState:
    def __init__(self):
        self.lock = threading.Lock()
        self.chord = None
        self.fingers = None
        self.candidate = None
        self.candidate_frames = 0

    def update(self, chord, fingers):
        with self.lock:
            self.fingers = fingers

            if chord is None:
                self.candidate = None
                self.candidate_frames = 0
                self.chord = None
                return

            if chord == self.candidate:
                self.candidate_frames += 1
            else:
                self.candidate = chord
                self.candidate_frames = 1

            # Avoid switching chords because of one bad camera frame.
            if self.candidate_frames >= 3:
                self.chord = chord


guitar_state = GuitarState()


# ---------------------------------------------------------
# HAND GESTURE DETECTION
# ---------------------------------------------------------

def count_fingers(hand, handedness):
    lm = hand.landmark
    count = 0

    # Thumb
    if handedness == "Right":
        if lm[4].x < lm[3].x:
            count += 1
    else:
        if lm[4].x > lm[3].x:
            count += 1

    # Index, middle, ring and little fingers
    for tip, joint in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        if lm[tip].y < lm[joint].y:
            count += 1

    return count


class GuitarVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.60,
        )

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        chord = None
        fingers = None

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]

            handedness = "Right"
            if result.multi_handedness:
                handedness = (
                    result.multi_handedness[0]
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
                f"CHORD: {chord} | FINGERS: {fingers}",
                (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (80, 255, 130),
                2,
            )
        else:
            cv2.putText(
                image,
                "Show one hand to play",
                (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

        guitar_state.update(chord, fingers)

        return av.VideoFrame.from_ndarray(image, format="bgr24")


# ---------------------------------------------------------
# CONTINUOUS GUITAR AUDIO
# The browser microphone provides an audio stream, but its
# captured sound is replaced by synthesized guitar audio.
# ---------------------------------------------------------

class GuitarAudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.last_chord = None
        self.sample_position = 0
        self.lock = threading.Lock()

    def make_strum(self, chord, sample_count, sample_rate):
        """Create a repeating plucked-string-like chord."""
        if chord not in CHORD_NOTES:
            return np.zeros(sample_count, dtype=np.float32)

        # Repeat the strum while the same gesture is held.
        positions = (
            self.sample_position + np.arange(sample_count)
        ) / sample_rate
        local_time = positions % STRUM_LENGTH

        audio = np.zeros(sample_count, dtype=np.float64)
        notes = CHORD_NOTES[chord]

        for index, frequency in enumerate(notes):
            delay = index * 0.025
            note_time = local_time - delay
            active = note_time >= 0
            safe_time = np.maximum(note_time, 0)

            # Fundamental plus softer harmonics.
            string = (
                np.sin(2 * np.pi * frequency * safe_time)
                + 0.40 * np.sin(
                    2 * np.pi * frequency * 2 * safe_time
                )
                + 0.16 * np.sin(
                    2 * np.pi * frequency * 3 * safe_time
                )
            )

            envelope = np.exp(-2.2 * safe_time)
            envelope *= active
            audio += string * envelope / len(notes)

        # Fade the loop boundary to reduce audible clicks.
        fade_length = min(int(sample_rate * 0.015), sample_count)
        if fade_length > 0:
            fade = np.linspace(0, 1, fade_length)
            boundary = (local_time < 0.015) | (
                local_time > STRUM_LENGTH - 0.015
            )
            audio[local_time < 0.015] *= fade[
                :np.count_nonzero(local_time < 0.015)
            ] if np.count_nonzero(local_time < 0.015) <= fade_length else 1

        peak = np.max(np.abs(audio)) if audio.size else 0
        if peak > 0:
            audio = audio / max(peak, 1.0) * 0.65

        return audio.astype(np.float32)

    def recv(self, frame):
        sample_rate = frame.sample_rate or SAMPLE_RATE
        sample_count = frame.samples
        fmt = frame.format.name
        layout = frame.layout.name
        channels = len(frame.layout.channels)

        with guitar_state.lock:
            chord = guitar_state.chord

        with self.lock:
            if chord != self.last_chord:
                self.last_chord = chord
                self.sample_position = 0

            if chord is None:
                mono = np.zeros(sample_count, dtype=np.float32)
            else:
                mono = self.make_strum(
                    chord, sample_count, sample_rate
                )

            self.sample_position += sample_count

        # Match the browser's incoming audio format and layout.
        if fmt.startswith("s16"):
            converted = np.clip(mono * 32767, -32768, 32767).astype(
                np.int16
            )
        elif fmt.startswith("s32"):
            converted = np.clip(
                mono * 2147483647, -2147483648, 2147483647
            ).astype(np.int32)
        elif fmt.startswith("u8"):
            converted = np.clip(
                mono * 127 + 128, 0, 255
            ).astype(np.uint8)
        else:
            converted = mono.astype(np.float32)

        if frame.format.is_planar:
            output = np.tile(converted, (channels, 1))
        else:
            output = np.repeat(converted, channels).reshape(1, -1)

        new_frame = av.AudioFrame.from_ndarray(
            output, format=fmt, layout=layout
        )
        new_frame.sample_rate = sample_rate
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base

        return new_frame


# ---------------------------------------------------------
# USER INTERFACE
# ---------------------------------------------------------

st.markdown("### 🎵 How to play")

st.markdown(
    """
    | Hand gesture | Chord |
    |---|---|
    | ✊ Closed fist | Em |
    | ☝️ One finger | C |
    | ✌️ Two fingers | D |
    | Three fingers | G |
    | Four fingers | A |
    | ✋ Open hand | E |
    """
)

st.info(
    "Click START GUITAR and allow both camera and microphone "
    "permissions. Hold your hand in view to play. Change your "
    "gesture to switch chords; hide your hand to silence the guitar."
)

ctx = webrtc_streamer(
    key="gesture-guitar-live",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=GuitarVideoProcessor,
    audio_processor_factory=GuitarAudioProcessor,
    media_stream_constraints={
        "video": True,
        "audio": True,
    },
    async_processing=True,
)

st.markdown("### 🎸 Live chord")

with st.container():
    with guitar_state.lock:
        current_chord = guitar_state.chord
        current_fingers = guitar_state.fingers

    if ctx.state.playing:
        if current_chord:
            st.subheader(f"🎶 Playing: {current_chord}")
            st.write(f"Detected extended fingers: {current_fingers}")
        else:
            st.write("Waiting for a hand gesture...")
    else:
        st.write("Start the camera and microphone to begin.")

st.caption(
    "Tip: Use a quiet room, keep your hand fully visible, "
    "and allow browser camera and microphone access."
)
