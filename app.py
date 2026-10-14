import streamlit as st
import cv2
import mediapipe as mp
import av
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


# ============================================================
# 🎸 GESTURE GUITAR AI
# Hand gestures control different guitar chords
# Webcam + MediaPipe + Streamlit WebRTC
# ============================================================

st.set_page_config(
    page_title="Gesture Guitar AI",
    page_icon="🎸",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
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
        margin-top: 20px;
        border: 2px solid #555;
    }

    .chord-name {
        font-size: 55px;
        font-weight: bold;
    }

    .gesture-table {
        font-size: 18px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🎸 GESTURE GUITAR AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Control guitar chords using your hand gestures</div>',
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "chord" not in st.session_state:
    st.session_state.chord = "Waiting..."

if "gesture" not in st.session_state:
    st.session_state.gesture = "No hand detected"


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# ============================================================
# FINGER DETECTION
# ============================================================

def count_fingers(hand_landmarks, handedness):
    """
    Detects how many fingers are raised.
    """

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

    if landmarks[8].y < landmarks[6].y:
        fingers.append(1)
    else:
        fingers.append(0)

    # --------------------------------------------------------
    # MIDDLE
    # --------------------------------------------------------

    if landmarks[12].y < landmarks[10].y:
        fingers.append(1)
    else:
        fingers.append(0)

    # --------------------------------------------------------
    # RING
    # --------------------------------------------------------

    if landmarks[16].y < landmarks[14].y:
        fingers.append(1)
    else:
        fingers.append(0)

    # --------------------------------------------------------
    # LITTLE
    # --------------------------------------------------------

    if landmarks[20].y < landmarks[18].y:
        fingers.append(1)
    else:
        fingers.append(0)

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

        self.last_change = time.time()

    def recv(self, frame):

        # Convert WebRTC frame to OpenCV
        image = frame.to_ndarray(format="bgr24")

        # Mirror webcam image
        image = cv2.flip(image, 1)

        # Convert BGR → RGB
        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # Detect hands
        results = self.hands.process(rgb_image)

        detected_chord = "Waiting..."
        detected_gesture = "No hand detected"

        # ----------------------------------------------------
        # HAND DETECTED
        # ----------------------------------------------------

        if results.multi_hand_landmarks:

            hand_landmarks = results.multi_hand_landmarks[0]

            handedness = results.multi_handedness[0].classification[0].label

            finger_count = count_fingers(
                hand_landmarks,
                handedness
            )

            detected_chord, detected_gesture = get_chord(
                finger_count
            )

            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Display gesture on camera
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
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 150),
                3
            )

        else:

            detected_chord = "Waiting..."
            detected_gesture = "No hand detected"

            cv2.putText(
                image,
                "Show your hand",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

        # Store latest result
        self.chord = detected_chord
        self.gesture = detected_gesture

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )


# ============================================================
# MAIN LAYOUT
# ============================================================

left, right = st.columns([2, 1])


# ============================================================
# WEBCAM
# ============================================================

with left:

    st.subheader("📷 Your Webcam")

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
# CHORD DISPLAY
# ============================================================

with right:

    st.subheader("🎵 Current Chord")

    chord_placeholder = st.empty()
    gesture_placeholder = st.empty()

    if ctx.video_processor:

        chord = ctx.video_processor.chord
        gesture = ctx.video_processor.gesture

        chord_placeholder.markdown(
            f"""
            <div class="chord-box">
                <div>Current Chord</div>
                <div class="chord-name">{chord}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        gesture_placeholder.info(
            f"Detected Gesture: {gesture}"
        )

    else:

        chord_placeholder.markdown(
            """
            <div class="chord-box">
                <div>Current Chord</div>
                <div class="chord-name">🎸</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        gesture_placeholder.info(
            "Click START and show your hand."
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
# INSTRUCTIONS
# ============================================================

st.divider()

st.subheader("🎮 How to Play")

st.write(
    """
    1. Click **START** above.
    
    2. Allow the browser to access your webcam.
    
    3. Show one hand clearly in front of the camera.
    
    4. Change your hand gesture.
    
    5. The AI detects the number of raised fingers.
    
    6. The corresponding guitar chord appears on the screen.
    """
)

st.info(
    "💡 Tip: Keep your hand clearly visible and "
    "reasonably close to the webcam."
)
