# === Monitor de Atención Visual en Tiempo Real ===
# Autor: Nicolás Bargioni | ISSD 2025 | Procesamiento de Imágenes

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="Monitor de Atención", page_icon="🎯", layout="wide")

# Inicializar MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# Configuración RTC para WebRTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
    ]}
)

# === Estado de sesión ===
if "ventana_atencion" not in st.session_state:
    st.session_state.ventana_atencion = deque(maxlen=100)
    st.session_state.x_vals = deque(maxlen=100)
    st.session_state.total_frames = 0
    st.session_state.atencion_frames = 0
    st.session_state.attention_log = []

# === UI Principal ===
st.title("🎯 Monitor de Atención Visual en Tiempo Real")
st.markdown("""
Este programa utiliza visión por computadora para analizar tu nivel de atención durante una videollamada.
Evalúa si tu rostro está centrado y si tu mirada se mantiene hacia el frente.
Ideal para contextos educativos, de trabajo remoto o validación de presencia.

👁️‍🗨️ A través de la webcam, el sistema detecta si desviás la mirada, girás la cabeza o bajás la vista,
y muestra un indicador visual de atención junto a un gráfico en tiempo real.
""")

st.subheader("👉 La premisa es la siguiente 👈")
st.markdown("Para demostrar tu atención, procurá estar justo en medio de donde te muestra la cámara 😉")

# === Sidebar ===
with st.sidebar:
    st.subheader("🤓 Umbrales de Atención")

    with st.expander("🎛 Ajustes de Umbrales", expanded=False):
        st.markdown("""
        Ajustá la sensibilidad del sistema de atención:
        - **Giro izquierda/derecha**: margen de movimiento horizontal permitido.
        - **Cabeza baja**: inclinación vertical antes de penalizar.
        """)
        umbral_giro_izquierda = st.slider("Giro hacia izquierda", 0.0, 1.0, 0.4, step=0.01)
        umbral_giro_derecha = st.slider("Giro hacia derecha", 0.0, 1.0, 0.6, step=0.01)
        umbral_ojos_y_baja = st.slider("Cabeza baja", 0.0, 1.0, 0.25, step=0.01)

    st.markdown("---")
    st.subheader("⚙️ Configuración")

    mostrar_landmarks = st.checkbox("😀 Mostrar landmarks faciales", value=True)
    st.caption("Visualiza los puntos y líneas sobre tu rostro (FaceMesh).")

    usar_segmentacion = st.checkbox("🖼 Activar segmentación semántica", value=True)
    st.caption("Valida que haya una persona real (no una imagen).")

    ver_mascara = st.checkbox("👽 Ver máscara de segmentación", value=False)
    st.caption("Superpone una máscara verde sobre la persona detectada.")

    st.markdown("---")

    if st.button("🔄 Reiniciar estadísticas"):
        st.session_state.ventana_atencion.clear()
        st.session_state.x_vals.clear()
        st.session_state.total_frames = 0
        st.session_state.atencion_frames = 0
        st.session_state.attention_log.clear()
        st.rerun()

# === Funciones de procesamiento ===

def evaluar_atencion(landmarks, w, h, umbral_izq, umbral_der, umbral_bajo):
    """Evalúa si el rostro está centrado y la mirada al frente"""
    score = 0
    detalles = []

    nose = landmarks.landmark[1]
    right_eye = landmarks.landmark[33]
    left_eye = landmarks.landmark[263]
    eyes_center_x = (right_eye.x + left_eye.x) / 2
    eyes_center_y = (right_eye.y + left_eye.y) / 2

    if umbral_izq < nose.x < umbral_der:
        score += 0.5
    else:
        detalles.append("Nariz fuera de centro")

    if umbral_izq < eyes_center_x < umbral_der:
        score += 0.5
    else:
        detalles.append("Ojos fuera de centro")

    forehead_y = landmarks.landmark[10].y
    chin_y = landmarks.landmark[152].y
    nose_rel_y = (nose.y - forehead_y) / (chin_y - forehead_y) if (chin_y - forehead_y) > 0 else 0

    if nose_rel_y > 0.7:
        score = 0
        detalles.append("Cabeza muy baja")
    elif eyes_center_y > umbral_bajo:
        score -= 0.3
        detalles.append("Mirada baja")

    return max(0, score), detalles

def dibujar_landmarks(image, landmarks):
    """Dibuja la malla facial"""
    mp_drawing.draw_landmarks(
        image,
        landmarks,
        mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1)
    )
    return image

def detectar_presencia(mask, umbral=0.1):
    """Verifica si hay persona real"""
    return np.mean(mask > 0.6) > umbral

def aplicar_mascara(frame, mask, alpha=0.4):
    """Aplica máscara de segmentación verde"""
    mask_3c = np.stack([mask] * 3, axis=-1)
    color_mask = np.zeros_like(frame, dtype=np.uint8)
    color_mask[:] = (0, 255, 0)
    blended = np.where(
        mask_3c > 0.1,
        (alpha * frame + (1 - alpha) * color_mask).astype(np.uint8),
        frame
    )
    return blended

def graficar_atencion(ventana, x_vals):
    """Genera gráfico de atención"""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(list(x_vals), list(ventana), color='limegreen', linewidth=2)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Atención (%)')
    ax.set_title('Índice de Atención en Tiempo Real')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

# === Procesador de Video ===

class AttentionProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.segmentador = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
        self.umbral_izq = 0.4
        self.umbral_der = 0.6
        self.umbral_bajo = 0.25
        self.mostrar_landmarks = True
        self.usar_segmentacion = True
        self.ver_mascara = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        results = self.face_mesh.process(rgb)
        segment = self.segmentador.process(rgb)

        st.session_state.total_frames += 1
        hay_persona = True

        if self.usar_segmentacion:
            hay_persona = detectar_presencia(segment.segmentation_mask)

        if hay_persona and results.multi_face_landmarks:
            for rostro in results.multi_face_landmarks:
                if self.mostrar_landmarks:
                    img = dibujar_landmarks(img, rostro)

                score, _ = evaluar_atencion(
                    rostro, w, h,
                    self.umbral_izq, self.umbral_der, self.umbral_bajo
                )

                if score >= 0.7:
                    st.session_state.atencion_frames += 1
                    texto = "ATENTO"
                    color = (0, 255, 0)
                else:
                    texto = "NO ATENTO"
                    color = (0, 0, 255)

                cv2.putText(img, texto, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        else:
            cv2.putText(img, "Sin rostro", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 255), 2)

        # Calcular índice
        if st.session_state.total_frames > 0:
            idx = int((st.session_state.atencion_frames / st.session_state.total_frames) * 100)
            st.session_state.ventana_atencion.append(idx)
            st.session_state.x_vals.append(st.session_state.total_frames)
            st.session_state.attention_log.append(idx)
            cv2.putText(img, f"Atencion: {idx}%", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if self.ver_mascara and self.usar_segmentacion:
            img = aplicar_mascara(img, segment.segmentation_mask)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# === Layout principal ===

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Video en Tiempo Real")
    ctx = webrtc_streamer(
        key="attention-monitor",
        video_processor_factory=AttentionProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Actualizar parámetros del procesador
    if ctx.video_processor:
        ctx.video_processor.umbral_izq = umbral_giro_izquierda
        ctx.video_processor.umbral_der = umbral_giro_derecha
        ctx.video_processor.umbral_bajo = umbral_ojos_y_baja
        ctx.video_processor.mostrar_landmarks = mostrar_landmarks
        ctx.video_processor.usar_segmentacion = usar_segmentacion
        ctx.video_processor.ver_mascara = ver_mascara

with col2:
    st.subheader("📊 Estadísticas")

    if st.session_state.total_frames > 0:
        idx_actual = int((st.session_state.atencion_frames / st.session_state.total_frames) * 100)
        st.metric("🎯 Índice de Atención", f"{idx_actual}%")
        st.metric("📷 Frames procesados", st.session_state.total_frames)
        st.metric("✅ Frames atentos", st.session_state.atencion_frames)

        if len(st.session_state.ventana_atencion) > 1:
            fig = graficar_atencion(st.session_state.ventana_atencion, st.session_state.x_vals)
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info("Iniciá el video para ver las estadísticas")

# === Resumen final ===
if st.session_state.attention_log:
    with st.expander("📋 Ver resumen completo"):
        promedio = sum(st.session_state.attention_log) / len(st.session_state.attention_log)
        st.markdown(f"**Promedio total de atención:** {promedio:.2f}%")
        st.line_chart(st.session_state.attention_log)

# === Footer ===
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    Desarrollado por <strong>Nicolás Bargioni</strong> | Año 2025 | ISSD: Inteligencia Artificial y Ciencia de Datos 🧠
</div>
""", unsafe_allow_html=True)
