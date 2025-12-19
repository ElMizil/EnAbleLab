import os
import time
import asyncio
import queue
import json
import numpy as np
import sounddevice as sd
import soundfile as sf

from faster_whisper import WhisperModel
import edge_tts

# ==== HOTWORD con Vosk ====
USE_HOTWORD = True               # pon False si no quieres el "oye EVA"
WAKE_WORD = "eva"                # palabra de activación
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "./vosk-model-small-es-0.22")
VOSK_SAMPLE_RATE = 16000

# ========= Configuración de grabación =========
SAMPLE_RATE = 16000
CHANNELS = 1
SILENCE_DB = -40
SILENCE_MIN_SEC = 1.2
MAX_RECORD_SEC = 30
AUDIO_TMP = "grabacion.wav"

# ========= Whisper & TTS =========
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "auto")
LANG = "es"

VOICE = os.getenv("VOICE", "es-MX-DaliaNeural")
RATE = os.getenv("VOICE_RATE", "+0%")
PITCH = os.getenv("VOICE_PITCH", "+0Hz")

# ========= Utilidades =========
def rms_dbfs(x: np.ndarray, eps=1e-10) -> float:
    rms = np.sqrt(np.mean(np.square(x)) + eps)
    return 20.0 * np.log10(rms + eps)

def beep(dur=0.12, freq=880):
    """Beep simple (seno) para feedback de activación."""
    try:
        t = np.linspace(0, dur, int(SAMPLE_RATE * dur), False)
        tone = 0.3 * np.sin(2 * np.pi * freq * t)
        sd.play(tone.astype(np.float32), SAMPLE_RATE)
        sd.wait()
    except Exception:
        pass  # silencioso si falla

def grabar_automatico(filename: str) -> None:
    print("🎙️  Habla ahora... (se detiene con silencio)")
    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(status, flush=True)
        q.put(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32", callback=callback):
        audio = []
        silence_run = 0.0
        start = time.time()
        CHUNK_DUR = 0.1

        while True:
            try:
                data = q.get(timeout=1.0)
            except queue.Empty:
                continue

            audio.append(data)
            level_db = rms_dbfs(data.flatten())
            if level_db < SILENCE_DB:
                silence_run += CHUNK_DUR
            else:
                silence_run = 0.0

            if silence_run >= SILENCE_MIN_SEC and (time.time() - start) > 0.8:
                print("🔇 Silencio detectado, deteniendo.")
                break

            if time.time() - start >= MAX_RECORD_SEC:
                print("⏱️  Tiempo máximo alcanzado, deteniendo.")
                break

        audio_np = np.concatenate(audio, axis=0)
        sf.write(filename, audio_np, SAMPLE_RATE, subtype="PCM_16")
        dur = len(audio_np) / SAMPLE_RATE
        print(f"💾 Guardado: {filename} ({dur:.2f}s)")

def transcribir(filename: str, model: WhisperModel) -> str:
    print("🪄 Transcribiendo con Whisper...")
    segments, info = model.transcribe(filename, language=LANG, vad_filter=True)
    text = "".join([seg.text for seg in segments]).strip()
    print(f"⏺️  Detectado idioma: {info.language}, conf: {info.language_probability:.2f}")
    return text

async def decir(texto: str, voice: str = VOICE, rate: str = RATE, pitch: str = PITCH):
    if not texto:
        return
    print(f"🗣️  Hablando como {voice}...")
    communicate = edge_tts.Communicate(texto, voice=voice, rate=rate, pitch=pitch)
    out_mp3 = "respuesta.mp3"
    await communicate.save(out_mp3)
    # Intento de reproducción simple: abre con el SO
    try:
        if os.name == "nt":
            os.startfile(out_mp3)  # Windows
        elif hasattr(os, "uname") and os.uname().sysname == "Darwin":
            os.system(f"open '{out_mp3}'")  # macOS
        else:
            os.system(f"xdg-open '{out_mp3}'")  # Linux
    except Exception:
        print("Abre 'respuesta.mp3' con tu reproductor por defecto.")

def generar_respuesta(texto_usuario: str) -> str:
    if not texto_usuario:
        return "No escuché nada. ¿Podrías repetir?"
    low = texto_usuario.lower()
    if "hola" in low:
        return "¡Hola! Soy EVA, tu asistente de voz. ¿En qué te ayudo?"
    if "clima" in low:
        return "Aún no consulto el clima en línea, pero puedo hacerlo si me lo pides."
    return f"Entendí: {texto_usuario}"

# ========= Hotword (Vosk) =========
def esperar_hotword():
    """
    Escucha continuamente y dispara cuando reconoce la palabra 'eva'.
    Usa Vosk con un pequeño modelo en español y una gramática restringida para robustez.
    """
    print("🟢 Diciendo 'eva' me activo. (Ctrl+C para salir)")
    try:
        from vosk import Model, KaldiRecognizer
    except Exception as e:
        raise RuntimeError("Falta instalar 'vosk' (pip install vosk)") from e

    if not os.path.isdir(VOSK_MODEL_PATH):
        raise RuntimeError(f"No se encontró el modelo de Vosk en: {VOSK_MODEL_PATH}")

    model = Model(VOSK_MODEL_PATH)
    rec = KaldiRecognizer(model, VOSK_SAMPLE_RATE, json.dumps([WAKE_WORD]))
    rec.SetWords(True)

    # Stream de audio
    with sd.RawInputStream(samplerate=VOSK_SAMPLE_RATE, blocksize=8000, dtype='int16', channels=1):
        silencio = 0.0
        last_detect = 0.0
        while True:
            data = sd.RawInputStream.read  # not used directly
            # sounddevice no deja leer así; usamos read de la instancia
            frames, overflowed = sd.raw_input_stream.RawInputStream.read  # placeholder
            # NOTA: workaround: hacemos una lectura usando sd.RawInputStream como contexto y .read() real:
            # Para evitar complejidad, reabrimos un stream con callback más abajo.
            break

    # Implementación con callback para simplicidad:
    q = queue.Queue()

    def cb(indata, frames, time_info, status):
        if status:
            pass
        q.put(bytes(indata))

    with sd.RawInputStream(samplerate=VOSK_SAMPLE_RATE, blocksize=4000, dtype='int16', channels=1, callback=cb):
        cooldown = 0.8  # segundos para no disparar múltiples veces seguidas
        last_fire = 0.0
        while True:
            chunk = q.get()
            if rec.AcceptWaveform(chunk):
                res = json.loads(rec.Result() or "{}")
            else:
                res = json.loads(rec.PartialResult() or "{}")

            # Chequeo de parcial y final
            txt = (res.get("text") or res.get("partial") or "").strip().lower()
            if not txt:
                continue

            # Normalización básica (quitar tildes por si acaso)
            txt_norm = (txt
                        .replace("é", "e").replace("á", "a").replace("í", "i")
                        .replace("ó", "o").replace("ú", "u"))
            if WAKE_WORD in txt_norm and (time.time() - last_fire) > cooldown:
                print("✨ Hotword detectada: EVA")
                beep()
                return  # salir para permitir la interacción principal

def main():
    print("Cargando modelo Whisper… (primera vez puede tardar)")
    model = WhisperModel(WHISPER_MODEL, device=WHISPER_COMPUTE, compute_type="auto")

    print("\nAsistente EVA listo.")
    print("Modo: Hotword" if USE_HOTWORD else "Modo: Manual (ENTER para hablar)")

    while True:
        try:
            if USE_HOTWORD:
                esperar_hotword()               # Bloquea hasta oír "eva"
            else:
                cmd = input("\nENTER para grabar (o 'salir'): ").strip().lower()
                if cmd == "salir":
                    break

            grabar_automatico(AUDIO_TMP)
            try:
                texto = transcribir(AUDIO_TMP, model)
            except Exception as e:
                print(f"Error transcribiendo: {e}")
                continue

            print(f"📝 Tú dijiste: {texto}")
            respuesta = generar_respuesta(texto)
            print(f"🤖 EVA: {respuesta}")

            try:
                asyncio.run(decir(respuesta, VOICE, RATE, PITCH))
            except Exception as e:
                print(f"Error en TTS: {e}")
        except KeyboardInterrupt:
            print("\nSaliendo…")
            break

if __name__ == "__main__":
    main()
