"""
speech_engine.py
----------------
Asynchronous TTS using pyttsx3 with a background thread.
"""

import pyttsx3
import threading


class SpeechEngine:
    def __init__(self, rate: int = 150, volume: float = 1.0, voice_index: int = 0):
        self._rate = rate
        self._volume = volume
        self._voice_index = voice_index
        self._lock = threading.Lock()

    def start(self):
        pass

    def stop(self):
        pass

    def speak(self, text: str):
        if text.strip():
            t = threading.Thread(target=self._say, args=(text,), daemon=True)
            t.start()

    def _say(self, text: str):
        with self._lock:
            print(f"[TTS] Speaking: '{text}'")
            engine = pyttsx3.init()
            engine.setProperty("rate", self._rate)
            engine.setProperty("volume", self._volume)
            voices = engine.getProperty("voices")
            if voices and self._voice_index < len(voices):
                engine.setProperty("voice", voices[self._voice_index].id)
            engine.say(text)
            engine.runAndWait()
            engine.stop()

    def is_busy(self) -> bool:
        return False