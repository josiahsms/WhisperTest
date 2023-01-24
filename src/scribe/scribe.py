import io
import tempfile
from typing import Any

import numpy as np
import pyperclip
import streamlit as st
import whisper


class Scribe:
    def __init__(self) -> None:
        with st.spinner("Loading model..."):
            self.model = whisper.load_model("base")

        self.options = whisper.DecodingOptions(fp16=False)

    def run(self):
        st.title("Scribe")

        audio_file = st.file_uploader("Upload a file")

        if audio_file is None:
            return

        st.audio(audio_file)

        result = self.process_audio(audio_file)

        st.header("Result")
        st.write(result)  # type: ignore

        st.header("Copy to clipboard")
        if st.button("Copy"):
            pyperclip.copy(result)  # type: ignore

    @st.experimental_memo
    def process_audio(_self, audio_file: io.BytesIO):
        audio = _self.load_audio(audio_file)
        result = _self.transcribe(audio)
        return result

    def load_audio(self, file: io.BytesIO) -> np.ndarray[Any, Any]:
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(file.read())

            audio = whisper.load_audio(tmp.name)

        return audio

    def transcribe(self, audio: np.ndarray[Any, Any]) -> str:
        audio = whisper.pad_or_trim(audio)  # type: ignore

        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)  # type: ignore

        result = whisper.decode(
            self.model,
            mel,
            self.options,
        )

        return result.text  # type: ignore
