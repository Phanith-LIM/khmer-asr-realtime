import json
from pathlib import Path
from faster_whisper import WhisperModel
import io
import gradio as gr
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastrtc import (
    AdditionalOutputs,
    ReplyOnPause,
    Stream,
    audio_to_bytes,
    get_twilio_turn_credentials,
)
from gradio.utils import get_space
from pydantic import BaseModel


cur_dir = Path(__file__).parent

load_dotenv()


model_size = "./fast-whisper-small-khmer"
model = WhisperModel(model_size, device="cpu", compute_type="int8", local_files_only=True)


async def transcribe(audio: tuple[int, np.ndarray], transcript: str):
    # Convert audio to bytes and wrap in file-like object
    audio_bytes = audio_to_bytes(audio)
    audio_file_like = io.BytesIO(audio_bytes)
    
    # Transcribe
    segments, _ = model.transcribe(audio_file_like, beam_size=5)

    # Return updated transcript
    full_text = transcript + " " + "".join(segment.text for segment in segments)
    yield AdditionalOutputs(full_text)


transcript = gr.Textbox(label="Transcript")
stream = Stream(
    ReplyOnPause(transcribe),
    modality="audio",
    mode="send",
    additional_inputs=[transcript],
    additional_outputs=[transcript],
    additional_outputs_handler=lambda a, b: b,
    rtc_configuration=get_twilio_turn_credentials() if get_space() else None,
    concurrency_limit=5 if get_space() else None,
    time_limit=90 if get_space() else None,
)

app = FastAPI()

stream.mount(app)


class SendInput(BaseModel):
    webrtc_id: str
    transcript: str


@app.post("/send_input")
def send_input(body: SendInput):
    stream.set_input(body.webrtc_id, body.transcript)


@app.get("/transcript")
def _(webrtc_id: str):
    async def output_stream():
        async for output in stream.output_stream(webrtc_id):
            transcript = output.args[0].split("\n")[-1]
            yield f"event: output\ndata: {transcript}\n\n"

    return StreamingResponse(output_stream(), media_type="text/event-stream")


@app.get("/")
def index():
    rtc_config = get_twilio_turn_credentials() if get_space() else None
    html_content = (cur_dir / "index.html").read_text()
    html_content = html_content.replace("__RTC_CONFIGURATION__", json.dumps(rtc_config))
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import os

    if (mode := os.getenv("MODE")) == "UI":
        stream.ui.launch(server_port=7860)
    elif mode == "PHONE":
        stream.fastphone(host="0.0.0.0", port=7860)
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=7860)