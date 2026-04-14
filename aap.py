import os
import gradio as gr
import whisper
from groq import Groq
from gtts import gTTS

# =========================
# API KEY (FROM SECRETS)
# =========================
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

# =========================
# LOAD WHISPER
# =========================
model = whisper.load_model("base")

# =========================
# MEMORY
# =========================
chat_history = []

# =========================
# VOICE AI FUNCTION
# =========================
def voice_ai(audio):
    global chat_history

    if audio is None:
        return "No audio detected", None

    # 🎤 Speech → Text
    result = model.transcribe(audio)
    user_text = result["text"]

    chat_history.append({"role": "user", "content": user_text})

    # 🧠 LLM (Groq LLaMA)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Reply briefly and clearly."}
        ] + chat_history[-5:]
    )

    ai_reply = response.choices[0].message.content
    chat_history.append({"role": "assistant", "content": ai_reply})

    # 🔊 Text → Speech
    tts = gTTS(text=ai_reply, lang="en")
    tts.save("reply.mp3")

    return ai_reply, "reply.mp3"

# =========================
# RESET CHAT
# =========================
def reset_chat():
    global chat_history
    chat_history = []
    return "Chat cleared"

# =========================
# UI
# =========================
with gr.Blocks() as demo:
    gr.Markdown("# 🎙️ Voice AI Assistant")

    audio_input = gr.Audio(type="filepath", label="🎤 Speak here")
    text_output = gr.Textbox(label="🤖 AI Response")
    audio_output = gr.Audio(label="🔊 Voice Reply")

    btn = gr.Button("🗑️ Clear Chat")

    audio_input.change(
        fn=voice_ai,
        inputs=audio_input,
        outputs=[text_output, audio_output]
    )

    btn.click(
        fn=reset_chat,
        outputs=text_output
    )

demo.launch()
