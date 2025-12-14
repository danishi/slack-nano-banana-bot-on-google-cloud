import os
import asyncio
import io
import json
from typing import Any, List

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.starlette.async_handler import AsyncSlackRequestHandler
from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig

# Environment variables
load_dotenv()
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_SIGNING_SECRET = os.environ["SLACK_SIGNING_SECRET"]
PROJECT_ID = os.environ.get("GOOGLE_PROJECT")
LOCATION = os.environ.get("MODEL_LOCATION", "global")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-3-pro-image-preview")
ALLOWED_SLACK_WORKSPACE = os.environ.get("ALLOWED_SLACK_WORKSPACE")

# Initialize Slack Bolt AsyncApp
bolt_app = AsyncApp(token=SLACK_BOT_TOKEN, signing_secret=SLACK_SIGNING_SECRET)
handler = AsyncSlackRequestHandler(bolt_app)

fastapi_app = FastAPI()


def _extract_text(obj: Any) -> List[str]:
    texts: List[str] = []
    if isinstance(obj, dict):
        t = obj.get("text")
        if isinstance(t, str):
            texts.append(t)
        for v in obj.values():
            texts.extend(_extract_text(v))
    elif isinstance(obj, list):
        for item in obj:
            texts.extend(_extract_text(item))
    return texts


async def _build_contents_from_thread(client, channel: str, thread_ts: str) -> List[types.Content]:
    """Fetch thread messages and build google-genai contents."""
    history = await client.conversations_replies(channel=channel, ts=thread_ts, limit=50)
    contents: List[types.Content] = []

    import re
    async with httpx.AsyncClient(timeout=30.0) as http_client:
        for msg in sorted(history["messages"], key=lambda m: float(m["ts"])):
            is_bot = bool(msg.get("bot_id") or msg.get("subtype") == "bot_message")
            role = "model" if is_bot else "user"
            parts = []

            text = msg.get("text") or ""
            text = re.sub(r"<@[^>]+>\s*", "", text).strip()
            if not text:
                text = "\n".join(_extract_text(msg.get("blocks", []))).strip()

            if text:
                parts.append(types.Part.from_text(text=text))

            for f in msg.get("files", []):
                mimetype = (f.get("mimetype") or "")
                url = f.get("url_private_download")
                if not url:
                    continue

                supported = (
                    mimetype.startswith(("image/", "video/", "audio/", "text/"))
                    or mimetype == "application/pdf"
                )
                if not supported:
                    continue

                resp = await http_client.get(
                    url,
                    headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
                )
                resp.raise_for_status()

                if mimetype.startswith("text/"):
                    parts.append(types.Part.from_text(text=resp.text))
                else:
                    parts.append(types.Part.from_bytes(data=resp.content, mime_type=mimetype))

            if parts:
                contents.append(types.Content(role=role, parts=parts))

    if not contents:
        contents = [types.Content(role="user", parts=[types.Part.from_text(text="(no content)")])]
    return contents


def _split_text(text: str, limit: int = 3000) -> List[str]:
    """Split text into chunks that fit within Slack's block text limit."""
    if not text:
        return [""]
    return [text[i : i + limit] for i in range(0, len(text), limit)]


def _format_model_response(response: types.GenerateContentResponse) -> tuple[str, List[bytes]]:
    """Return combined text and image payloads from Gemini response."""

    text_parts: List[str] = []
    images: List[bytes] = []

    for part in response.parts:
        if getattr(part, "thought", None):
            continue
        if part.text:
            text_parts.append(part.text)
        else:
            image = part.as_image()
            if image:
                buffer = io.BytesIO()
                image.save(buffer, format=image.format or "PNG")
                images.append(buffer.getvalue())

    combined_text = "\n\n".join(text_parts) if text_parts else (response.text or "")
    return combined_text, images


@bolt_app.event("app_mention")
@bolt_app.event("message")
async def handle_mention(body, say, client, logger, ack):
    # Ack as soon as possible to avoid Slack retries that can cause duplicated responses
    await ack()

    event = body["event"]
    channel = event["channel"]
    thread_ts = event.get("thread_ts") or event["ts"]

    contents = await _build_contents_from_thread(client, channel, thread_ts)

    def call_gemini() -> types.GenerateContentResponse:
        genai_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
        response = genai_client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=GenerateContentConfig(
                system_instruction="""
                You are acting as a Slack Bot. All your responses must be formatted using Slack-compatible Markdown.

                ### Formatting Rules
                - **Headings / emphasis**: Use `*bold*` for section titles or important words.
                - *Italics*: Use `_underscores_` for emphasis when needed.
                - Lists: Use `-` for unordered lists, and `1.` for ordered lists.
                - Code snippets: Use triple backticks (```) for multi-line code blocks, and backticks (`) for inline code.
                - Links: Use `<https://example.com|display text>` format.
                - Blockquotes: Use `>` at the beginning of a line.

                Always structure your response clearly, using these rules so it renders correctly in Slack.
                """,
                response_modalities=["TEXT", "IMAGE"],
                tools=[
                    {"url_context": {}},
                    {"google_search": {}},
                ],
            )
        )
        return response

    try:
        gemini_response = await asyncio.to_thread(call_gemini)
        reply_text, reply_images = _format_model_response(gemini_response)
    except Exception as e:
        logger.exception("Gemini call failed")
        reply_text = f"Error from Gemini: {e}"
        reply_images = []
    chunks = _split_text(reply_text)
    if all(not chunk for chunk in chunks):
        chunks = ["Here are your results from Gemini."] if reply_images else ["(no response content)"]
    first_chunk, *rest_chunks = chunks
    await say(
        blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": first_chunk}}],
        text=first_chunk,
        thread_ts=thread_ts,
    )
    for chunk in rest_chunks:
        await say(
            blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": chunk}}],
            text=chunk,
            thread_ts=thread_ts,
        )

    for idx, image_bytes in enumerate(reply_images, start=1):
        await client.files_upload_v2(
            channel=channel,
            thread_ts=thread_ts,
            filename=f"gemini-response-{idx}.png",
            title=f"Gemini response {idx}",
            file=io.BytesIO(image_bytes),
        )


@fastapi_app.post("/slack/events")
async def slack_events(req: Request):
    retry_num = req.headers.get("x-slack-retry-num")
    if retry_num is not None:
        return JSONResponse(status_code=404, content={"error": "ignored_slack_retry"})

    raw_body = await req.body()
    data = json.loads(raw_body)
    challenge = data.get("challenge")
    if challenge:
        return JSONResponse(content={"challenge": challenge})

    team_id = data.get("team_id")
    if ALLOWED_SLACK_WORKSPACE and team_id != ALLOWED_SLACK_WORKSPACE:
        return JSONResponse(status_code=403, content={"error": f"{team_id}:workspace_not_allowed"})
    return await handler.handle(req)


@fastapi_app.get("/")
async def root():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:fastapi_app", host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
