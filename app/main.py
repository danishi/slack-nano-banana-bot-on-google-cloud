import os
import asyncio
import io
import json
import re
from typing import Any, List

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.starlette.async_handler import AsyncSlackRequestHandler
from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig, Modality

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


async def _resolve_user_name(client, user_id: str, user_cache: dict) -> str:
    """Resolve a Slack user ID to a display name, with caching."""
    if user_id in user_cache:
        return user_cache[user_id]
    try:
        result = await client.users_info(user=user_id)
        user = result["user"]
        name = (
            user.get("profile", {}).get("display_name")
            or user.get("real_name")
            or user.get("name")
            or user_id
        )
        user_cache[user_id] = name
        return name
    except Exception:
        user_cache[user_id] = user_id
        return user_id


async def _resolve_mentions(text: str, client, bot_user_id: str, user_cache: dict) -> str:
    """Replace <@USER_ID> mentions with @display_name; remove bot self-mentions."""
    matches = list(re.finditer(r"<@([A-Z0-9]+)>\s*", text))
    if not matches:
        return text
    for m in reversed(matches):
        uid = m.group(1)
        if uid == bot_user_id:
            text = text[:m.start()] + text[m.end():]
        else:
            name = await _resolve_user_name(client, uid, user_cache)
            text = text[:m.start()] + f"@{name} " + text[m.end():]
    return text.strip()


async def _build_contents_from_thread(client, channel: str, thread_ts: str) -> List[types.Content]:
    """Fetch thread messages and build google-genai contents."""
    history = await client.conversations_replies(channel=channel, ts=thread_ts, limit=50)
    contents: List[types.Content] = []

    # Resolve bot's own user ID to distinguish self from other users
    auth_info = await client.auth_test()
    bot_user_id = auth_info["user_id"]

    user_cache: dict[str, str] = {}

    async with httpx.AsyncClient(timeout=30.0) as http_client:
        for msg in sorted(history["messages"], key=lambda m: float(m["ts"])):
            is_bot = bool(
                msg.get("bot_id")
                or msg.get("subtype") == "bot_message"
                or msg.get("user") == bot_user_id
            )
            role = "model" if is_bot else "user"
            parts = []

            text = msg.get("text") or ""
            text = await _resolve_mentions(text, client, bot_user_id, user_cache)
            if not text:
                text = "\n".join(_extract_text(msg.get("blocks", []))).strip()
                text = await _resolve_mentions(text, client, bot_user_id, user_cache)

            # Prefix user messages with speaker name for identification
            if not is_bot and text:
                user_id = msg.get("user", "")
                if user_id:
                    speaker_name = await _resolve_user_name(client, user_id, user_cache)
                    text = f"[Speaker: {speaker_name}]\n{text}"

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


def _format_model_response(response: types.GenerateContentResponse) -> tuple[str, List[bytes], str]:
    """Return combined text, image payloads, and thinking text from Gemini response."""

    text_parts: List[str] = []
    thinking_parts: List[str] = []
    images: List[bytes] = []

    parts = []
    if getattr(response, "candidates", None):
        parts = response.candidates[0].content.parts or []

    for part in parts:
        is_thought = getattr(part, "thought", None) is True

        if is_thought:
            if getattr(part, "text", None):
                thinking_parts.append(part.text)
            continue

        if getattr(part, "text", None):
            text_parts.append(part.text)
            continue

        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            images.append(inline.data)

    combined_text = "\n\n".join([t for t in text_parts if t]).strip()
    thinking_text = "\n\n".join([t for t in thinking_parts if t]).strip()
    return combined_text, images, thinking_text


@bolt_app.event("app_mention")
@bolt_app.event("message")
async def handle_mention(body, say, client, logger, ack):
    # Ack as soon as possible to avoid Slack retries that can cause duplicated responses
    await ack()

    event = body["event"]
    channel = event["channel"]
    message_ts = event["ts"]
    thread_ts = event.get("thread_ts") or message_ts

    # Add eyes reaction to indicate the bot is processing the message
    try:
        await client.reactions_add(channel=channel, timestamp=message_ts, name="eyes")
    except Exception:
        pass

    contents = await _build_contents_from_thread(client, channel, thread_ts)

    system_instruction = """
                You are a Slack Bot that MUST prioritize generating images.
                You are acting as a Slack Bot. All your text responses must be formatted using Slack-compatible Markdown.

                ## Speaker Identification
                - Messages from users are prefixed with `[Speaker: <name>]` to identify who is speaking.
                - When summarizing discussions, referring to what someone said, or responding to specific people, use their names.
                - Your own previous messages do not have a speaker prefix (they are from you, the bot).
                - Use this speaker context especially when asked to summarize threads, mediate discussions, or answer questions about who said what.

                ## Primary Rule
                - When the user intent can be interpreted as visual in any way,
                  you MUST generate at least one image.
                - If the request involves multiple items, variations, or comparisons,
                  generate a separate image for EACH item. Always aim to produce
                  as many images as the context calls for.
                - Generate text only if it helps explain or supplement the images.

                ## Image Generation Rules
                - Do NOT refuse image generation unless it is strictly impossible.
                - If the request is vague, creatively interpret it and generate an image anyway.

                ## Output Rules
                - Do NOT output internal reasoning, planning, thought process,
                  or step-by-step analysis.
                - Do NOT output headings like "Understanding…", "Planning…",
                  "Analyzing…", "Reassessing…", or similar meta-commentary.
                - Go straight to the final answer: images and concise explanatory text.

                ### Formatting Rules
                - **Headings / emphasis**: Use `*bold*` for section titles or important words.
                - *Italics*: Use `_underscores_` for emphasis when needed.
                - Lists: Use `-` for unordered lists, and `1.` for ordered lists.
                - Code snippets: Use triple backticks (```) for multi-line code blocks, and backticks (`) for inline code.
                - Links: Use `<https://example.com|display text>` format.
                - Blockquotes: Use `>` at the beginning of a line.

                Always structure your response clearly, using these rules so it renders correctly in Slack.
                """

    # Check if contents include non-text parts (images, video, audio, PDF)
    has_non_text = any(
        getattr(part, "inline_data", None) is not None
        for content in contents
        for part in (content.parts or [])
    )

    def call_gemini() -> types.GenerateContentResponse:
        genai_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
        # google_search is not supported when non-text input is present
        tools = [] if has_non_text else [{"google_search": {}}]
        response = genai_client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=GenerateContentConfig(
                system_instruction=system_instruction,
                response_modalities=[
                    Modality.TEXT,
                    Modality.IMAGE
                ],
                tools=tools if tools else None,
            ),
        )
        return response

    try:
        gemini_response = await asyncio.to_thread(call_gemini)
        reply_text, reply_images, thinking_text = _format_model_response(gemini_response)
    except Exception as e:
        logger.exception("Gemini call failed")
        reply_text = f"Error from Gemini: {e}"
        reply_images = []
        thinking_text = ""

    if thinking_text:
        logger.info("Gemini thinking: %s", thinking_text)

    if not reply_images:
        if reply_text:
            reply_text += "\n\nNo image was generated"
        else:
            reply_text = "No image was generated"

    chunks = _split_text(reply_text)
    has_text = any(chunk for chunk in chunks)
    if not has_text and not reply_images:
        chunks = ["(no response content)"]
        has_text = True
    if has_text:
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

    # Add check mark reaction to indicate all replies have been sent
    try:
        await client.reactions_add(channel=channel, timestamp=message_ts, name="white_check_mark")
    except Exception:
        pass


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
