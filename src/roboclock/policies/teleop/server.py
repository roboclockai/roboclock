"""The browser side of teleoperation: one aiohttp server per process, on its own thread.

The page and its websocket use relative URLs, so the server works behind a path-prefix proxy
such as code-server's ``/proxy/<port>/``. The simulation thread publishes camera frames and a
status dict; the browser sends the set of held keys (repeated as a heartbeat) and one-shot
commands. Held keys that stop arriving are dropped (a dead-man), so a closed tab or a lost
connection stops the arms.
"""

import asyncio
import io
import json
import threading
import time
from pathlib import Path

import numpy as np

PAGE = Path(__file__).with_name("index.html")
KEY_TIMEOUT = 0.3  # s without a key-state message after which held keys are dropped
JPEG_QUALITY = 80


def encode_jpeg(image, quality=JPEG_QUALITY):
    from PIL import Image

    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[2] == 4:
        array = array[..., :3]
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


class TeleopServer:
    def __init__(self, host="0.0.0.0", port=8765):
        self.host, self.port = host, port
        self.lock = threading.Lock()
        self.keys = frozenset()
        self.keys_time = 0.0
        self.commands = []
        self.frames = {}  # camera name -> JPEG bytes, the newest frame
        self.frame_id = 0
        self.status = {}
        self.clients = 0
        self.loop = None
        self.error = None
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, name="roboclock-teleop", daemon=True)
        self.thread.start()
        if not self.ready.wait(10):
            raise RuntimeError(f"Teleop server did not start on {host}:{port}")
        if self.error is not None:
            raise self.error

    # ------------------------------------------------------------- simulation side
    def held_keys(self):
        with self.lock:
            if time.monotonic() - self.keys_time > KEY_TIMEOUT:
                return frozenset()
            return self.keys

    def pop_commands(self):
        with self.lock:
            commands, self.commands = self.commands, []
        return commands

    def publish_frames(self, images):
        """`images`: camera name -> HxWx3 array. Encoded here, on the caller's thread."""
        encoded = {name: encode_jpeg(image) for name, image in images.items() if image is not None}
        with self.lock:
            self.frames = encoded
            self.frame_id += 1

    def publish_status(self, **status):
        with self.lock:
            self.status = {**self.status, **status}

    # ------------------------------------------------------------------ server side
    def _run(self):
        try:
            from aiohttp import web

            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            app = web.Application()
            app.router.add_get("/", self._page)
            app.router.add_get("/index.html", self._page)
            app.router.add_get("/ws", self._socket)
            runner = web.AppRunner(app, access_log=None)
            self.loop.run_until_complete(runner.setup())
            self.loop.run_until_complete(web.TCPSite(runner, self.host, self.port).start())
        except Exception as error:  # reported to the constructor
            self.error = error
            self.ready.set()
            return
        self.ready.set()
        self.loop.run_forever()

    async def _page(self, request):
        from aiohttp import web

        return web.Response(
            text=PAGE.read_text(encoding="utf-8"),
            content_type="text/html",
            headers={"Cache-Control": "no-store"},
        )

    async def _socket(self, request):
        from aiohttp import WSMsgType, web

        socket = web.WebSocketResponse(heartbeat=10, max_msg_size=1 << 20)
        await socket.prepare(request)
        with self.lock:
            self.clients += 1
        sender = asyncio.ensure_future(self._send(socket))
        try:
            async for message in socket:
                if message.type != WSMsgType.TEXT:
                    continue
                try:
                    data = json.loads(message.data)
                except ValueError:
                    continue
                with self.lock:
                    if "keys" in data:
                        self.keys = frozenset(str(key) for key in data["keys"])
                        self.keys_time = time.monotonic()
                    if "command" in data:
                        self.commands.append(data)
        finally:
            sender.cancel()
            with self.lock:
                self.clients -= 1
                self.keys = frozenset()
        return socket

    async def _send(self, socket):
        """Push the newest frames (binary: name length, name, JPEG) and status (text)."""
        sent_frame, sent_status = -1, None
        while not socket.closed:
            with self.lock:
                frame_id, frames = self.frame_id, self.frames
                status = dict(self.status, clients=self.clients)
            try:
                if frame_id != sent_frame:
                    for name, jpeg in frames.items():
                        label = name.encode()
                        await socket.send_bytes(bytes([len(label)]) + label + jpeg)
                    sent_frame = frame_id
                if status != sent_status:
                    await socket.send_str(json.dumps(status))
                    sent_status = status
            except (ConnectionError, RuntimeError):
                return
            await asyncio.sleep(0.01)


_SERVER = None


def get_server(port=8765, host="0.0.0.0"):
    """The process's server; the evaluator builds a new policy per episode, the server lives on."""
    global _SERVER
    if _SERVER is None:
        _SERVER = TeleopServer(host, port)
    return _SERVER
