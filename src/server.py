import asyncio
import json
import os
import mimetypes
from http import HTTPStatus

try:
    import websockets
except ImportError:
    websockets = None

class DashboardServer:
    """
    Unified Async HTTP & WebSocket Server for the DRDO 2.5D LiDAR Mapping Dashboard.
    Serves web/ assets over HTTP on port 8000 and broadcasts real-time telemetry frames over WebSocket.
    """
    def __init__(self, host: str = "0.0.0.0", port: int = 8000, web_dir: str = "./web"):
        self.host = host
        self.port = port
        self.web_dir = os.path.abspath(web_dir)
        self.connected_clients = set()
        self.server = None
        self.loop = None
        self.latest_payload = None

    async def register(self, websocket):
        self.connected_clients.add(websocket)
        # Send latest frame immediately on connection
        if self.latest_payload:
            try:
                await websocket.send(self.latest_payload)
            except Exception:
                pass

    async def unregister(self, websocket):
        self.connected_clients.discard(websocket)

    async def ws_handler(self, websocket, path=None):
        await self.register(websocket)
        try:
            async for message in websocket:
                # Client incoming message handling (if any)
                pass
        except Exception:
            pass
        finally:
            await self.unregister(websocket)

    async def http_handler(self, reader, writer):
        """Ultra-fast non-blocking static HTTP file server for dashboard frontend"""
        try:
            line = await reader.readline()
            if not line:
                writer.close()
                return

            req_line = line.decode('utf-8', errors='ignore').strip()
            parts = req_line.split()
            if len(parts) < 2:
                writer.close()
                return

            method, path = parts[0], parts[1]
            # Read headers until empty line
            while True:
                h_line = await reader.readline()
                if not h_line or h_line == b'\r\n':
                    break

            if method != "GET":
                writer.write(b"HTTP/1.1 405 Method Not Allowed\r\n\r\n")
                await writer.drain()
                writer.close()
                return

            # Clean URL path
            clean_path = path.split('?')[0].lstrip('/')
            if clean_path == "" or clean_path == "/":
                clean_path = "index.html"

            target_file = os.path.normpath(os.path.join(self.web_dir, clean_path))
            # Security check: prevent directory traversal
            if not target_file.startswith(self.web_dir) or not os.path.exists(target_file) or os.path.isdir(target_file):
                writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\n\r\nNot Found")
                await writer.drain()
                writer.close()
                return

            mime_type, _ = mimetypes.guess_type(target_file)
            mime_type = mime_type or "application/octet-stream"

            with open(target_file, "rb") as f:
                content = f.read()

            header = (
                f"HTTP/1.1 200 OK\r\n"
                f"Content-Type: {mime_type}\r\n"
                f"Content-Length: {len(content)}\r\n"
                f"Access-Control-Allow-Origin: *\r\n"
                f"Connection: close\r\n\r\n"
            ).encode('utf-8')

            writer.write(header + content)
            await writer.drain()
        except Exception:
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    def broadcast_sync(self, frame_data: dict):
        """Synchronously queue a broadcast to all active WebSocket clients"""
        if not self.loop or not self.loop.is_running():
            return
            
        payload = json.dumps(frame_data)
        self.latest_payload = payload
        
        if len(self.connected_clients) > 0:
            asyncio.run_coroutine_threadsafe(self._broadcast_async(payload), self.loop)

    async def _broadcast_async(self, payload: str):
        if not self.connected_clients:
            return
            
        disconnected = set()
        for ws in self.connected_clients:
            try:
                await ws.send(payload)
            except Exception:
                disconnected.add(ws)
                
        for ws in disconnected:
            self.connected_clients.discard(ws)

    async def start(self):
        """Starts both HTTP server on port 8000 and WebSocket server on /ws (or port 8000)"""
        self.loop = asyncio.get_running_loop()
        
        # Start HTTP server on self.port (8000)
        await asyncio.start_server(self.http_handler, self.host, self.port)
        
        # Start WebSocket server on self.port or dedicated sub-handler
        if websockets is not None:
            # Run WebSocket server on same port if supported or port + 1
            self.ws_server = await websockets.serve(self.ws_handler, self.host, self.port + 1)
            print(f"[Dashboard Server] Web UI: http://localhost:{self.port} | WS: ws://localhost:{self.port + 1}")
        else:
            print(f"[Dashboard Server] Web UI: http://localhost:{self.port}")
