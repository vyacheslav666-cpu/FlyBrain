"""`flybrain serve`: сервер страницы и данных нейронов (FastAPI, коммит C7, D-015) и «живой» мозг (C9)."""

import asyncio
import json
import mimetypes
import sys
import threading
import time
import urllib.request
import webbrowser
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from flybrain import live, processed, webdata
from flybrain.engine import Engine
from flybrain.lookup import NeuronLookup
from flybrain.paths import PROCESSED_DIR, PROJECT_ROOT
from flybrain.serverctl import stop_old_server
from flybrain.version import code_version

WEB_DIR = PROJECT_ROOT / "web"
HOST = "127.0.0.1"  # только эта машина: наружу сервер не открываем
DEFAULT_PORT = 8765
PING_ANSWER = "flybrain"
# Версия кода, который этот процесс прочитал при запуске (D-028). Считаем один раз: если файлы потом изменятся, сервер станет «устаревшим».
CODE_VERSION = code_version()

# Модули в браузере должны приходить как JavaScript. На Windows тип берётся из реестра и бывает неверным.
mimetypes.add_type("text/javascript", ".js")


LIVE_QUEUE_FRAMES = 100  # ~4 с кадров: если страница не успевает их забирать, лишние выбрасываются


def create_app(processed_dir=PROCESSED_DIR, web_dir=WEB_DIR) -> FastAPI:
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        neurons = processed.load_neurons(processed_dir)
        legend = webdata.load_legend()
        names = webdata.load_yaml(webdata.NAMES_FILE)
        connectivity = processed.load_connectivity(processed_dir)
        state["cloud"] = webdata.build_cloud_gzip(neurons, legend)
        state["meta"] = webdata.build_meta(neurons, legend, names)
        state["lookup"] = NeuronLookup(neurons, connectivity, names, legend)
        # Живой мозг (C9): один повтор на видеокарте. Движок строится, только когда откроется первая страница.
        sugar = processed.load_groups(processed_dir)["stimuli"]["sugar"]
        state["live"] = live.LiveSim(
            lambda: Engine(connectivity["indptr"], connectivity["indices"], connectivity["weight"], n_trials=1),
            sugar["indices"], test_label=sugar["name_ru"], test_about=sugar["about_ru"])
        state["live"].start()
        yield
        state["live"].stop()

    app = FastAPI(title="FlyBrain", lifespan=lifespan)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(web_dir / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/api/ping")
    def ping() -> dict:
        return {"app": PING_ANSWER, "version": CODE_VERSION}

    @app.get("/api/meta")
    def meta() -> JSONResponse:
        return JSONResponse(state["meta"])

    @app.get("/api/cloud.bin")
    def cloud() -> Response:
        # Сжимаем один раз при запуске; браузер распаковывает сам.
        return Response(
            state["cloud"],
            media_type="application/octet-stream",
            headers={"Content-Encoding": "gzip", "Cache-Control": "no-cache"},
        )

    def known_neuron(index: int) -> NeuronLookup:
        lookup = state["lookup"]
        if not 0 <= index < len(lookup.neurons):
            raise HTTPException(status_code=404, detail="нет нейрона с таким номером")
        return lookup

    @app.get("/api/neuron/{index}")
    def neuron(index: int) -> JSONResponse:
        return JSONResponse(known_neuron(index).info(index), headers={"Cache-Control": "no-cache"})

    @app.get("/api/neuron/{index}/links")
    def neuron_links(index: int) -> JSONResponse:
        return JSONResponse(known_neuron(index).links(index), headers={"Cache-Control": "no-cache"})

    @app.get("/api/search")
    def search(q: str = Query(max_length=100)) -> JSONResponse:
        return JSONResponse(state["lookup"].search(q), headers={"Cache-Control": "no-cache"})

    @app.websocket("/ws/live")
    async def live_stream(socket: WebSocket) -> None:
        """Поток живого мозга: сервер шлёт кадры (двоичные) и состояние (JSON), страница — команды (JSON)."""
        await socket.accept()
        loop = asyncio.get_running_loop()
        outbox: asyncio.Queue = asyncio.Queue(maxsize=LIVE_QUEUE_FRAMES)

        def put(item) -> None:  # выполняется в цикле сервера
            if outbox.full() and isinstance(item, bytes):
                return  # страница не успевает: кадр пропадает (состояние не выбрасываем)
            if outbox.full():
                outbox.get_nowait()
            outbox.put_nowait(item)

        def send(item) -> None:  # вызывается из потока расчёта
            loop.call_soon_threadsafe(put, item)

        async def pump() -> None:
            while True:
                item = await outbox.get()
                if isinstance(item, bytes):
                    await socket.send_bytes(item)
                else:
                    await socket.send_text(item)

        sim = state["live"]
        sim.subscribe(send)
        sender = asyncio.create_task(pump())
        try:
            while True:
                message = await socket.receive_json()
                if isinstance(message, dict):
                    sim.command(message)
        except (WebSocketDisconnect, ValueError):
            pass
        finally:
            sim.unsubscribe(send)
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)

    app.mount("/web", StaticFiles(directory=web_dir), name="web")
    return app


def _ping(port: int) -> dict | None:
    """Ответ нашего сервера на `/api/ping` или `None`, если порт молчит или отвечает кто-то другой."""
    try:
        with urllib.request.urlopen(f"http://{HOST}:{port}/api/ping", timeout=1) as answer:
            data = json.loads(answer.read())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("app") == PING_ANSWER else None


def _port_is_busy(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        return probe.connect_ex((HOST, port)) == 0


def _open_browser_when_ready(port: int) -> None:
    url = f"http://{HOST}:{port}/"
    for _ in range(300):  # до 30 секунд: сервер сначала готовит данные
        answer = _ping(port)
        if answer is not None and answer.get("version") == CODE_VERSION:
            webbrowser.open(url)
            return
        time.sleep(0.1)


def run_serve(port: int = DEFAULT_PORT, open_browser: bool = True) -> int:
    if not (PROCESSED_DIR / processed.BUILD_INFO_FILE).exists():
        print("[ОШИБКА] нет подготовленных данных. Выполните по очереди:\n"
              "    uv run flybrain fetch-data\n    uv run flybrain build-data", file=sys.stderr)
        return 1
    url = f"http://{HOST}:{port}/"
    if _port_is_busy(port):
        answer = _ping(port)
        if answer is None:
            print(f"[ОШИБКА] порт {port} занят другой программой. Попробуйте: uv run flybrain serve --port 8766",
                  file=sys.stderr)
            return 1
        if answer.get("version") == CODE_VERSION:
            print(f"FlyBrain уже запущен: {url}")
            if open_browser:
                webbrowser.open(url)
            return 0
        # Тот же адрес, но сервер собран из другого кода (например, запущен до последнего коммита): заменяем.
        print(f"На порту {port} работает устаревшая копия FlyBrain "
              f"(версия кода {answer.get('version') or 'неизвестна'}, нужна {CODE_VERSION}). Останавливаю её…")
        result = stop_old_server(port, port_is_busy=lambda: _port_is_busy(port))
        if not result.ok:
            print(f"[ОШИБКА] {result.message}. Закройте окно старого сервера вручную и запустите снова.",
                  file=sys.stderr)
            return 1
        print("Старая копия остановлена.")

    import uvicorn

    print(f"FlyBrain: сервер запускается, адрес {url}")
    print("Чтобы остановить, закройте это окно или нажмите Ctrl+C.")
    if open_browser:
        threading.Thread(target=_open_browser_when_ready, args=(port,), daemon=True).start()
    uvicorn.run(create_app(), host=HOST, port=port, log_level="warning")
    return 0
