"""`flybrain serve`: сервер страницы и данных нейронов (FastAPI, коммит C7, D-015)."""

import mimetypes
import sys
import threading
import time
import urllib.request
import webbrowser
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from flybrain import processed, webdata
from flybrain.lookup import NeuronLookup
from flybrain.paths import PROCESSED_DIR, PROJECT_ROOT

WEB_DIR = PROJECT_ROOT / "web"
HOST = "127.0.0.1"  # только эта машина: наружу сервер не открываем
DEFAULT_PORT = 8765
PING_ANSWER = "flybrain"

# Модули в браузере должны приходить как JavaScript. На Windows тип берётся из реестра и бывает неверным.
mimetypes.add_type("text/javascript", ".js")


def create_app(processed_dir=PROCESSED_DIR, web_dir=WEB_DIR) -> FastAPI:
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        neurons = processed.load_neurons(processed_dir)
        legend = webdata.load_legend()
        names = webdata.load_yaml(webdata.NAMES_FILE)
        state["cloud"] = webdata.build_cloud_gzip(neurons, legend)
        state["meta"] = webdata.build_meta(neurons, legend, names)
        state["lookup"] = NeuronLookup(neurons, processed.load_connectivity(processed_dir), names, legend)
        yield

    app = FastAPI(title="FlyBrain", lifespan=lifespan)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(web_dir / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/api/ping")
    def ping() -> dict:
        return {"app": PING_ANSWER}

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

    app.mount("/web", StaticFiles(directory=web_dir), name="web")
    return app


def _our_server_answers(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://{HOST}:{port}/api/ping", timeout=1) as answer:
            return PING_ANSWER.encode() in answer.read()
    except OSError:
        return False


def _port_is_busy(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        return probe.connect_ex((HOST, port)) == 0


def _open_browser_when_ready(port: int) -> None:
    url = f"http://{HOST}:{port}/"
    for _ in range(300):  # до 30 секунд: сервер сначала готовит данные
        if _our_server_answers(port):
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
        if _our_server_answers(port):
            print(f"FlyBrain уже запущен: {url}")
            if open_browser:
                webbrowser.open(url)
            return 0
        print(f"[ОШИБКА] порт {port} занят другой программой. Попробуйте: uv run flybrain serve --port 8766",
              file=sys.stderr)
        return 1

    import uvicorn

    print(f"FlyBrain: сервер запускается, адрес {url}")
    print("Чтобы остановить, закройте это окно или нажмите Ctrl+C.")
    if open_browser:
        threading.Thread(target=_open_browser_when_ready, args=(port,), daemon=True).start()
    uvicorn.run(create_app(), host=HOST, port=port, log_level="warning")
    return 0
