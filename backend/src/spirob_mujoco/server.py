"""FastAPI server streaming simulation state over a WebSocket.

Protocol (JSON text frames):
  server -> client on connect:  {"type": "meta", ...SpiRobSim.meta()}
  server -> client at ~60 Hz:   {"type": "state", ...SpiRobSim.state()}
  client -> server commands:
    {"type": "cableForces", "values": [f0, f1, f2]}         # Newtons
    {"type": "autoGrasp", "primaryCable": 0}
    {"type": "reset"}
    {"type": "object", "kind": "sphere|box|cylinder|soft_sphere|none",
     "sizeMm": 52, "mass": 0.045}                            # rebuilds the sim

Positions are MuJoCo Z-up meters; the frontend adapter maps to Three.js
Y-up as (x, y, z)_three = (x, z, -y)_mujoco.
"""

import argparse
import asyncio
import contextlib
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .settings import OBJECT_KINDS, SimSettings
from .simulation import SpiRobSim

FRAME_RATE = 60.0

app = FastAPI(title="SpiRob MuJoCo backend")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _apply_command(sim: SpiRobSim, message: dict) -> SpiRobSim:
    kind = message.get("type")
    if kind == "cableForces":
        sim.set_cable_forces(message["values"])
    elif kind == "autoGrasp":
        sim.reset()
        sim.start_auto_grasp(int(message.get("primaryCable", 0)))
    elif kind == "reset":
        sim.reset()
    elif kind == "object":
        object_kind = message.get("kind", sim.settings.object_kind)
        if object_kind not in OBJECT_KINDS:
            raise ValueError(f"unknown object kind: {object_kind!r}")
        sim = SpiRobSim(
            SimSettings(
                stiffness=sim.settings.stiffness,
                damping_ratio=sim.settings.damping_ratio,
                cable_friction=sim.settings.cable_friction,
                body_friction=sim.settings.body_friction,
                object_kind=object_kind,
                object_size_mm=float(message.get("sizeMm", sim.settings.object_size_mm)),
                object_mass=float(message.get("mass", sim.settings.object_mass)),
            )
        )
    return sim


async def _receive_loop(websocket: WebSocket, queue: asyncio.Queue) -> None:
    while True:
        message = await websocket.receive_json()
        await queue.put(message)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    sim = SpiRobSim()
    await websocket.send_json({"type": "meta", **sim.meta()})

    queue: asyncio.Queue = asyncio.Queue()
    receiver = asyncio.create_task(_receive_loop(websocket, queue))
    frame_interval = 1.0 / FRAME_RATE
    next_frame = time.monotonic()

    try:
        while True:
            rebuilt = False
            while not queue.empty():
                previous = sim
                sim = _apply_command(sim, queue.get_nowait())
                rebuilt = rebuilt or sim is not previous
            if rebuilt:
                await websocket.send_json({"type": "meta", **sim.meta()})

            sim.step(frame_interval)
            await websocket.send_json({"type": "state", **sim.state()})

            next_frame += frame_interval
            delay = next_frame - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                # Fell behind real time (heavy flex scene): drop the deficit
                # instead of accumulating it.
                next_frame = time.monotonic()
                await asyncio.sleep(0)
    except WebSocketDisconnect:
        pass
    finally:
        receiver.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await receiver


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="SpiRob MuJoCo backend server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
