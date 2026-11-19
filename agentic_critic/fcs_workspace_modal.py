"""One persistent, network-isolated CPU workspace per FCS episode."""

import asyncio
from contextlib import asynccontextmanager

import modal


@asynccontextmanager
async def open_workspace(timeout, memory_mb=2048):
    app = await modal.App.lookup.aio("miles-fcs-workspaces", create_if_missing=True)
    image = modal.Image.debian_slim().apt_install("g++").run_commands("mkdir -p /work")
    box = await modal.Sandbox.create.aio(
        app=app,
        image=image,
        workdir="/work",
        block_network=True,
        cpu=1,
        memory=memory_mb,
        timeout=timeout,
        runtime="vm",
    )
    try:

        async def execute(command, seconds):
            proc = await box.exec.aio(
                "timeout",
                "-k",
                "2",
                str(seconds),
                "bash",
                "-c",
                command,
                workdir="/work",
                timeout=seconds + 10,
            )
            stdout, stderr, code = await asyncio.gather(proc.stdout.read.aio(), proc.stderr.read.aio(), proc.wait.aio())
            if code == 124:
                raise TimeoutError()
            return {"returncode": code, "output": stdout + stderr}

        yield execute
    finally:
        await box.terminate.aio()
