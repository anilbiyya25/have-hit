"""Analysis services for the asynchronous video pipeline.

Each stage is a separate module so it can be tested, replaced or skipped on its
own. `analysis_pipeline` is the only thing that knows the order they run in.

Nothing here imports `app`, so the whole pipeline is usable from a script or a
test without standing up FastAPI.
"""
