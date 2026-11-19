r"""
Keep 3D Slicer open and reload the reconstruction automatically.

Run once inside Slicer (TRHexArray3D_gpu.py does this for you the first time,
by launching:  Slicer --python-script slicer_watch.py).
After that, every time the reconstruction writes img_slicer.npz, this clears
the scene and reloads it with load_recon_into_slicer.py - no new window.

Must sit in the same folder as load_recon_into_slicer.py and img_slicer.npz.
"""
import os
import time
import qt
import slicer

HERE   = os.path.dirname(os.path.abspath(globals().get("__file__", os.getcwd())))
WATCH  = os.path.join(HERE, "img_slicer.npz")             # file written by the reconstruction
LOADER = os.path.join(HERE, "load_recon_into_slicer.py")  # your loader, unchanged
BEAT   = os.path.join(HERE, ".slicer_watching")           # tells the reconstruction Slicer is open

_state = {"mtime": None}


def _tick():
    # heartbeat: the reconstruction script checks this file's age to see if Slicer is open
    try:
        with open(BEAT, "w") as f:
            f.write(str(time.time()))
    except OSError:
        pass

    try:
        m = os.path.getmtime(WATCH)
    except OSError:
        return                       # no reconstruction yet
    if m == _state["mtime"]:
        return                       # nothing new
    _state["mtime"] = m

    slicer.mrmlScene.Clear(0)        # remove the previous volume (and any leftovers)
    try:
        exec(open(LOADER).read(), {"__file__": LOADER})
    except Exception as e:
        print("Reload failed:", e)


# if this script is run twice, stop the older watcher first
_old = getattr(slicer, "_recon_watch_timer", None)
if _old is not None:
    _old.stop()

_timer = qt.QTimer()
_timer.timeout.connect(_tick)
_timer.start(1000)                   # check once a second
slicer._recon_watch_timer = _timer   # keep a reference so it isn't garbage-collected
_tick()
print(f"Watching {WATCH}: rerun the reconstruction and this window will reload it.")