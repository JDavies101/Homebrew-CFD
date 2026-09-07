# lightweight progress bar + ETA for long simulation loops
import time

class Progress:
    def __init__(self, total):
        self.total = total
        self.t0 = time.time()

    # call once per step; `health` is an optional value to display (e.g. max|u|)
    def update(self, step, health=None):
        done = step + 1
        el = time.time() - self.t0
        frac = done / self.total
        eta = el / frac - el if frac > 0 else 0.0
        bar = "#" * int(30 * frac) + "-" * (30 - int(30 * frac))
        msg = f"\r[{bar}] {done}/{self.total} {frac*100:5.1f}%  {el:5.0f}s  ETA {eta:5.0f}s"
        if health is not None:
            msg += f"  max|u|={health:.3g}"
        print(msg, end="", flush=True)

    def done(self):
        print()   # newline after the bar
