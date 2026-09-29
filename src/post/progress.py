# lightweight progress bar + ETA for long simulation loops
import time

class Progress:
    """
    Single-line progress bar with elapsed time, ETA and an optional health value (e.g. max|u|).
    """

    def __init__(self, total):
        """
        Start the timer for a loop of total steps.
        """

        self.total = total
        self.start_time = time.time()
        self.last_health = None

    def update(self, step, health=None):
        """
        Redraw the bar; call once per step, health is an optional value to display.
        """

        steps_done = step + 1
        elapsed = time.time() - self.start_time
        fraction = steps_done / self.total
        eta = elapsed / fraction - elapsed if fraction > 0 else 0.0
        bar = "#" * int(30 * fraction) + "-" * (30 - int(30 * fraction))
        message = f"\r[{bar}] {steps_done}/{self.total} {fraction * 100:5.1f}%  {elapsed:5.0f}s  ETA {eta:5.0f}s"

        if health is not None:
            self.last_health = health
            message += f"  max|u|={health:.3g}"

        print(message, end="", flush=True)

    def done(self):
        """
        Draw the full bar and end the line, keeping the last health reading visible.
        """

        elapsed = time.time() - self.start_time
        message = f"\r[{'#' * 30}] {self.total}/{self.total} 100.0%  {elapsed:5.0f}s  ETA     0s"

        if self.last_health is not None:
            message += f"  max|u|={self.last_health:.3g}"

        print(message + " " * 10)  # trailing spaces clear any leftover tail
