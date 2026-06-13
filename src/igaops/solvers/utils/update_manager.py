class UpdateManager:
    """
    Class to manages updating in solvers:
    Tracks the time-step and nonlinear iterations.
    """

    def __init__(self):
        self.step_count = 0
        self.newton_iter = 0

    def reset_newton(self):
        """Sets to zero the counter of Newton iteration."""
        self.newton_iter = 0

    def increment_newton(self):
        """Increases the counter of Newton iteration."""
        self.newton_iter += 1

    def increment_step(self):
        """
        Increases the counter of time-step.
        It also reset the counter of Newton and Lagrange.
        """
        self.step_count += 1
        self.reset_newton()

    def get_status(self):
        return f"UpdateManager, time-step: {self.step_count}, iteration: {self.newton_iter}"
