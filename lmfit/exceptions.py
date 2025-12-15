class MinimizerException(Exception):
    """General Purpose Exception."""

    def __init__(self, msg):
        Exception.__init__(self)
        self.msg = msg

    def __str__(self):
        """string"""
        return f"{self.msg}"


class AbortFitException(MinimizerException):
    """Raised when a fit is aborted by the user."""
