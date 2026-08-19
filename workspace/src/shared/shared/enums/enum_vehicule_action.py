# Define constants for vehicle actions
STOP = 0
START = 1
LEFT = 2
RIGHT = 3
NO_HAND = 4


def action_to_str(action: int) -> str:
    """Convert segment type to string representation."""
    if action == STOP:
        return "STOP"
    elif action == START:
        return "START"
    elif action == LEFT:
        return "LEFT"
    elif action == RIGHT:
        return "RIGHT"
    elif action == NO_HAND:
        return "NO HAND"
    else:
        return "UNKNOWN"