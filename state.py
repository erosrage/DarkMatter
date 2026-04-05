import pickle
import os

STATE_FILE = "state.pkl"

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "rb") as f:
            return pickle.load(f)
    return {
        "positions": {},
        "profit": 0.0,
        "cycles": 0
    }

def save_state(state):
    with open(STATE_FILE, "wb") as f:
        pickle.dump(state, f)
