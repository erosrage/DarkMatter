import pickle
import os

STATE_FILE = "state.pkl"

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "rb") as f:
            state = pickle.load(f)
        # Migrate older state files that lack newer keys
        state.setdefault("open_orders", {})
        state.setdefault("initial_portfolio", None)
        return state
    return {
        "positions": {},
        "profit": 0.0,
        "cycles": 0,
        "open_orders": {},        # symbol -> list of open order IDs
        "initial_portfolio": None,
    }

def save_state(state):
    with open(STATE_FILE, "wb") as f:
        pickle.dump(state, f)
