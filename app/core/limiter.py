from slowapi import Limiter
from slowapi.util import get_remote_address

# Global limiter instance — imported by endpoints and wired into app in main.py
limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])
