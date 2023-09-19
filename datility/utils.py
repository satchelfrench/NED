
import subprocess
import importlib.util

def _check_and_install(name: str) -> bool:
    spec = importlib.util.find_spec(name)
    if spec is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", name])
        spec = importlib.util.find_spec(name)

    return (spec is not None)
