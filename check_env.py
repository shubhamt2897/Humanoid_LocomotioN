"""Diagnostic script: check RL environment dependencies (PyTorch, MuJoCo, etc.)."""

def check(name, import_name=None):
    import_name = import_name or name
    try:
        module = __import__(import_name)
        version = getattr(module, "__version__", "unknown version")
        print(f"[OK]      {name:<10} -> {version}")
    except ImportError:
        print(f"[MISSING] {name:<10} -> not installed")


if __name__ == "__main__":
    print("Checking RL environment dependencies...\n")
    check("torch")
    check("mujoco")
    check("rsl_rl")
    check("wandb")
    check("numpy")
