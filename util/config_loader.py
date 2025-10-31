import os, configparser

def load_config(env_section: str = None, path: str = "config.ini"):
    """Return (base_url) from config.ini."""
    cfg = configparser.ConfigParser()
    read = cfg.read(path)
    
    if not read:
        raise FileNotFoundError(f"Couldn't find {path} in the current directory.")
    section = env_section or os.getenv("API_ENV", "api")
    
    if section not in cfg:
        raise KeyError(f"Section [{section}] not found in {path}.")
    base_url = cfg[section].get("uri", "").rstrip("/")
    
    if not base_url:
        raise ValueError(f"[{section}] uri is empty in {path}.")

    return base_url