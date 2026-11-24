"""Domain-scoped resource resolution, independent of simulator initialization."""

from pathlib import Path
from urllib.parse import urlsplit

from roboclock.envs.registry import get_environment

PACKAGE_ROOT = Path(__file__).parent
SHARED_ASSET_ROOT = PACKAGE_ROOT / "assets/shared"


def resolve_asset(reference, *, environment="lab", base_dir=None) -> Path:
    """Resolve shared://, env://, <domain>://, absolute or local relative paths.

    Plain catalog-relative paths use base_dir when supplied. The legacy lab
    assets/hdr and assets/textures spellings remain readable in saved configs.
    """
    reference = str(reference)
    scheme, separator, location = reference.partition("://")
    if separator:
        relative = Path(location)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Asset URI must stay inside its namespace: {reference}")
        if scheme == "shared":
            return SHARED_ASSET_ROOT / relative
        domain = environment if scheme == "env" else scheme
        return get_environment(domain).asset_root / relative
    path = Path(reference)
    if path.is_absolute():
        return path
    if path.parts[:2] == ("assets", "hdr"):
        return SHARED_ASSET_ROOT / Path(*path.parts[1:])
    if path.parts[:2] in (("assets", "textures"), ("assets", "catalog.yaml")):
        return get_environment(environment).asset_root / Path(*path.parts[1:])
    return Path(base_dir or get_environment(environment).asset_root) / path


def resolve_asset_reference(reference, *, environment="lab", base_dir=None) -> str:
    """Preserve remote USD URLs; resolve local resources to absolute filenames."""
    if urlsplit(str(reference)).scheme in ("omniverse", "http", "https", "file"):
        return str(reference)
    return str(resolve_asset(reference, environment=environment, base_dir=base_dir).resolve())
