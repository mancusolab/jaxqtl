from importlib.metadata import PackageNotFoundError, version  # pragma: no cover


try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError


from jax import config


config.update("jax_enable_x64", True)

# Avoid eager subpackage imports here: importing `jaxqtl` should not pull in optional/heavy
# dependencies from `io/` or `map/` (e.g. scanpy/decoupler), and users can import what they need.
