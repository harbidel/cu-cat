import importlib

class DepManager:
    def __init__(self):
        self.pkgs = {}

    def __getattr__(self, pkg:str):
        self._add_deps(pkg)
        try:
            return self.pkgs[pkg]
        except KeyError:
            return None

    def _add_deps(self, pkg:str):
        try:
            pkg_val = importlib.import_module(pkg)
            self.pkgs[pkg] = pkg_val
            setattr(self, pkg, pkg_val)
            if pkg is 'cuml' and cuml.__version__ < "24.02.00" and cuml.__version__ > "23.06.00":
                cuml.internals.base_return_types._process_generic = cuml.internals.base_helpers._process_generic
        except:
            pass

    def import_from(self,pkg:str, name:str):
        try:
            module = __import__(pkg, fromlist=[name])
            self.pkgs[name] = module
        except:
            pass


deps = DepManager()


