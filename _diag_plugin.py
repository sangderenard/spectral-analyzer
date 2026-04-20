import sys, os, importlib.util, traceback
sys.path.insert(0, r'c:\Users\alber\Downloads\spectral-analyzer')

plugin_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'sm_plugins', 'orchestral_resonance.py')
print("plugin_path:", plugin_path, "exists:", os.path.isfile(plugin_path))

spec = importlib.util.spec_from_file_location("sm_plugin_orchestral_resonance", plugin_path)
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
    print("LOADED OK")
except Exception:
    traceback.print_exc()
