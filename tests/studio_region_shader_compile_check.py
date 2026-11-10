"""Offline full region shader syntax/type gate; never creates a GPU context."""
import pathlib
import shutil
import subprocess
import tempfile
import demo_pluck_gl as renderer


def run():
    compiler=shutil.which('glslangValidator')
    if compiler is None:
        raise RuntimeError('Install an official Vulkan SDK glslangValidator for the offline gate.')
    for name,source in [('forward',renderer._GPU_RAY_FIELD_REGION_CS),('camera',renderer._GPU_SENSOR_REGION_CS)]:
        with tempfile.TemporaryDirectory(prefix='studio-region-glsl-') as d:
            path=pathlib.Path(d)/f'{name}.comp'
            path.write_text(source,encoding='utf-8')
            subprocess.run([compiler,'-S','comp',str(path)],check=True,timeout=60)
            print(f'PASS {name} full shader offline GLSL compilation',flush=True)

if __name__=='__main__':run()
