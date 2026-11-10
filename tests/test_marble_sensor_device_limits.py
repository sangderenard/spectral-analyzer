import pytest
from camera_designer.marble_preview_worker import _sensor_device_limits

def test_actual_texture_limit_and_known_minimum(monkeypatch):
 from OpenGL import GL
 for name,value in [('GL_MAX_TEXTURE_SIZE',3379),('GL_NUM_EXTENSIONS',33309),('GL_EXTENSIONS',7939)]:monkeypatch.setattr(GL,name,value,raising=False)
 monkeypatch.setattr(GL,'glGetIntegerv',lambda name:8192 if name==GL.GL_MAX_TEXTURE_SIZE else 0,raising=False)
 result=_sensor_device_limits(4096,4096)
 assert result['sensor_texture_bytes']==768*1024*1024
 assert result['available_gpu_memory_bytes'] is None
 with pytest.raises(ValueError,match='exceeds device'): _sensor_device_limits(16384,1)

def test_driver_memory_remains_advisory(monkeypatch):
 from OpenGL import GL
 for name,value in [('GL_MAX_TEXTURE_SIZE',3379),('GL_NUM_EXTENSIONS',33309),('GL_EXTENSIONS',7939)]:monkeypatch.setattr(GL,name,value,raising=False)
 monkeypatch.setattr(GL,'glGetIntegerv',lambda name:8192 if name==GL.GL_MAX_TEXTURE_SIZE else (1 if name==GL.GL_NUM_EXTENSIONS else 100),raising=False)
 monkeypatch.setattr(GL,'glGetStringi',lambda *args:b'GL_NVX_gpu_memory_info',raising=False)
 result=_sensor_device_limits(4096,4096)
 assert result['available_gpu_memory_bytes']==102400
 assert 'advisory' in result['memory_report']
