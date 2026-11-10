"""Bounded sensor-ray ring with optional linear eviction checkpoint; no decay."""
from dataclasses import dataclass

MAX_HISTORY_BYTES = 2 * 1024 * 1024 * 1024
HISTORY_MEMORY_RESERVE = 128 * 1024 * 1024
MAX_EXACT_HISTORY_SAMPLES = (1 << 53) - 1


def check_history_sample_count(completed_samples):
    if completed_samples >= MAX_EXACT_HISTORY_SAMPLES:
        raise OverflowError("ray history reached the exact double-precision sample-count limit; restart live rays")

@dataclass(frozen=True)
class SensorHistoryLayout:
    width: int
    height: int
    layers: int
    samples: int
    checkpoint: bool = False

    def __post_init__(self):
        values = (self.width, self.height, self.layers, self.samples)
        if any(type(v) is not int for v in values):
            raise ValueError("sensor history dimensions must be integers")
        if type(self.checkpoint) is not bool:
            raise ValueError("checkpoint must be a boolean")
        if not (1 <= self.width <= 2048 and 1 <= self.height <= 2048 and
                1 <= self.layers <= 8 and self.samples >= 1):
            raise ValueError("invalid sensor history dimensions or sample count")
        if self.nbytes > MAX_HISTORY_BYTES:
            raise ValueError("sensor ray history exceeds the 2 GiB GPU memory limit")

    @property
    def entries(self):
        return self.width * self.height * self.layers

    @property
    def nbytes(self):
        return self.entries * (4 * self.samples + self.state_bytes)

    @property
    def state_bytes(self):
        # Ring total, plus optional checkpoint sum, compensation and count.
        return 32 if self.checkpoint else 8

    @property
    def buffer_sizes(self):
        return self.entries * 4 * self.samples, self.entries * self.state_bytes


def check_history_device(layout, max_block_bytes, available_bytes=None, reclaim_bytes=0):
    """Check actual SSBO limits and optional driver free-memory estimate.

    Free memory is only an estimate; allocation must still be checked. The
    reserve avoids consuming the last reported device memory. Old ring storage
    will be released before replacement, so it can be counted once here.
    """
    if max_block_bytes <= 0 or max(layout.buffer_sizes) > max_block_bytes:
        raise ValueError("sensor history exceeds this GPU's per-buffer SSBO size limit; lower history or resolution")
    if available_bytes is not None and layout.nbytes > max(0, available_bytes + reclaim_bytes - HISTORY_MEMORY_RESERVE):
        raise MemoryError("not enough reported GPU memory for ray history plus 128 MiB headroom; lower history or resolution")


def replace_history_buffers(owner, layout, gl=None):
    """Validate and replace an accumulator's ring without a double-size peak.

    On a driver allocation failure, partial buffers are deleted and the owner
    records no history allocation. The live worker reports the error and exits;
    it never silently substitutes a shorter history or starts another job.
    """
    if gl is None:
        from OpenGL import GL as gl
    if layout:
        max_block = int(gl.glGetIntegerv(gl.GL_MAX_SHADER_STORAGE_BLOCK_SIZE))
        available = None
        # Standard GL has no portable available-VRAM query. Use the driver's
        # current estimate only when its documented extension is advertised.
        extensions = {gl.glGetStringi(gl.GL_EXTENSIONS, i)
                      for i in range(int(gl.glGetIntegerv(gl.GL_NUM_EXTENSIONS)))}
        if b"GL_NVX_gpu_memory_info" in extensions:
            available = max(0, int(gl.glGetIntegerv(0x9049))) * 1024
        check_history_device(layout, max_block, available, owner._history_bytes)
        owner._history_block_limit = max_block
    if owner._history_buffers:
        gl.glFinish()  # old storage must be reclaimable before a near-cap resize
        gl.glDeleteBuffers(len(owner._history_buffers), owner._history_buffers)
    owner._history_buffers = []
    owner._history_samples = owner._history_bytes = 0
    owner._history_checkpoint = False
    buffers = []
    try:
        for size in layout.buffer_sizes if layout else ():
            buffer = int(gl.glGenBuffers(1))
            buffers.append(buffer)
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, buffer)
            gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, size, None, gl.GL_DYNAMIC_DRAW)
            if int(gl.glGetBufferParameteriv(gl.GL_SHADER_STORAGE_BUFFER, gl.GL_BUFFER_SIZE)) != size:
                raise MemoryError("GPU did not allocate the requested history buffer")
    except Exception as error:
        if buffers:
            gl.glDeleteBuffers(len(buffers), buffers)
        raise MemoryError("GPU ray-history allocation failed; lower history or resolution") from error
    finally:
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, 0)
    owner._history_buffers = buffers
    owner._history_samples = layout.samples if layout else 0
    owner._history_bytes = layout.nbytes if layout else 0
    owner._history_checkpoint = layout.checkpoint if layout else False

# Each invocation owns one pixel. Row mode guarantees unique writers; random
# sprinkle mode can collide and is forbidden while history is enabled. The
# sensor still traces its real path and spectral response before this call.
# Samples are float32; the running total is float64 to retain small terms when
# evicting a bright path. Re-sum each full-window replacement to prevent an
# unbounded subtraction-error history. Optional checkpoints retain evicted
# samples separately as a compensated linear sum and count, never decay.
SENSOR_HISTORY_GLSL = r"""
layout(std430, binding = 6) buffer SensorRayHistory { float ray_history[]; };
layout(std430, binding = 7) buffer SensorRayTotals { double ray_totals[]; };
uniform int uHistorySamples;
uniform int uHistorySlot;
uniform int uHistoryCount;
uniform int uHistoryCheckpoint;
float sensor_history_value(int layer, ivec2 px, float value) {
    int pixel = px.y * uSensorSize.x + px.x;
    int address = (layer * uSensorSize.y * uSensorSize.x + pixel);
    int base = address * uHistorySamples;
    int slot = base + uHistorySlot;
    int state = address * (uHistoryCheckpoint != 0 ? 4 : 1);
    double total = uHistoryCount == 0 ? 0.0 : ray_totals[state];
    if (uHistoryCheckpoint != 0 && uHistoryCount == 0) {
        ray_totals[state+1] = 0.0;
        ray_totals[state+2] = 0.0;
        ray_totals[state+3] = 0.0;
    }
    if (uHistoryCount == uHistorySamples) {
        double expired = double(ray_history[slot]);
        total -= expired;
        if (uHistoryCheckpoint != 0) {
            // Flatten each evicted linear sensor sample exactly once. Keep a
            // compensated sum and a weight, never a profiled/tonemapped image.
            double sum = ray_totals[state+1];
            double next = sum + expired;
            ray_totals[state+2] += abs(sum) >= abs(expired)
                ? (sum-next)+expired : (expired-next)+sum;
            ray_totals[state+1] = next;
            ray_totals[state+3] += 1.0;
        }
    }
    ray_history[slot] = value;
    total += double(value);
    int count = min(uHistoryCount + 1, uHistorySamples);
    // A complete replacement cycle is a natural exact-state rebase. Also
    // rebase after every full-window eviction to eliminate cancellation ghosts
    // when the last bright ray leaves an otherwise dark window. Large histories
    // cost more bandwidth; preserve exact finite-state behavior rather than decay.
    if (uHistoryCount == uHistorySamples) {
        total = 0.0;
        for (int s = 0; s < uHistorySamples; ++s) total += double(ray_history[base+s]);
    }
    ray_totals[state] = total;
    if (uHistoryCheckpoint != 0) {
        return float(((ray_totals[state+1]+ray_totals[state+2])+total)
                     / (ray_totals[state+3]+double(count)));
    }
    return float(total / double(count));
}
"""
