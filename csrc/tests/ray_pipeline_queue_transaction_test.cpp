#include "ray_pipeline.h"

#include <cassert>
#include <vector>

int main() {
    PipelineQueue<int> queue;
    assert(queue.push(2));
    assert(queue.push_front(1));
    const auto checkpoint = queue.copy_transaction_state();

    std::vector<int> drained;
    assert(queue.drain(drained, 8) == 2);
    assert((drained == std::vector<int>{1, 2}));
    assert(queue.push(9));

    queue.restore_transaction_state(checkpoint);
    drained.clear();
    assert(queue.drain(drained, 8) == 2);
    assert((drained == std::vector<int>{1, 2}));

    // A checkpoint remains reusable across more than one rejected attempt.
    assert(queue.push(10));
    queue.restore_transaction_state(checkpoint);
    drained.clear();
    assert(queue.drain(drained, 8) == 2);
    assert((drained == std::vector<int>{1, 2}));

    queue.set_done();
    const auto done_checkpoint = queue.copy_transaction_state();
    PipelineQueue<int> restored_done;
    restored_done.restore_transaction_state(done_checkpoint);
    assert(restored_done.is_done());
    return 0;
}
