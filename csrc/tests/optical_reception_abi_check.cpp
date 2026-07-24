#include "optical_reception_abi.h"

#include <cstddef>
#include <cstdint>

int main() {
    using optical_reception::OpticalReceptionToken;
    OpticalReceptionToken token{};
    token.reception_key_handle = 0x0102030405060708ull;
    token.pool_id = 9u;
    token.flags =
        optical_reception::TokenActive |
        optical_reception::TokenCompletion |
        optical_reception::TokenCoherent;
    return (
        sizeof(token) == 96u &&
        offsetof(OpticalReceptionToken, pool_id) == 56u &&
        offsetof(OpticalReceptionToken, flags) == 80u &&
        token.reception_key_handle == 0x0102030405060708ull &&
        token.pool_id == 9u
    ) ? 0 : 1;
}
