#include <criterion/criterion.h>
#include <criterion/theories.h>
#include "profiling.h"


TheoryDataPoints(profiling, startAndEndRange) = {
    DataPoints(char const *, "", "mknaJHAnlkml", " ")
};

Theory((char const * label), profiling, startAndEndRange) {
    RangeHandleType handle = startProfilingRange(label);
    endProfilingRange(handle);
    // just test that nothing crashes here
    cr_assert(true);
}
