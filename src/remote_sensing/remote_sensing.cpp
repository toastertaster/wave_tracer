/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

#if !defined(REALISTIC_REMOTE_SENSING)
#error "src/remote_sensing/ is compiled only with REALISTIC_REMOTE_SENSING (cmake/wt_remote_sensing.cmake)"
#endif

#include <wt/remote_sensing/remote_sensing.hpp>

bool wt::remote_sensing::enabled() noexcept { return true; }
