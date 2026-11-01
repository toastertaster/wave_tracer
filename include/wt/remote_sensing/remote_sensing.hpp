/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

#pragma once

/**
 * @brief Realistic remote sensing: Earth-observation scenes, atmospheres and detector models.
 *        Compiled only with the CMake option REALISTIC_REMOTE_SENSING (cmake/wt_remote_sensing.cmake),
 *        which defines REALISTIC_REMOTE_SENSING.
 */
namespace wt::remote_sensing {

#if defined(REALISTIC_REMOTE_SENSING)
/**
 * @brief Returns TRUE: the library was built with the remote sensing features.
 *        Defined in src/remote_sensing/, so calling it also checks that those sources were compiled in.
 */
[[nodiscard]] bool enabled() noexcept;
#else
[[nodiscard]] constexpr bool enabled() noexcept { return false; }
#endif

}
