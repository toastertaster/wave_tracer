/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

#pragma once

#include <cassert>
#include <wt/math/common.hpp>

namespace wt {

/**
 * @brief Encodes a 3d normal vector in a 64-bit structure.
 */
class encoded_normal_t {
private:
    // Octahedral encoding, stored in single precision regardless of f_t so
    // the structure stays 64 bits.
    glm::vec<2, float> enc{};

private:
    static constexpr inline vec2_t vec2_oct_wrap(vec2_t v) noexcept {
        return (vec2_t{ 1 } - m::abs(vec2_t{ v.y,v.x }) ) * 
            (m::mix(vec2_t{ -1 }, vec2_t{ 1 }, v>=vec2_t{ 0 }));
    }

public:
    encoded_normal_t() noexcept = default;
    explicit encoded_normal_t(dir3_t normal) noexcept {
        vec3_t n = vec3_t{ normal } / (m::abs(normal.x) + m::abs(normal.y) + m::abs(normal.z));
        auto xy = n.z>=0 ? vec2_t{ n } : vec2_oct_wrap(vec2_t{ n });
        xy = xy*f_t(.5) + f_t(.5);

        assert(m::isfinite(xy));

        enc = glm::vec<2, float>{ xy };
    }

    encoded_normal_t(encoded_normal_t&&) noexcept = default;
    encoded_normal_t(const encoded_normal_t&) noexcept = default;
    encoded_normal_t& operator=(encoded_normal_t&&) noexcept = default;
    encoded_normal_t& operator=(const encoded_normal_t&) noexcept = default;

    encoded_normal_t& operator=(const dir3_t& n) noexcept {
        *this = encoded_normal_t{ n };
        return *this;
    }

    explicit operator dir3_t() const noexcept {
        auto enc = vec2_t{ this->enc };
        enc = enc*f_t(2) - vec2_t{ 1 };

        const auto z = 1 - m::abs(enc.x) - m::abs(enc.y);
        vec3_t n = {
            z>=0 ? enc : vec2_oct_wrap(enc),
            z
        };
        return m::normalize(n);
    }
};

static_assert(sizeof(encoded_normal_t) == 8);

}
