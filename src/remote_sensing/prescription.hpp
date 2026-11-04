/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

#pragma once

// Readers of a telescope's prescription (src/remote_sensing/ only): its numbers in double precision, as the scene
// loader's quantities are f_t.

#include <cmath>
#include <cstdlib>
#include <format>
#include <optional>
#include <string>

#include <wt/math/common.hpp>
#include <wt/scene/loader/node.hpp>
#include <wt/util/format/parse.hpp>

namespace wt::remote_sensing::prescription {

/**
 * @brief A length, in metres: a number and a unit (m, km, cm, mm, um or nm).
 */
inline double parse_length(const std::string& str) {
    // (strtod, not from_chars: older libc++ lacks floating-point from_chars)
    const char* first = str.c_str();
    char* end = nullptr;
    const double value = std::strtod(first, &end);
    if (end==first || !std::isfinite(value))
        throw std::format_error("malformed length \"" + str + "\"");
    auto unit = std::string{ end };
    while (!unit.empty() && unit.front()==' ') unit.erase(unit.begin());
    while (!unit.empty() && unit.back()==' ') unit.pop_back();
    if (unit=="m")                  return value;
    if (unit=="km")                 return value*1e3;
    if (unit=="cm")                 return value*1e-2;
    if (unit=="mm")                 return value*1e-3;
    if (unit=="um" || unit=="µm")  return value*1e-6;
    if (unit=="nm")                 return value*1e-9;
    throw std::format_error("length \"" + str + "\" needs a unit (m, km, cm, mm, um or nm)");
}

/**
 * @brief Reads `<quantity name="name" value="..."/>` as a length (`parse_length`).
 */
inline bool read_length(const scene::loader::node_t& item, const std::string& name, std::optional<double>& out) {
    if (item.name()!="quantity" || item["name"]!=name)
        return false;
    if (out)
        throw std::format_error("Node \"" + name + "\" already provided");
    out = parse_length(item["value"]);
    return true;
}

/**
 * @brief Reads `<float name="name" value="..."/>`.
 */
inline bool read_number(const scene::loader::node_t& item, const std::string& name, std::optional<double>& out) {
    if (item.name()!="float" || item["name"]!=name)
        return false;
    if (out)
        throw std::format_error("Node \"" + name + "\" already provided");
    out = stofd_strict(item["value"]);
    return true;
}

/**
 * @brief Reads `<vector name="name" value="x, y"/>` as two lengths.
 */
inline bool read_length_vec2(const scene::loader::node_t& item, const std::string& name, std::optional<glm::dvec2>& out) {
    if (item.name()!="vector" || item["name"]!=name)
        return false;
    if (out)
        throw std::format_error("Node \"" + name + "\" already provided");
    const auto& v = item["value"];
    const auto comma = v.find(',');
    if (comma==std::string::npos || v.find(',', comma+1)!=std::string::npos)
        throw std::format_error("\"" + name + "\" needs two lengths, e.g. \"0m, 0.3m\"");
    out = glm::dvec2{ parse_length(v.substr(0, comma)), parse_length(v.substr(comma+1)) };
    return true;
}

}
