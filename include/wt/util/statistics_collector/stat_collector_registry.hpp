/*
 *
 * wave tracer
 * Copyright  Shlomi Steinberg
 * Authors:  Umut Emre, Shlomi Steinberg
 *
 * LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
 *
 */

#pragma once

#include <atomic>
#include <cstdint>

#include <tbb/tbb.h>

#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include "stat_collector.hpp"

namespace wt::stats {

/**
 * @brief Identifies the calling thread, uniquely over the lifetime of the process.
 *        Unlike std::thread::id, which the OS may reuse once a thread exits, a serial is never reused, so a
 *        new thread never collides with the collectors of an exited one.
 */
inline std::uint64_t this_thread_serial() noexcept {
    static std::atomic<std::uint64_t> next_serial{ 0 };
    thread_local const std::uint64_t serial = next_serial.fetch_add(1, std::memory_order_relaxed);
    return serial;
}

using collector_map_key_t = std::pair<std::string, std::uint64_t>;

struct collector_map_key_hash {
    static bool equal(const collector_map_key_t& x, const collector_map_key_t& y) {
        return x == y;
    }

    static std::size_t hash(const collector_map_key_t& p) {
        return (std::hash<std::string>{}(p.first) ^ std::hash<std::uint64_t>{}(p.second));
    }
};

/**
 * @brief Singleton to register all stat collectors
 */
class stat_collector_registry_t {
    tbb::concurrent_hash_map<collector_map_key_t, std::unique_ptr<stat_collector_t>, collector_map_key_hash> collectors;

public:
    stat_collector_registry_t() = default;
    stat_collector_registry_t(stat_collector_registry_t& other) = delete;
    void operator=(const stat_collector_registry_t&) = delete;

    static stat_collector_registry_t& instance() noexcept {
        static stat_collector_registry_t instance;
        return instance;
    }

    /**
     * @brief Instantiates a new collector of type T.
     *        (thread safe)
     */
    template <typename T>
        requires std::is_base_of_v<stat_collector_t, T>
    T* make_collector(auto&&... args) {
        auto collector = std::make_unique<T>(std::forward<decltype(args)>(args)...);
        auto result = collector.get();
        auto key = std::make_pair(collector->name, this_thread_serial());

        if (!collectors.insert({key, std::move(collector)})) {
            throw std::runtime_error(std::format("stat collector with name '{}' registered more than once", key.first));
        }

        return result;
    }

    /**
     * @brief Get a copy of all stat_collector_t to view, aggregating counters that are instantiated in multiple threads.
     *        This getter is thread safe, however data to collectors may be written during access.
     */
    [[nodiscard]] std::vector<std::unique_ptr<stat_collector_t>> get_collectors() const;
};

}  // namespace wt
