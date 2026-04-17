package com.vudance.recommendation_system.service;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;

import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
public class RedisService {
    private final RedisTemplate<Object, Object> redisTemplate;

    public void set(String key, Object value, long timeoutSeconds) {
        redisTemplate.opsForValue().set(key, value);
        redisTemplate.expire(key, timeoutSeconds, java.util.concurrent.TimeUnit.SECONDS);
    }
    public Object get(String key) {
        return redisTemplate.opsForValue().get(key);
    }
    public void delete(String key) {
        redisTemplate.delete(key);
    }
    public Map<Object, Object> getHash(String key) {
        return redisTemplate.opsForHash().entries(key);
    }

    public List<String> getSortedHashKeys(String key) {
        Map<Object, Object> hash = getHash(key);
        if (hash == null || hash.isEmpty()) return Collections.emptyList();

        return hash.entrySet().stream()
            .sorted((a, b) -> Double.compare(
                Double.parseDouble(b.getValue().toString()),
                Double.parseDouble(a.getValue().toString())
            ))
            .map(e -> e.getKey().toString())
            .collect(Collectors.toList());
    }
}
