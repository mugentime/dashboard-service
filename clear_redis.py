#!/usr/bin/env python3
"""Clear stale bot data from Redis"""
import os
import redis

REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')

try:
    client = redis.from_url(REDIS_URL, decode_responses=True)

    # Get all bot keys
    bot_keys = client.keys('bot:*')

    if bot_keys:
        print(f"Found {len(bot_keys)} bot keys in Redis:")
        for key in bot_keys:
            print(f"  - {key}")

        # Delete all bot keys
        deleted = client.delete(*bot_keys)
        print(f"\n✅ Deleted {deleted} keys from Redis")
    else:
        print("No bot keys found in Redis")

except Exception as e:
    print(f"❌ Error: {e}")
