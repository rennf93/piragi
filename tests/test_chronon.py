#!/usr/bin/env python3
"""
Test script for chronon.ai website loading.

This tests the original issue where loading chronon.ai caused memory exhaustion
and system crash on Mac due to the 8GB embedding model.

Expected behavior with v0.1.3+:
- Should load without crashing
- Should use ~420MB for embedding model (not 8GB)
- Should successfully query the loaded content
"""

import sys
import time
import traceback
from typing import Any


def get_memory_usage_mb() -> Any | None:
    """Get current process memory usage in MB."""
    try:
        import os

        import psutil

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return None


def test_chronon_loading() -> bool:
    """Test loading chronon.ai website."""
    from piragi import Ragi

    print("=" * 70)
    print("TEST: Loading chronon.ai documentation")
    print("=" * 70)

    mem_start = get_memory_usage_mb()
    if mem_start:
        print(f"Initial memory: {mem_start:.0f} MB")

    print("\n1. Loading chronon.ai...")
    start = time.time()

    try:
        kb = Ragi("https://chronon.ai", config={"auto_update": {"enabled": False}})
        elapsed = time.time() - start

        mem_after_load = get_memory_usage_mb()

        print(f"   ✓ Loaded successfully in {elapsed:.1f}s")
        print(f"   ✓ Chunks loaded: {kb.count()}")

        if mem_after_load and mem_start:
            mem_increase = mem_after_load - mem_start
            print(f"   ✓ Memory after load: {mem_after_load:.0f} MB")
            print(f"   ✓ Memory increase: {mem_increase:.0f} MB")

            if mem_increase > 2000:
                print(f"   ⚠ WARNING: Memory increase too high ({mem_increase:.0f} MB)")
                print("   ⚠ Expected ~500-800 MB with all-mpnet-base-v2 model")
                return False

    except Exception as e:
        print(f"   ✗ FAILED to load: {e}")
        traceback.print_exc()
        return False

    print("\n2. Testing queries...")

    test_queries = [
        "What is Chronon?",
        "How do I define features?",
        "What is a GroupBy?",
    ]

    for query in test_queries:
        try:
            answer = kb.ask(query)
            preview = answer.text[:100].replace("\n", " ")
            print(f"   ✓ Query: '{query}'")
            print(f"     Answer: {preview}...")
            print(f"     Citations: {len(answer.citations)}")

            # Basic sanity checks
            if len(answer.text) < 20:
                print("   ⚠ WARNING: Answer is suspiciously short")
                print(f"     Full answer: {answer.text}")

            if len(answer.citations) == 0:
                print("   ⚠ WARNING: No citations provided")

        except Exception as e:
            print(f"   ✗ Query failed: '{query}'")
            print(f"     Error: {e}")
            return False

    mem_final = get_memory_usage_mb()
    if mem_final and mem_start:
        print(f"\n3. Final memory: {mem_final:.0f} MB")
        print(f"   Total increase: {mem_final - mem_start:.0f} MB")

    print("\n" + "=" * 70)
    print("✓✓✓ ALL TESTS PASSED ✓✓✓")
    print("=" * 70)

    return True


def test_embedding_model() -> bool:
    """Verify the correct embedding model is being used."""
    from piragi import Ragi

    print("\n" + "=" * 70)
    print("TEST: Verify embedding model configuration")
    print("=" * 70)

    kb = Ragi(config={"auto_update": {"enabled": False}})

    # Check the model name
    model_name = kb.embedder.model_name
    print(f"\nEmbedding model in use: {model_name}")

    if model_name == "all-mpnet-base-v2":
        print("✓ Correct model (all-mpnet-base-v2, ~420MB)")
        return True
    elif model_name == "nvidia/llama-embed-nemotron-8b":
        print("✗ WRONG model (nvidia/llama-embed-nemotron-8b, ~8GB)")
        print("  This will cause memory issues!")
        return False
    else:
        print(f"? Unknown model: {model_name}")
        return True


if __name__ == "__main__":
    try:
        # Test 1: Embedding model check
        model_ok = test_embedding_model()

        # Test 2: Load chronon.ai
        chronon_ok = test_chronon_loading()

        if model_ok and chronon_ok:
            print("\n🎉 All tests passed!")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed!")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠ Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print("\n\n✗✗✗ UNEXPECTED ERROR ✗✗✗")
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)
