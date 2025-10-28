#!/usr/bin/env python3
"""
STORM CLI Test Suite
Unit + Integration + Smoke Tests
"""

import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path

# Test counters
tests_run = 0
tests_passed = 0
tests_failed = 0

def test(name):
    """Decorator for test functions"""
    def decorator(func):
        def wrapper():
            global tests_run, tests_passed, tests_failed
            tests_run += 1
            print(f"\n{'='*80}")
            print(f"TEST {tests_run}: {name}")
            print('='*80)
            try:
                func()
                tests_passed += 1
                print(f"✅ PASSED: {name}")
                return True
            except AssertionError as e:
                tests_failed += 1
                print(f"❌ FAILED: {name}")
                print(f"   Error: {e}")
                return False
            except Exception as e:
                tests_failed += 1
                print(f"❌ ERROR: {name}")
                print(f"   Exception: {e}")
                return False
        return wrapper
    return decorator

# ============================================================================
# UNIT TESTS
# ============================================================================

@test("Unit: Check Python 3 available")
def test_python_version():
    result = subprocess.run(['python3', '--version'], capture_output=True, text=True)
    assert result.returncode == 0, "Python 3 not found"
    print(f"   Python version: {result.stdout.strip()}")

@test("Unit: Check storm file exists and is executable")
def test_storm_exists():
    storm_path = Path('storm')
    assert storm_path.exists(), "storm file not found"
    assert os.access(storm_path, os.X_OK), "storm is not executable"
    print(f"   storm file: {storm_path.absolute()}")

@test("Unit: Check dependencies installed (dspy-ai)")
def test_dspy_installed():
    result = subprocess.run(['python3', '-c', 'import dspy'], capture_output=True)
    assert result.returncode == 0, "dspy-ai not installed"
    print("   dspy-ai is installed")

@test("Unit: Check dependencies installed (pydantic)")
def test_pydantic_installed():
    result = subprocess.run(['python3', '-c', 'import pydantic'], capture_output=True)
    assert result.returncode == 0, "pydantic not installed"
    print("   pydantic is installed")

@test("Unit: Check OPENROUTER_API_KEY is set")
def test_api_key_set():
    api_key = os.environ.get('OPENROUTER_API_KEY')
    assert api_key is not None, "OPENROUTER_API_KEY not set"
    assert len(api_key) > 10, "OPENROUTER_API_KEY seems invalid (too short)"
    print(f"   API key: {api_key[:10]}...{api_key[-4:]}")

@test("Unit: CLI --help works")
def test_help_command():
    result = subprocess.run(['python3', 'storm', '--help'], capture_output=True, text=True)
    assert result.returncode == 0, f"--help failed with code {result.returncode}"
    assert 'STORM CLI' in result.stdout, "Help text doesn't contain 'STORM CLI'"
    assert 'usage:' in result.stdout.lower(), "Help text missing usage"
    print("   Help command works")

@test("Unit: CLI --version works")
def test_version_command():
    result = subprocess.run(['python3', 'storm', '--version'], capture_output=True, text=True)
    assert result.returncode == 0, f"--version failed with code {result.returncode}"
    assert '1.0.0' in result.stdout or '1.0.0' in result.stderr, "Version not found"
    print(f"   Version: {result.stderr.strip() if result.stderr else result.stdout.strip()}")

@test("Unit: CLI rejects invalid arguments")
def test_invalid_args():
    result = subprocess.run(['python3', 'storm', '--invalid-arg'], capture_output=True, text=True)
    assert result.returncode != 0, "Should fail with invalid argument"
    print("   Correctly rejects invalid arguments")

# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@test("Integration: Generate short article (200 words)")
def test_short_article():
    result = subprocess.run(
        ['python3', 'storm', 'Test Topic Short', '--words', '200'],
        capture_output=True,
        text=True,
        timeout=120
    )
    assert result.returncode == 0, f"Generation failed with code {result.returncode}"
    assert 'ARTICLE' in result.stdout, "Output missing article section"

    # Check word count is reasonable (150-250 words for 200 target)
    article_start = result.stdout.find('ARTICLE')
    if article_start > 0:
        article_text = result.stdout[article_start:]
        word_count = len(article_text.split())
        print(f"   Generated ~{word_count} words (target: 200)")
        assert word_count >= 150, f"Article too short: {word_count} words"

@test("Integration: Generate medium article (500 words)")
def test_medium_article():
    result = subprocess.run(
        ['python3', 'storm', 'Artificial Intelligence Basics', '--words', '500'],
        capture_output=True,
        text=True,
        timeout=180
    )
    assert result.returncode == 0, f"Generation failed with code {result.returncode}"
    # Check for completion message (with emoji prefix)
    output = result.stdout + result.stderr
    assert '📝 Complete:' in output or 'Complete:' in output, "Output missing completion message"
    assert '📊 RESULTS' in result.stdout, "Output missing results section"
    print("   500-word article generated successfully")

@test("Integration: Save to text file")
def test_save_txt():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        output_file = f.name

    try:
        result = subprocess.run(
            ['python3', 'storm', 'Test Save TXT', '--words', '200', '-o', output_file, '-f', 'txt'],
            capture_output=True,
            text=True,
            timeout=120
        )
        assert result.returncode == 0, f"Save to TXT failed with code {result.returncode}"

        # Check file was created and has content
        assert Path(output_file).exists(), "Output file not created"
        content = Path(output_file).read_text()
        assert len(content) > 100, "Output file seems empty or too short"
        print(f"   Created {output_file} with {len(content)} characters")
    finally:
        Path(output_file).unlink(missing_ok=True)

@test("Integration: Save to markdown file")
def test_save_markdown():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        output_file = f.name

    try:
        result = subprocess.run(
            ['python3', 'storm', 'Test Save MD', '--words', '200', '-o', output_file, '-f', 'md'],
            capture_output=True,
            text=True,
            timeout=120
        )
        assert result.returncode == 0, f"Save to MD failed with code {result.returncode}"

        # Check file structure
        content = Path(output_file).read_text()
        assert '# Test Save MD' in content, "Markdown missing title"
        assert '**Word Count:**' in content, "Markdown missing metadata"
        assert '---' in content, "Markdown missing separator"
        print(f"   Created valid markdown file: {output_file}")
    finally:
        Path(output_file).unlink(missing_ok=True)

@test("Integration: JSON output")
def test_json_output():
    result = subprocess.run(
        ['python3', 'storm', 'Test JSON', '--words', '200', '--json-only'],
        capture_output=True,
        text=True,
        timeout=120
    )
    assert result.returncode == 0, f"JSON generation failed with code {result.returncode}"

    # Parse JSON
    try:
        data = json.loads(result.stdout)
        assert 'article' in data, "JSON missing 'article' field"
        assert 'metadata' in data, "JSON missing 'metadata' field"
        assert 'research' in data, "JSON missing 'research' field"
        assert 'word_count' in data['metadata'], "JSON metadata missing word_count"
        print(f"   Valid JSON with {data['metadata']['word_count']} words")
    except json.JSONDecodeError as e:
        raise AssertionError(f"Invalid JSON output: {e}")

@test("Integration: Verbose mode")
def test_verbose_mode():
    result = subprocess.run(
        ['python3', 'storm', 'Test Verbose', '--words', '200', '-v'],
        capture_output=True,
        text=True,
        timeout=120
    )
    assert result.returncode == 0, f"Verbose mode failed with code {result.returncode}"
    # Should have more detailed logging (DEBUG logs go to stderr)
    output = result.stdout + result.stderr
    assert 'DEBUG' in output or '✓ Section' in output, "Verbose mode not producing extra output"
    print("   Verbose logging working")

# ============================================================================
# SMOKE TESTS (Real-world scenarios)
# ============================================================================

@test("Smoke: Generate article on 'Quantum Computing' (800 words)")
def test_smoke_quantum():
    result = subprocess.run(
        ['python3', 'storm', 'Quantum Computing', '--words', '800'],
        capture_output=True,
        text=True,
        timeout=180
    )
    assert result.returncode == 0, "Quantum Computing article failed"
    assert 'quantum' in result.stdout.lower(), "Article doesn't mention 'quantum'"
    output = result.stdout + result.stderr
    assert '📝 Complete:' in output or 'Complete:' in output, "Generation didn't complete"
    print("   Quantum Computing article generated")

@test("Smoke: Check for repetition (no duplicate paragraphs)")
def test_smoke_no_repetition():
    result = subprocess.run(
        ['python3', 'storm', 'Machine Learning Test', '--words', '600'],
        capture_output=True,
        text=True,
        timeout=180
    )
    assert result.returncode == 0, "Generation failed"

    # Extract article text - look for "📄 ARTICLE" section
    lines = result.stdout.split('\n')
    article_lines = []
    in_article = False
    for i, line in enumerate(lines):
        # Match the actual output format: "📄 ARTICLE" on its own line
        if '📄 ARTICLE' in line or (line.strip() == 'ARTICLE' and i > 0 and '='*40 in lines[i-1]):
            in_article = True
            continue
        # Skip the separator line right after the marker
        if in_article and '='*40 in line and len(article_lines) == 0:
            continue
        # Stop at the next major separator or tip line
        if in_article and (('='*40 in line and len(article_lines) > 5) or '💡 Tip:' in line):
            break
        if in_article and line.strip():
            article_lines.append(line.strip())

    # Check for duplicate lines (allowing for common words like "the", "a", etc.)
    long_lines = [l for l in article_lines if len(l) > 50]
    unique_lines = set(long_lines)

    # Should not have significant repetition
    if len(long_lines) == 0:
        # If no long lines found, check if we have any article content at all
        assert len(article_lines) > 10, f"Failed to extract article (only {len(article_lines)} lines found)"
        print(f"   Article extracted: {len(article_lines)} lines (no long lines > 50 chars to check)")
    else:
        repetition_ratio = len(unique_lines) / len(long_lines)
        print(f"   Unique lines: {len(unique_lines)}/{len(long_lines)} ({repetition_ratio:.1%})")
        assert repetition_ratio > 0.85, f"Too much repetition detected: {repetition_ratio:.1%} unique"

@test("Smoke: Word count accuracy (target 500, expect 450-550)")
def test_smoke_word_count_accuracy():
    result = subprocess.run(
        ['python3', 'storm', 'Word Count Test', '--words', '500', '--json-only'],
        capture_output=True,
        text=True,
        timeout=180
    )
    assert result.returncode == 0, "Generation failed"

    data = json.loads(result.stdout)
    actual_words = data['metadata']['word_count']
    target_words = data['metadata']['target_words']

    accuracy = (actual_words / target_words) * 100
    print(f"   Target: {target_words}, Actual: {actual_words} ({accuracy:.1f}%)")

    # Should be within 90-110% of target
    assert 0.90 <= accuracy/100 <= 1.10, f"Word count too far from target: {accuracy:.1f}%"

@test("Smoke: Multiple formats work correctly")
def test_smoke_all_formats():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Generate all three formats
        for fmt in ['txt', 'md', 'json']:
            output_file = tmpdir / f"test.{fmt}"
            result = subprocess.run(
                ['python3', 'storm', 'Format Test', '--words', '200',
                 '-o', str(output_file), '-f', fmt],
                capture_output=True,
                text=True,
                timeout=120
            )
            assert result.returncode == 0, f"Failed to generate {fmt} format"
            assert output_file.exists(), f"{fmt} file not created"
            assert output_file.stat().st_size > 100, f"{fmt} file too small"
            print(f"   ✓ {fmt.upper()} format: {output_file.stat().st_size} bytes")

@test("Smoke: Handles special characters in topic")
def test_smoke_special_chars():
    result = subprocess.run(
        ['python3', 'storm', 'AI & ML: The Future', '--words', '200'],
        capture_output=True,
        text=True,
        timeout=120
    )
    assert result.returncode == 0, "Failed with special characters in topic"
    print("   Handles special characters correctly")

@test("Smoke: Fast execution (200 words in under 60 seconds)")
def test_smoke_performance():
    import time
    start = time.time()

    result = subprocess.run(
        ['python3', 'storm', 'Performance Test', '--words', '200'],
        capture_output=True,
        text=True,
        timeout=120
    )

    duration = time.time() - start
    assert result.returncode == 0, "Generation failed"
    assert duration < 60, f"Too slow: {duration:.1f}s (expected < 60s)"
    print(f"   Generated 200 words in {duration:.1f}s")

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def main():
    print("\n" + "="*80)
    print("🌪️  STORM CLI - COMPREHENSIVE TEST SUITE")
    print("="*80)
    print("\nRunning Unit Tests, Integration Tests, and Smoke Tests...\n")

    # Run all tests
    test_python_version()
    test_storm_exists()
    test_dspy_installed()
    test_pydantic_installed()
    test_api_key_set()
    test_help_command()
    test_version_command()
    test_invalid_args()

    test_short_article()
    test_medium_article()
    test_save_txt()
    test_save_markdown()
    test_json_output()
    test_verbose_mode()

    test_smoke_quantum()
    test_smoke_no_repetition()
    test_smoke_word_count_accuracy()
    test_smoke_all_formats()
    test_smoke_special_chars()
    test_smoke_performance()

    # Print results
    print("\n" + "="*80)
    print("📊 TEST RESULTS")
    print("="*80)
    print(f"Total Tests: {tests_run}")
    print(f"✅ Passed: {tests_passed}")
    print(f"❌ Failed: {tests_failed}")
    print(f"Success Rate: {(tests_passed/tests_run*100):.1f}%")
    print("="*80)

    if tests_failed == 0:
        print("\n🎉 ALL TESTS PASSED! 100% SUCCESS RATE")
        return 0
    else:
        print(f"\n⚠️  {tests_failed} TEST(S) FAILED")
        return 1

if __name__ == '__main__':
    sys.exit(main())
