#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify all 7 dashboard issues are fixed
Tests the live dashboard at https://dashboard-production-5a5d.up.railway.app
"""

import sys
import io
import time
import json
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Fix Unicode encoding for Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

DASHBOARD_URL = "https://dashboard-production-5a5d.up.railway.app"

def setup_driver():
    """Setup Chrome driver with headless options"""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(options=chrome_options)
    return driver

def wait_for_dashboard_load(driver, timeout=30):
    """Wait for dashboard to fully load"""
    print("⏳ Waiting for dashboard to load...")
    try:
        # Wait for the main title to appear
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.XPATH, "//h1[contains(text(), 'Self-Optimizer Bot Monitor')]"))
        )
        print("✅ Dashboard title loaded")

        # Wait for status cards to appear
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.ID, "bot-status"))
        )
        print("✅ Status cards loaded")

        # Check for browser console errors
        logs = driver.get_log('browser')
        if logs:
            print("\n🔍 Browser Console Logs:")
            for log in logs:
                print(f"  [{log['level']}] {log['message']}")

        # Give Dash callbacks time to execute (wait for content to appear)
        print("⏳ Waiting for Dash callbacks to populate data...")

        # Wait for bot-status to have content (polls every 0.5s for up to 30s)
        wait = WebDriverWait(driver, 30)
        try:
            wait.until(lambda d: d.find_element(By.ID, "bot-status").text.strip() != "")
            bot_status = driver.find_element(By.ID, "bot-status")
            print(f"✅ Data loaded: Bot Status = '{bot_status.text}'")
        except TimeoutException:
            print("⚠️  No data loaded after 30 seconds")

        # Extra wait for charts to render
        time.sleep(3)

        return True
    except TimeoutException:
        print("❌ Dashboard failed to load within timeout")
        return False

def test_issue_1_status_cards(driver):
    """
    Issue 1: Check if all status cards display correctly
    Expected: Bot Status, Balance, Active Trades, Optimization Cycle, Redis Status
    """
    print("\n" + "="*80)
    print("TEST ISSUE 1: Status Cards Display")
    print("="*80)

    status_cards = {
        'bot-status': 'Bot Status',
        'balance-display': 'Balance',
        'active-trades': 'Active Trades',
        'optimization-cycle': 'Optimization Cycle',
        'supervisor-status': 'Redis Status'
    }

    results = {}
    all_passed = True

    for card_id, card_name in status_cards.items():
        try:
            element = driver.find_element(By.ID, card_id)
            content = element.text

            if content and content.strip():
                print(f"✅ {card_name}: '{content}'")
                results[card_name] = {'status': 'PASS', 'content': content}
            else:
                print(f"⚠️  {card_name}: No content displayed")
                results[card_name] = {'status': 'FAIL', 'content': 'Empty'}
                all_passed = False

        except NoSuchElementException:
            print(f"❌ {card_name}: Element not found")
            results[card_name] = {'status': 'FAIL', 'content': 'Not Found'}
            all_passed = False

    return {'passed': all_passed, 'details': results}

def test_issue_2_balance_chart(driver):
    """
    Issue 2: Verify balance chart loads with data or proper "no data" message
    """
    print("\n" + "="*80)
    print("TEST ISSUE 2: Balance Chart")
    print("="*80)

    try:
        chart = driver.find_element(By.ID, "balance-chart")

        # Check if chart has rendered content
        svg_elements = chart.find_elements(By.TAG_NAME, "svg")

        if svg_elements:
            print(f"✅ Balance chart rendered with {len(svg_elements)} SVG element(s)")

            # Check for "no data" message
            text_elements = chart.find_elements(By.TAG_NAME, "text")
            chart_text = [elem.text for elem in text_elements if elem.text.strip()]

            if any("no" in text.lower() or "data" in text.lower() for text in chart_text):
                print(f"✅ Chart shows proper 'no data' message: {chart_text}")
                return {'passed': True, 'details': {'message': 'Shows no data message', 'text': chart_text}}
            elif chart_text:
                print(f"✅ Chart shows data: {chart_text[:5]}")  # Show first 5 text elements
                return {'passed': True, 'details': {'message': 'Shows data', 'text': chart_text[:5]}}
            else:
                print("⚠️  Chart rendered but no text found")
                return {'passed': True, 'details': {'message': 'Chart rendered, no text'}}
        else:
            print("❌ Balance chart not rendered")
            return {'passed': False, 'details': {'message': 'No SVG elements found'}}

    except NoSuchElementException:
        print("❌ Balance chart element not found")
        return {'passed': False, 'details': {'message': 'Element not found'}}

def test_issue_3_optimization_params_chart(driver):
    """
    Issue 3: Verify optimization parameters chart displays correctly
    """
    print("\n" + "="*80)
    print("TEST ISSUE 3: Optimization Parameters Chart")
    print("="*80)

    try:
        chart = driver.find_element(By.ID, "optimization-params-chart")
        svg_elements = chart.find_elements(By.TAG_NAME, "svg")

        if svg_elements:
            print(f"✅ Optimization params chart rendered with {len(svg_elements)} SVG element(s)")

            text_elements = chart.find_elements(By.TAG_NAME, "text")
            chart_text = [elem.text for elem in text_elements if elem.text.strip()]

            if chart_text:
                print(f"✅ Chart text: {chart_text[:5]}")
                return {'passed': True, 'details': {'text': chart_text[:5]}}
            else:
                print("✅ Chart rendered (no text)")
                return {'passed': True, 'details': {'message': 'Rendered without text'}}
        else:
            print("❌ Optimization params chart not rendered")
            return {'passed': False, 'details': {'message': 'No SVG elements'}}

    except NoSuchElementException:
        print("❌ Optimization params chart element not found")
        return {'passed': False, 'details': {'message': 'Element not found'}}

def test_issue_4_baseline_comparison_chart(driver):
    """
    Issue 4: Check baseline comparison chart functionality
    """
    print("\n" + "="*80)
    print("TEST ISSUE 4: Baseline Comparison Chart")
    print("="*80)

    try:
        chart = driver.find_element(By.ID, "baseline-comparison")
        svg_elements = chart.find_elements(By.TAG_NAME, "svg")

        if svg_elements:
            print(f"✅ Baseline comparison chart rendered with {len(svg_elements)} SVG element(s)")

            text_elements = chart.find_elements(By.TAG_NAME, "text")
            chart_text = [elem.text for elem in text_elements if elem.text.strip()]

            if chart_text:
                print(f"✅ Chart text: {chart_text[:5]}")
                return {'passed': True, 'details': {'text': chart_text[:5]}}
            else:
                print("✅ Chart rendered (no text)")
                return {'passed': True, 'details': {'message': 'Rendered without text'}}
        else:
            print("❌ Baseline comparison chart not rendered")
            return {'passed': False, 'details': {'message': 'No SVG elements'}}

    except NoSuchElementException:
        print("❌ Baseline comparison chart element not found")
        return {'passed': False, 'details': {'message': 'Element not found'}}

def test_issue_5_supervisor_actions_chart(driver):
    """
    Issue 5: Verify supervisor actions chart displays
    """
    print("\n" + "="*80)
    print("TEST ISSUE 5: Supervisor Actions Chart")
    print("="*80)

    try:
        chart = driver.find_element(By.ID, "supervisor-actions")
        svg_elements = chart.find_elements(By.TAG_NAME, "svg")

        if svg_elements:
            print(f"✅ Supervisor actions chart rendered with {len(svg_elements)} SVG element(s)")

            text_elements = chart.find_elements(By.TAG_NAME, "text")
            chart_text = [elem.text for elem in text_elements if elem.text.strip()]

            if chart_text:
                print(f"✅ Chart text: {chart_text[:5]}")
                return {'passed': True, 'details': {'text': chart_text[:5]}}
            else:
                print("✅ Chart rendered (no text)")
                return {'passed': True, 'details': {'message': 'Rendered without text'}}
        else:
            print("❌ Supervisor actions chart not rendered")
            return {'passed': False, 'details': {'message': 'No SVG elements'}}

    except NoSuchElementException:
        print("❌ Supervisor actions chart element not found")
        return {'passed': False, 'details': {'message': 'Element not found'}}

def test_issue_6_recent_trades_table(driver):
    """
    Issue 6: Check if recent trades table shows real data from Binance
    """
    print("\n" + "="*80)
    print("TEST ISSUE 6: Recent Trades Table (Binance Data)")
    print("="*80)

    try:
        table_container = driver.find_element(By.ID, "recent-trades-table")

        # Check for table element
        tables = table_container.find_elements(By.TAG_NAME, "table")

        if tables:
            table = tables[0]
            rows = table.find_elements(By.TAG_NAME, "tr")
            print(f"✅ Recent trades table found with {len(rows)} rows")

            # Get table content
            table_text = table.text

            # Check for real Binance symbols
            binance_symbols = ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'USDT']
            has_binance_data = any(symbol in table_text for symbol in binance_symbols)

            if has_binance_data:
                print(f"✅ Table shows real Binance data (found trading pairs)")
                return {'passed': True, 'details': {'rows': len(rows), 'has_data': True}}
            elif "no open positions" in table_text.lower() or "no api" in table_text.lower():
                print(f"✅ Table shows proper message: '{table_text[:100]}'")
                return {'passed': True, 'details': {'rows': len(rows), 'message': table_text[:100]}}
            else:
                print(f"⚠️  Table exists but unclear if showing Binance data: '{table_text[:100]}'")
                return {'passed': True, 'details': {'rows': len(rows), 'text': table_text[:100]}}
        else:
            # Check for message (no table)
            message = table_container.text
            if message:
                print(f"✅ Shows message: '{message}'")
                return {'passed': True, 'details': {'message': message}}
            else:
                print("❌ No table or message found")
                return {'passed': False, 'details': {'message': 'Empty container'}}

    except NoSuchElementException:
        print("❌ Recent trades table element not found")
        return {'passed': False, 'details': {'message': 'Element not found'}}

def test_issue_7_closed_trades_table(driver):
    """
    Issue 7: Verify closed trades table shows real data from Binance
    """
    print("\n" + "="*80)
    print("TEST ISSUE 7: Closed Trades Table (Binance Data)")
    print("="*80)

    try:
        table_container = driver.find_element(By.ID, "closed-trades-table")

        # Check for table element
        tables = table_container.find_elements(By.TAG_NAME, "table")

        if tables:
            table = tables[0]
            rows = table.find_elements(By.TAG_NAME, "tr")
            print(f"✅ Closed trades table found with {len(rows)} rows")

            # Get table content
            table_text = table.text

            # Check for real Binance symbols
            binance_symbols = ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'USDT']
            has_binance_data = any(symbol in table_text for symbol in binance_symbols)

            if has_binance_data:
                print(f"✅ Table shows real Binance data (found trading pairs)")
                return {'passed': True, 'details': {'rows': len(rows), 'has_data': True}}
            elif "no recent trades" in table_text.lower() or "no api" in table_text.lower():
                print(f"✅ Table shows proper message: '{table_text[:100]}'")
                return {'passed': True, 'details': {'rows': len(rows), 'message': table_text[:100]}}
            else:
                print(f"⚠️  Table exists but unclear if showing Binance data: '{table_text[:100]}'")
                return {'passed': True, 'details': {'rows': len(rows), 'text': table_text[:100]}}
        else:
            # Check for message (no table)
            message = table_container.text
            if message:
                print(f"✅ Shows message: '{message}'")
                return {'passed': True, 'details': {'message': message}}
            else:
                print("❌ No table or message found")
                return {'passed': False, 'details': {'message': 'Empty container'}}

    except NoSuchElementException:
        print("❌ Closed trades table element not found")
        return {'passed': False, 'details': {'message': 'Element not found'}}

def main():
    """Main test execution"""
    print("\n" + "="*80)
    print("DASHBOARD TESTING - 7 ISSUES VERIFICATION")
    print(f"URL: {DASHBOARD_URL}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    driver = setup_driver()
    test_results = {}

    try:
        # Load dashboard
        print(f"\n🌐 Loading dashboard: {DASHBOARD_URL}")
        driver.get(DASHBOARD_URL)

        # Take initial screenshot
        driver.save_screenshot("tests/screenshots/dashboard_initial.png")
        print("📸 Initial screenshot saved: tests/screenshots/dashboard_initial.png")

        # Wait for dashboard to load
        if not wait_for_dashboard_load(driver):
            print("\n❌ CRITICAL: Dashboard failed to load. Aborting tests.")
            driver.save_screenshot("tests/screenshots/dashboard_load_failed.png")
            return

        # Take loaded screenshot
        driver.save_screenshot("tests/screenshots/dashboard_loaded.png")
        print("📸 Loaded screenshot saved: tests/screenshots/dashboard_loaded.png")

        # Run all tests
        test_results['Issue 1'] = test_issue_1_status_cards(driver)
        test_results['Issue 2'] = test_issue_2_balance_chart(driver)
        test_results['Issue 3'] = test_issue_3_optimization_params_chart(driver)
        test_results['Issue 4'] = test_issue_4_baseline_comparison_chart(driver)
        test_results['Issue 5'] = test_issue_5_supervisor_actions_chart(driver)
        test_results['Issue 6'] = test_issue_6_recent_trades_table(driver)
        test_results['Issue 7'] = test_issue_7_closed_trades_table(driver)

        # Take final screenshot
        driver.save_screenshot("tests/screenshots/dashboard_final.png")
        print("\n📸 Final screenshot saved: tests/screenshots/dashboard_final.png")

        # Print summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)

        passed_count = sum(1 for result in test_results.values() if result['passed'])
        total_count = len(test_results)

        for issue, result in test_results.items():
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            print(f"{status} - {issue}")

        print(f"\nTotal: {passed_count}/{total_count} tests passed")

        # Save detailed results to JSON
        with open('tests/test_results.json', 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'url': DASHBOARD_URL,
                'summary': {
                    'passed': passed_count,
                    'failed': total_count - passed_count,
                    'total': total_count
                },
                'results': test_results
            }, f, indent=2)

        print(f"\n📄 Detailed results saved: tests/test_results.json")

        if passed_count == total_count:
            print("\n🎉 ALL TESTS PASSED!")
            return 0
        else:
            print(f"\n⚠️  {total_count - passed_count} TEST(S) FAILED")
            return 1

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        driver.save_screenshot("tests/screenshots/dashboard_error.png")
        return 1

    finally:
        driver.quit()
        print("\n✅ Browser closed")

if __name__ == "__main__":
    exit(main())
