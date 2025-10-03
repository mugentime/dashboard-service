#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script to test dashboard callback directly
"""

import sys
import io
import requests
import json

# Fix Unicode encoding for Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

DASHBOARD_URL = "https://dashboard-production-5a5d.up.railway.app"

def test_callback():
    """Test the Dash callback endpoint"""
    url = f"{DASHBOARD_URL}/_dash-update-component"

    # Construct callback payload
    payload = {
        "output": "..bot-status.children...balance-display.children...active-trades.children...optimization-cycle.children...supervisor-status.children...balance-chart.figure...optimization-params-chart.figure...baseline-comparison.figure...supervisor-actions.figure...recent-trades-table.children...closed-trades-table.children...bot-activity-log.children...supervisor-detailed-status.children...supervisor-alerts-table.children...supervisor-actions-table.children..",
        "outputs": [
            {"id": "bot-status", "property": "children"},
            {"id": "balance-display", "property": "children"},
            {"id": "active-trades", "property": "children"},
            {"id": "optimization-cycle", "property": "children"},
            {"id": "supervisor-status", "property": "children"},
            {"id": "balance-chart", "property": "figure"},
            {"id": "optimization-params-chart", "property": "figure"},
            {"id": "baseline-comparison", "property": "figure"},
            {"id": "supervisor-actions", "property": "figure"},
            {"id": "recent-trades-table", "property": "children"},
            {"id": "closed-trades-table", "property": "children"},
            {"id": "bot-activity-log", "property": "children"},
            {"id": "supervisor-detailed-status", "property": "children"},
            {"id": "supervisor-alerts-table", "property": "children"},
            {"id": "supervisor-actions-table", "property": "children"}
        ],
        "inputs": [
            {"id": "interval-component", "property": "n_intervals", "value": 0}
        ],
        "changedPropIds": ["interval-component.n_intervals"],
        "state": []
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    print(f"Testing callback: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)[:500]}...")

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        print(f"\nStatus Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")

        if response.status_code == 200:
            try:
                data = response.json()
                print(f"\n✅ SUCCESS - Response:")
                print(json.dumps(data, indent=2)[:1000])
            except json.JSONDecodeError as e:
                print(f"\n❌ JSON Decode Error: {e}")
                print(f"Response text: {response.text[:500]}")
        else:
            print(f"\n❌ ERROR Response:")
            print(response.text[:1000])

    except requests.exceptions.Timeout:
        print("\n❌ Request timed out after 30 seconds")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_callback()
