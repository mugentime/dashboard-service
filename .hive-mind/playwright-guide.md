# Hive Mind - Playwright MCP Integration Guide

## Available Playwright MCP Tools

The Hive Mind now has access to Playwright MCP for browser automation and testing:

### 1. **mcp__playwright-mcp__playwright_navigate**
Navigate to a URL
```json
{
  "url": "https://example.com"
}
```

### 2. **mcp__playwright-mcp__playwright_screenshot**
Take a screenshot of the current page
```json
{
  "name": "dashboard-view",
  "full_page": true
}
```

### 3. **mcp__playwright-mcp__playwright_click**
Click an element
```json
{
  "selector": "#download-button"
}
```

### 4. **mcp__playwright-mcp__playwright_fill**
Fill input fields
```json
{
  "selector": "#username",
  "value": "testuser"
}
```

### 5. **mcp__playwright-mcp__playwright_evaluate**
Run JavaScript in the browser
```json
{
  "script": "document.querySelector('#status').innerText"
}
```

## Hive Mind Usage Pattern

When the Queen coordinator receives a task involving web testing or UI verification:

1. **Research Agent**: Analyze requirements, identify test scenarios
2. **Coder Agent**: Write test scripts if needed
3. **Tester Agent**: Use Playwright MCP to execute tests
4. **Analyst Agent**: Analyze results and generate reports

## Example: Dashboard Testing Workflow

```javascript
// Queen coordinates the swarm
Task("Research", "Identify dashboard components to test", "researcher")
Task("Test Dashboard", `
  Use Playwright MCP to test dashboard:
  1. Navigate to https://dashboard-production-5a5d.up.railway.app
  2. Screenshot the page
  3. Check for baseline chart (should be removed)
  4. Verify supervisor download button exists
  5. Test optimization parameters display
`, "tester")
Task("Analyze", "Generate test report from results", "analyst")
```

## Best Practices

1. **Always navigate first** before taking screenshots or interacting
2. **Use full_page: true** for complete screenshots
3. **Wait for elements** before clicking (Playwright auto-waits)
4. **Evaluate JavaScript** to check dynamic content
5. **Store results in memory** for other agents to access

## Integration with Hive Mind Memory

Store test results for coordination:
```javascript
mcp__claude-flow__memory_store({
  "key": "hive/test-results/dashboard",
  "value": {
    "timestamp": "2025-10-03T14:05:00Z",
    "tests": [...],
    "passed": 5,
    "failed": 0
  }
})
```
