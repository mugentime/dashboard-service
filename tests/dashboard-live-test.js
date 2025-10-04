const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

async function testDashboard() {
  const results = {
    timestamp: new Date().toISOString(),
    url: 'https://dashboard-production-5a5d.up.railway.app/',
    tests: [],
    screenshots: [],
    consoleErrors: [],
    networkErrors: []
  };

  let browser;
  let context;
  let page;

  try {
    console.log('Launching browser...');
    browser = await chromium.launch({ headless: true });
    context = await browser.newContext({
      viewport: { width: 1920, height: 1080 }
    });
    page = await context.newPage();

    // Capture console messages
    page.on('console', msg => {
      const type = msg.type();
      const text = msg.text();
      console.log(`[Browser Console ${type}]: ${text}`);
      if (type === 'error') {
        results.consoleErrors.push(text);
      }
    });

    // Capture network errors
    page.on('requestfailed', request => {
      const failure = `${request.url()} - ${request.failure().errorText}`;
      console.log(`[Network Error]: ${failure}`);
      results.networkErrors.push(failure);
    });

    console.log('\n=== TEST 1: Navigate to Dashboard ===');
    await page.goto(results.url, { waitUntil: 'networkidle', timeout: 30000 });
    console.log('✓ Successfully loaded dashboard');
    results.tests.push({ name: 'Navigation', status: 'PASS', details: 'Page loaded successfully' });

    // Take full page screenshot
    const screenshotPath1 = path.join(__dirname, 'screenshot-full-page.png');
    await page.screenshot({ path: screenshotPath1, fullPage: true });
    console.log(`✓ Full page screenshot saved: ${screenshotPath1}`);
    results.screenshots.push(screenshotPath1);

    console.log('\n=== TEST 2: Check Page Title ===');
    const title = await page.title();
    console.log(`Page title: "${title}"`);
    results.tests.push({ name: 'Page Title', status: 'INFO', details: title });

    console.log('\n=== TEST 3: Check Bot Optimization Parameters Section ===');

    // Wait for the optimization parameters section
    try {
      await page.waitForSelector('h2:has-text("Bot Optimization Parameters")', { timeout: 5000 });
      console.log('✓ Found "Bot Optimization Parameters" heading');
      results.tests.push({ name: 'Section Heading', status: 'PASS', details: 'Bot Optimization Parameters section exists' });
    } catch (e) {
      console.log('✗ Could not find "Bot Optimization Parameters" heading');
      results.tests.push({ name: 'Section Heading', status: 'FAIL', details: 'Section heading not found' });
    }

    // Check for specific parameters
    const expectedParams = [
      'momentum_threshold',
      'confidence_multiplier',
      'volume_weight',
      'short_ma_weight',
      'med_ma_weight',
      'min_confidence_threshold'
    ];

    console.log('\nChecking for parameters:');
    const foundParams = [];
    const missingParams = [];

    for (const param of expectedParams) {
      const exists = await page.locator(`text="${param}"`).count() > 0;
      if (exists) {
        console.log(`  ✓ ${param} - FOUND`);
        foundParams.push(param);

        // Try to get the value
        try {
          const valueElement = await page.locator(`text="${param}"`).locator('..').textContent();
          console.log(`    Value context: ${valueElement.trim()}`);
        } catch (e) {
          console.log(`    Could not extract value`);
        }
      } else {
        console.log(`  ✗ ${param} - MISSING`);
        missingParams.push(param);
      }
    }

    results.tests.push({
      name: 'Parameters Display',
      status: foundParams.length === 6 ? 'PASS' : 'PARTIAL',
      details: `Found ${foundParams.length}/6 parameters: ${foundParams.join(', ')}`,
      missing: missingParams
    });

    console.log('\n=== TEST 4: Check for Chart Rendering ===');
    const chartExists = await page.locator('canvas').count() > 0 ||
                       await page.locator('svg').count() > 0 ||
                       await page.locator('[class*="chart"]').count() > 0;

    if (chartExists) {
      console.log('✓ Chart element found');
      results.tests.push({ name: 'Chart Rendering', status: 'PASS', details: 'Chart element exists' });
    } else {
      console.log('✗ No chart element found');
      results.tests.push({ name: 'Chart Rendering', status: 'FAIL', details: 'No chart element detected' });
    }

    console.log('\n=== TEST 5: Check Redis Status ===');

    // Check for Redis status message
    const pageText = await page.textContent('body');
    const hasRedisOffline = pageText.toLowerCase().includes('redis offline');

    if (hasRedisOffline) {
      console.log('✗ Redis appears to be OFFLINE');
      results.tests.push({ name: 'Redis Status', status: 'FAIL', details: 'Redis offline message detected' });
    } else {
      console.log('✓ No "Redis offline" message found');

      // Try to find timestamp
      const timestampRegex = /updated|last\s+update|timestamp/i;
      const hasTimestamp = timestampRegex.test(pageText);

      if (hasTimestamp) {
        console.log('✓ Found update timestamp indicator');
        results.tests.push({ name: 'Redis Status', status: 'PASS', details: 'Redis appears online with timestamps' });
      } else {
        console.log('? Could not confirm timestamp presence');
        results.tests.push({ name: 'Redis Status', status: 'PARTIAL', details: 'No offline message, but timestamp unclear' });
      }
    }

    // Take screenshot of optimization parameters section
    console.log('\n=== TEST 6: Screenshot Optimization Section ===');
    try {
      const section = page.locator('h2:has-text("Bot Optimization Parameters")').locator('..');
      const screenshotPath2 = path.join(__dirname, 'screenshot-optimization-section.png');
      await section.screenshot({ path: screenshotPath2 });
      console.log(`✓ Optimization section screenshot saved: ${screenshotPath2}`);
      results.screenshots.push(screenshotPath2);
    } catch (e) {
      console.log(`✗ Could not capture section screenshot: ${e.message}`);
    }

    console.log('\n=== TEST 7: Extract All Visible Data ===');

    // Get all parameter data from the page
    const parameterData = await page.evaluate(() => {
      const data = {};
      const elements = document.querySelectorAll('[class*="parameter"], [class*="metric"], [class*="stat"]');
      elements.forEach(el => {
        const text = el.textContent.trim();
        if (text) {
          data[text.substring(0, 50)] = text;
        }
      });
      return data;
    });

    console.log('Extracted parameter data:');
    console.log(JSON.stringify(parameterData, null, 2));
    results.tests.push({ name: 'Data Extraction', status: 'INFO', details: parameterData });

    console.log('\n=== TEST 8: Check for Error Messages ===');
    const errorMessages = await page.evaluate(() => {
      const errors = [];
      document.querySelectorAll('[class*="error"], [class*="alert"], [class*="warning"]').forEach(el => {
        const text = el.textContent.trim();
        if (text) errors.push(text);
      });
      return errors;
    });

    if (errorMessages.length > 0) {
      console.log('⚠ Found error messages on page:');
      errorMessages.forEach(msg => console.log(`  - ${msg}`));
      results.tests.push({ name: 'Error Messages', status: 'WARN', details: errorMessages });
    } else {
      console.log('✓ No error messages found');
      results.tests.push({ name: 'Error Messages', status: 'PASS', details: 'No visible errors' });
    }

    console.log('\n=== SUMMARY ===');
    console.log(`Total tests run: ${results.tests.length}`);
    console.log(`Screenshots captured: ${results.screenshots.length}`);
    console.log(`Console errors: ${results.consoleErrors.length}`);
    console.log(`Network errors: ${results.networkErrors.length}`);

  } catch (error) {
    console.error('Test execution error:', error);
    results.tests.push({ name: 'Test Execution', status: 'ERROR', details: error.message });
  } finally {
    if (page) await page.close();
    if (context) await context.close();
    if (browser) await browser.close();
  }

  // Save results to file
  const resultsPath = path.join(__dirname, 'test-results.json');
  fs.writeFileSync(resultsPath, JSON.stringify(results, null, 2));
  console.log(`\n✓ Test results saved to: ${resultsPath}`);

  return results;
}

testDashboard().catch(console.error);
