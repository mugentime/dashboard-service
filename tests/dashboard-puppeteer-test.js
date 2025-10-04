const puppeteer = require('puppeteer');
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
  let page;

  try {
    console.log('Launching browser...');
    browser = await puppeteer.launch({
      headless: 'new',
      args: ['--no-sandbox', '--disable-setuid-sandbox']
    });
    page = await browser.newPage();

    // Set viewport
    await page.setViewport({ width: 1920, height: 1080 });

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
    await page.goto(results.url, { waitUntil: 'networkidle2', timeout: 30000 });
    console.log('✓ Successfully loaded dashboard');
    results.tests.push({ name: 'Navigation', status: 'PASS', details: 'Page loaded successfully' });

    // Wait for Dash to render
    console.log('Waiting for Dash app to render...');
    await page.waitForSelector('#react-entry-point', { timeout: 10000 });
    await new Promise(resolve => setTimeout(resolve, 3000)); // Give Dash time to fully render

    // Take full page screenshot
    const screenshotPath1 = path.join(__dirname, 'screenshot-full-page.png');
    await page.screenshot({ path: screenshotPath1, fullPage: true });
    console.log(`✓ Full page screenshot saved: ${screenshotPath1}`);
    results.screenshots.push(screenshotPath1);

    console.log('\n=== TEST 2: Check Page Title ===');
    const title = await page.title();
    console.log(`Page title: "${title}"`);
    results.tests.push({ name: 'Page Title', status: 'INFO', details: title });

    console.log('\n=== TEST 3: Extract All Page Text ===');
    const pageText = await page.evaluate(() => document.body.innerText);
    console.log('Page content (first 1000 chars):');
    console.log(pageText.substring(0, 1000));
    console.log('...\n');

    console.log('\n=== TEST 4: Check Bot Optimization Parameters Section ===');

    // Check for the heading
    const hasOptimizationHeading = pageText.includes('Bot Optimization Parameters');
    if (hasOptimizationHeading) {
      console.log('✓ Found "Bot Optimization Parameters" heading');
      results.tests.push({ name: 'Section Heading', status: 'PASS', details: 'Bot Optimization Parameters section exists' });
    } else {
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
    const paramValues = {};

    for (const param of expectedParams) {
      if (pageText.includes(param)) {
        console.log(`  ✓ ${param} - FOUND`);
        foundParams.push(param);

        // Try to extract value using regex
        const regex = new RegExp(`${param}[:\\s]+([0-9.]+)`, 'i');
        const match = pageText.match(regex);
        if (match && match[1]) {
          paramValues[param] = match[1];
          console.log(`    Value: ${match[1]}`);
        }
      } else {
        console.log(`  ✗ ${param} - MISSING`);
        missingParams.push(param);
      }
    }

    results.tests.push({
      name: 'Parameters Display',
      status: foundParams.length === 6 ? 'PASS' : 'PARTIAL',
      details: `Found ${foundParams.length}/6 parameters`,
      foundParams: foundParams,
      missingParams: missingParams,
      parameterValues: paramValues
    });

    console.log('\n=== TEST 5: Check for Chart/Graph Elements ===');
    const hasChart = await page.evaluate(() => {
      const canvas = document.querySelectorAll('canvas');
      const svg = document.querySelectorAll('svg');
      const graphs = document.querySelectorAll('[class*="graph"], [class*="chart"], [id*="graph"], [id*="chart"]');
      return {
        canvasCount: canvas.length,
        svgCount: svg.length,
        graphElements: graphs.length
      };
    });

    console.log(`Canvas elements: ${hasChart.canvasCount}`);
    console.log(`SVG elements: ${hasChart.svgCount}`);
    console.log(`Graph/Chart elements: ${hasChart.graphElements}`);

    const chartExists = hasChart.canvasCount > 0 || hasChart.svgCount > 0 || hasChart.graphElements > 0;
    if (chartExists) {
      console.log('✓ Chart elements found');
      results.tests.push({ name: 'Chart Rendering', status: 'PASS', details: hasChart });
    } else {
      console.log('✗ No chart elements found');
      results.tests.push({ name: 'Chart Rendering', status: 'FAIL', details: 'No chart elements detected' });
    }

    console.log('\n=== TEST 6: Check Redis Status ===');

    const hasRedisOffline = pageText.toLowerCase().includes('redis offline');
    const hasRedisConnected = pageText.toLowerCase().includes('redis connected') ||
                             pageText.toLowerCase().includes('redis online');
    const hasTimestamp = /updated|last\s+update|timestamp|ago|seconds|minutes/i.test(pageText);

    if (hasRedisOffline) {
      console.log('✗ Redis appears to be OFFLINE');
      results.tests.push({
        name: 'Redis Status',
        status: 'FAIL',
        details: 'Redis offline message detected'
      });
    } else if (hasRedisConnected) {
      console.log('✓ Redis appears to be CONNECTED');
      results.tests.push({
        name: 'Redis Status',
        status: 'PASS',
        details: 'Redis connected'
      });
    } else {
      console.log('? Redis status unclear');
      results.tests.push({
        name: 'Redis Status',
        status: 'UNKNOWN',
        details: 'No clear Redis status indicator found'
      });
    }

    if (hasTimestamp) {
      console.log('✓ Found timestamp/update indicator');

      // Try to extract timestamp
      const timestampMatch = pageText.match(/(?:updated|last update|timestamp)[:\s]*([^\n]+)/i);
      if (timestampMatch) {
        console.log(`  Timestamp text: ${timestampMatch[1].substring(0, 100)}`);
        results.tests.push({
          name: 'Update Timestamp',
          status: 'PASS',
          details: timestampMatch[1].substring(0, 100)
        });
      }
    } else {
      console.log('✗ No timestamp indicator found');
      results.tests.push({
        name: 'Update Timestamp',
        status: 'FAIL',
        details: 'No timestamp found'
      });
    }

    // Try to capture specific section screenshot
    console.log('\n=== TEST 7: Screenshot Specific Sections ===');
    try {
      // Try to find the optimization section
      const optimizationSection = await page.$('h2, h3, h4, div');
      if (optimizationSection) {
        const screenshotPath2 = path.join(__dirname, 'screenshot-optimization-section.png');
        await optimizationSection.screenshot({ path: screenshotPath2 });
        console.log(`✓ Section screenshot saved: ${screenshotPath2}`);
        results.screenshots.push(screenshotPath2);
      }
    } catch (e) {
      console.log(`Could not capture section screenshot: ${e.message}`);
    }

    console.log('\n=== TEST 8: Check for Error Messages ===');
    const errorMessages = await page.evaluate(() => {
      const errors = [];
      const errorSelectors = [
        '[class*="error"]',
        '[class*="alert"]',
        '[class*="warning"]',
        '[class*="fail"]',
        '[role="alert"]'
      ];

      errorSelectors.forEach(selector => {
        document.querySelectorAll(selector).forEach(el => {
          const text = el.innerText.trim();
          if (text && text.length > 0 && text.length < 500) {
            errors.push(text);
          }
        });
      });
      return [...new Set(errors)]; // Remove duplicates
    });

    if (errorMessages.length > 0) {
      console.log('⚠ Found error/warning messages on page:');
      errorMessages.forEach(msg => console.log(`  - ${msg}`));
      results.tests.push({ name: 'Error Messages', status: 'WARN', details: errorMessages });
    } else {
      console.log('✓ No error messages found');
      results.tests.push({ name: 'Error Messages', status: 'PASS', details: 'No visible errors' });
    }

    console.log('\n=== TEST 9: Extract Structure Information ===');
    const structure = await page.evaluate(() => {
      const info = {
        headings: [],
        tables: 0,
        inputs: 0,
        buttons: 0,
        divs: document.querySelectorAll('div').length
      };

      // Get all headings
      ['h1', 'h2', 'h3', 'h4'].forEach(tag => {
        document.querySelectorAll(tag).forEach(el => {
          info.headings.push(`${tag}: ${el.innerText.trim()}`);
        });
      });

      info.tables = document.querySelectorAll('table').length;
      info.inputs = document.querySelectorAll('input').length;
      info.buttons = document.querySelectorAll('button').length;

      return info;
    });

    console.log('Page structure:');
    console.log(`  Headings: ${structure.headings.length}`);
    structure.headings.forEach(h => console.log(`    ${h}`));
    console.log(`  Tables: ${structure.tables}`);
    console.log(`  Input fields: ${structure.inputs}`);
    console.log(`  Buttons: ${structure.buttons}`);
    console.log(`  Divs: ${structure.divs}`);

    results.tests.push({
      name: 'Page Structure',
      status: 'INFO',
      details: structure
    });

    // Save the full page text for analysis
    const textPath = path.join(__dirname, 'page-content.txt');
    fs.writeFileSync(textPath, pageText);
    console.log(`\n✓ Full page text saved to: ${textPath}`);

    console.log('\n=== SUMMARY ===');
    console.log(`Total tests run: ${results.tests.length}`);
    console.log(`Screenshots captured: ${results.screenshots.length}`);
    console.log(`Console errors: ${results.consoleErrors.length}`);
    console.log(`Network errors: ${results.networkErrors.length}`);

    // Print summary of key findings
    console.log('\n=== KEY FINDINGS ===');
    console.log(`1. Parameters displayed: ${foundParams.length}/6`);
    if (foundParams.length > 0) {
      console.log(`   Found: ${foundParams.join(', ')}`);
    }
    if (missingParams.length > 0) {
      console.log(`   Missing: ${missingParams.join(', ')}`);
    }
    if (Object.keys(paramValues).length > 0) {
      console.log('   Values extracted:');
      Object.entries(paramValues).forEach(([key, val]) => {
        console.log(`     ${key}: ${val}`);
      });
    }
    console.log(`2. Redis status: ${hasRedisOffline ? 'OFFLINE' : hasRedisConnected ? 'CONNECTED' : 'UNKNOWN'}`);
    console.log(`3. Chart elements: ${chartExists ? 'YES' : 'NO'}`);
    console.log(`4. Error messages: ${errorMessages.length > 0 ? 'YES' : 'NO'}`);

  } catch (error) {
    console.error('Test execution error:', error);
    results.tests.push({ name: 'Test Execution', status: 'ERROR', details: error.message, stack: error.stack });
  } finally {
    if (page) await page.close();
    if (browser) await browser.close();
  }

  // Save results to file
  const resultsPath = path.join(__dirname, 'test-results.json');
  fs.writeFileSync(resultsPath, JSON.stringify(results, null, 2));
  console.log(`\n✓ Test results saved to: ${resultsPath}`);

  return results;
}

testDashboard().catch(console.error);
