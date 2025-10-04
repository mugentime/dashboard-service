const puppeteer = require('puppeteer');

async function checkRedisData() {
  let browser;
  let page;

  try {
    console.log('Launching browser...');
    browser = await puppeteer.launch({ headless: 'new' });
    page = await browser.newPage();

    console.log('Navigating to dashboard...');
    await page.goto('https://dashboard-production-5a5d.up.railway.app/', {
      waitUntil: 'networkidle2',
      timeout: 30000
    });

    console.log('Waiting for page to render...');
    await new Promise(resolve => setTimeout(resolve, 5000));

    // Check for Redis status section
    console.log('\n=== Extracting Redis Status ===');
    const redisInfo = await page.evaluate(() => {
      const body = document.body.innerText;

      // Find Redis Status section
      const redisMatch = body.match(/Redis Status[\s\S]{0,200}/i);

      return {
        fullText: body,
        redisSection: redisMatch ? redisMatch[0] : 'Not found',
        hasOffline: body.toLowerCase().includes('redis offline'),
        hasConnected: body.toLowerCase().includes('redis connected'),
        hasNoData: body.toLowerCase().includes('no data') || body.toLowerCase().includes('not available')
      };
    });

    console.log('Redis Status Section:', redisInfo.redisSection);
    console.log('Has "Redis offline":', redisInfo.hasOffline);
    console.log('Has "Redis connected":', redisInfo.hasConnected);
    console.log('Has "No data/not available":', redisInfo.hasNoData);

    // Check if optimization cycle has data
    console.log('\n=== Checking Optimization Cycle ===');
    const optCycle = await page.evaluate(() => {
      const body = document.body.innerText;
      const match = body.match(/Optimization Cycle[\s\S]{0,100}/i);
      return match ? match[0] : 'Not found';
    });
    console.log(optCycle);

    // Check the actual chart/graph for optimization parameters
    console.log('\n=== Checking for Parameter Chart ===');
    const chartInfo = await page.evaluate(() => {
      // Check for Plotly charts (Dash uses Plotly)
      const plotlyDivs = document.querySelectorAll('[class*="plotly"], [id*="optimization"]');
      const chartData = [];

      plotlyDivs.forEach(div => {
        chartData.push({
          id: div.id,
          className: div.className,
          innerHTML: div.innerHTML.substring(0, 200)
        });
      });

      return chartData;
    });

    console.log('Chart elements found:', chartInfo.length);
    chartInfo.forEach((chart, i) => {
      console.log(`Chart ${i + 1}:`, chart.id, chart.className);
    });

    // Try to intercept network requests to see what data is being fetched
    console.log('\n=== Checking Network Activity ===');

    // Reload with network monitoring
    const requests = [];
    page.on('response', async (response) => {
      const url = response.url();
      if (url.includes('_dash') || url.includes('update')) {
        try {
          const data = await response.text();
          if (data.length < 10000) { // Only log small responses
            requests.push({
              url: url,
              status: response.status(),
              data: data.substring(0, 500)
            });
          }
        } catch (e) {
          // Ignore errors reading response
        }
      }
    });

    console.log('Refreshing page to capture network data...');
    await page.reload({ waitUntil: 'networkidle2' });
    await new Promise(resolve => setTimeout(resolve, 3000));

    console.log(`\nCaptured ${requests.length} relevant network requests`);
    requests.slice(0, 3).forEach((req, i) => {
      console.log(`\nRequest ${i + 1}:`);
      console.log('URL:', req.url);
      console.log('Status:', req.status);
      console.log('Data preview:', req.data);
    });

  } catch (error) {
    console.error('Error:', error.message);
  } finally {
    if (page) await page.close();
    if (browser) await browser.close();
  }
}

checkRedisData().catch(console.error);
