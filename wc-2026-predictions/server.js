const http = require('http');
const fs = require('fs');
const path = require('path');
const { exec } = require('child_process');

const PORT = 3000;
const DATA_FILE = path.join(__dirname, 'data.json');

const mimeTypes = {
  '.html': 'text/html',
  '.js': 'application/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.svg': 'image/svg+xml',
  '.pdf': 'application/pdf'
};

const server = http.createServer((req, res) => {
  // Set CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  // Handle preflight OPTIONS request
  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  // Handle API request
  if (req.url === '/api/save-and-push' && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => {
      body += chunk.toString();
    });

    req.on('end', () => {
      try {
        // Validate JSON
        const state = JSON.parse(body);
        
        // Write to data.json
        fs.writeFile(DATA_FILE, JSON.stringify(state, null, 2), 'utf8', (err) => {
          if (err) {
            console.error('Error writing data.json:', err);
            res.writeHead(500, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ success: false, error: 'Failed to write data.json to disk.' }));
            return;
          }

          console.log('Successfully wrote data.json locally. Starting Git push...');

          // Only push when data.json actually changed.
          const gitCmd = `git add data.json
if git diff --quiet --cached -- data.json; then
  printf '{"success":true,"message":"Saved data.json locally. No Git changes to push."}'
else
  git commit -m "Update match results via local dashboard" && git push origin main
fi`;
          
          exec(gitCmd, (gitErr, stdout, stderr) => {
            if (gitErr) {
              console.error('Git error:', gitErr, stderr);
              res.writeHead(500, { 'Content-Type': 'application/json' });
              res.end(JSON.stringify({ 
                success: false, 
                error: 'Wrote data.json locally, but Git operations failed.', 
                details: stderr || gitErr.message 
              }));
              return;
            }

            const trimmedOutput = stdout.trim();
            if (trimmedOutput.startsWith('{')) {
              res.writeHead(200, { 'Content-Type': 'application/json' });
              res.end(trimmedOutput);
              return;
            }

            console.log('Git commands output:', stdout);
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ success: true, message: 'Successfully updated data.json and pushed to GitHub.' }));
          });
        });
      } catch (parseErr) {
        console.error('JSON Parse error:', parseErr);
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: false, error: 'Invalid JSON payload.' }));
      }
    });
  } 
  // Handle static file serving
  else if (req.method === 'GET') {
    let reqPath = req.url.split('?')[0].split('#')[0];
    if (reqPath === '/' || reqPath === '') {
      reqPath = '/index.html';
    }

    const filePath = path.join(__dirname, reqPath);

    fs.stat(filePath, (err, stats) => {
      if (err || !stats.isFile()) {
        res.writeHead(404, { 'Content-Type': 'text/plain' });
        res.end('404 Not Found');
        return;
      }

      const ext = path.extname(filePath).toLowerCase();
      const contentType = mimeTypes[ext] || 'application/octet-stream';

      res.writeHead(200, { 'Content-Type': contentType });
      const stream = fs.createReadStream(filePath);
      stream.pipe(res);
    });
  } else {
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Endpoint not found' }));
  }
});

server.listen(PORT, () => {
  console.log(`Helper server is running at http://localhost:${PORT}`);
  console.log(`Press Ctrl+C to stop the server.`);
});
