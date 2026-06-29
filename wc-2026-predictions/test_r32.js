const fs = require('fs');

const appCode = fs.readFileSync('app.js', 'utf8');
const dataRaw = fs.readFileSync('data.json', 'utf8');
const data = JSON.parse(dataRaw);

// Mock DOM
global.window = { location: { hostname: 'localhost', protocol: 'http:' } };
global.document = {
  addEventListener: () => {},
  querySelectorAll: () => [],
  getElementById: () => null,
  querySelector: () => null,
};

eval(appCode);
state.scores = data.users['Actual Results'];

// Run updateScoresAndStandings logic
updateScoresAndStandings();

console.log("R32 TEAMS:");
console.dir(globalKoTeams, {depth: null});
