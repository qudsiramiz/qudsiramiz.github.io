const fs = require('fs');

let appCode = fs.readFileSync('app.js', 'utf8');
const dataRaw = fs.readFileSync('data.json', 'utf8');
const data = JSON.parse(dataRaw);

appCode = appCode.replace(/let state = \{/g, 'var state = {');
appCode = appCode.replace(/let globalKoTeams = \{\};/g, 'var globalKoTeams = {};');

let script = `
  global.window = { location: { hostname: 'localhost', protocol: 'http:' } };
  global.document = {
    addEventListener: () => {},
    querySelectorAll: () => [],
    getElementById: () => null,
    querySelector: () => null,
  };
  
  ${appCode}
  
  state.scores = ${JSON.stringify(data.users['Actual Results'])};
  
  updateScoresAndStandings();
  
  console.log("R32 TEAMS:");
  console.log(JSON.stringify(globalKoTeams, null, 2));
`;

fs.writeFileSync('run_test.js', script);
