const fs = require('fs');
let appCode = fs.readFileSync('app.js', 'utf8');
const dataRaw = fs.readFileSync('data.json', 'utf8');
const data = JSON.parse(dataRaw);
const scoresStr = JSON.stringify(data.users['Actual Results']);
appCode = appCode.replace('scores: {},', `scores: ${scoresStr},`);
let script = `
  global.window = { location: { hostname: 'localhost', protocol: 'http:' } };
  global.document = {
    addEventListener: () => {}, querySelectorAll: () => [],
    getElementById: () => null, querySelector: () => null,
  };
  ${appCode}
  updateScoresAndStandings();
  console.log("QUALIFIED THIRDS:");
  Object.keys(globalKoTeams).forEach(k => console.log(k, globalKoTeams[k]));
`;
fs.writeFileSync('run_test2.js', script);
