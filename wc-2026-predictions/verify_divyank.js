const fs = require('fs');
const path = require('path');

function cleanTeam(name) {
  if (!name) return '';
  return name.trim().toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9]/g, '');
}

const matchesData = JSON.parse(fs.readFileSync(path.join(__dirname, 'matches.json'), 'utf8'));
const dataJson = JSON.parse(fs.readFileSync(path.join(__dirname, 'data.json'), 'utf8'));

const csvContent = fs.readFileSync(path.join(__dirname, 'divyank_predictions.csv'), 'utf8');
const lines = csvContent.split('\n');

const divyankData = dataJson.userScores["Divyank"] || {};
let allMatch = true;
let checkedCount = 0;

for (let i = 1; i < lines.length; i++) {
  const line = lines[i].trim();
  if (!line) continue;
  
  const parts = line.split(',');
  if (parts.length < 6) continue;
  
  const groupStr = parts[1].trim().replace(' ', ''); 
  const team1 = parts[2].trim();
  const team2 = parts[3].trim();
  const score1Str = parts[4].trim();
  const score2Str = parts[5].trim();
  
  if (score1Str === '' || score2Str === '') {
    continue;
  }
  
  const score1 = parseInt(score1Str, 10);
  const score2 = parseInt(score2Str, 10);
  
  if (isNaN(score1) || isNaN(score2)) {
    continue;
  }
  
  const groupMatches = matchesData.groups[groupStr];
  if (!groupMatches) continue;
  
  const match = groupMatches.find(m => {
    const m1 = cleanTeam(m.team1);
    const m2 = cleanTeam(m.team2);
    const p1 = cleanTeam(team1);
    const p2 = cleanTeam(team2);
    return (m1 === p1 && m2 === p2) || (m1 === p2 && m2 === p1);
  });
  
  if (!match) continue;
  
  const inverted = cleanTeam(match.team1) !== cleanTeam(team1);
  const expectedPredHome = inverted ? score2 : score1;
  const expectedPredAway = inverted ? score1 : score2;
  
  const actualPredHome = divyankData[`${match.id}_predHome`];
  const actualPredAway = divyankData[`${match.id}_predAway`];
  
  if (actualPredHome !== expectedPredHome || actualPredAway !== expectedPredAway) {
    console.log(`Mismatch found in ${match.id} (${match.team1} vs ${match.team2}):`);
    console.log(`  CSV says: ${team1} ${score1} - ${score2} ${team2} -> mapped to Home: ${expectedPredHome}, Away: ${expectedPredAway}`);
    console.log(`  JSON says: Home: ${actualPredHome}, Away: ${actualPredAway}`);
    allMatch = false;
  }
  checkedCount++;
}

if (allMatch) {
  console.log(`Success! All ${checkedCount} valid predictions exactly match between the CSV and data.json.`);
} else {
  console.log("Some mismatches were found.");
}
