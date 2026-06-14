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
const csvContent = fs.readFileSync(path.join(__dirname, 'divyank_predictions.csv'), 'utf8');
const lines = csvContent.split('\n');

const parsedPredictions = {};

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
  
  if (score1Str === '' || score2Str === '') continue;
  
  const score1 = parseInt(score1Str, 10);
  const score2 = parseInt(score2Str, 10);
  if (isNaN(score1) || isNaN(score2)) continue;
  
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
  const predHome = inverted ? score2 : score1;
  const predAway = inverted ? score1 : score2;
  
  console.log(`Setting ${match.id} (${match.team1} vs ${match.team2}) from CSV line: ${team1} ${score1}-${score2} ${team2} -> Home: ${predHome}, Away: ${predAway}`);
  
  parsedPredictions[`${match.id}_predHome`] = predHome;
  parsedPredictions[`${match.id}_predAway`] = predAway;
}
