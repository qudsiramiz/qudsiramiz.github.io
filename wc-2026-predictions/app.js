// Embedded Matches Database (extracted from main.tex)
const initialMatchesData = {
  "groups": {
    "GroupA": [
      { "id": "M1", "team1": "Mexico", "team2": "South Africa", "info": "M1, 13h, 11 June | UTC 19h" },
      { "id": "M2", "team1": "South Korea", "team2": "Czech Rep.", "info": "M2, 20h, 11 June | UTC 02h, 12 June" },
      { "id": "M25", "team1": "Czech Rep.", "team2": "South Africa", "info": "M25, 12h, 18 June | UTC 16h" },
      { "id": "M28", "team1": "Mexico", "team2": "South Korea", "info": "M28, 19h, 18 June | UTC 01h, 19 June" },
      { "id": "M53", "team1": "Czech Rep.", "team2": "Mexico", "info": "M53, 19h, 24 June | UTC 01h, 25 June" },
      { "id": "M54", "team1": "South Africa", "team2": "South Korea", "info": "M54, 19h, 24 June | UTC 01h, 25 June" }
    ],
    "GroupB": [
      { "id": "M3", "team1": "Canada", "team2": "Bosnia-Herz.", "info": "M3, 15h, 12 June | UTC 19h" },
      { "id": "M8", "team1": "Qatar", "team2": "Switzerland", "info": "M8, 12h, 13 June | UTC 19h" },
      { "id": "M26", "team1": "Switzerland", "team2": "Bosnia-Herz.", "info": "M26, 12h, 18 June | UTC 19h" },
      { "id": "M27", "team1": "Canada", "team2": "Qatar", "info": "M27, 15h, 18 June | UTC 22h" },
      { "id": "M51", "team1": "Switzerland", "team2": "Canada", "info": "M51, 12h, 24 June | UTC 19h" },
      { "id": "M52", "team1": "Bosnia-Herz.", "team2": "Qatar", "info": "M52, 12h, 24 June | UTC 19h" }
    ],
    "GroupC": [
      { "id": "M5", "team1": "Haiti", "team2": "Scotland", "info": "M5, 21h, 13 June | UTC 01h, 14 June" },
      { "id": "M7", "team1": "Brazil", "team2": "Morocco", "info": "M7, 18h, 13 June | UTC 22h" },
      { "id": "M29", "team1": "Brazil", "team2": "Haiti", "info": "M29, 21h, 19 June | UTC 01h, 20 June" },
      { "id": "M30", "team1": "Scotland", "team2": "Morocco", "info": "M30, 18h, 19 June | UTC 22h" },
      { "id": "M49", "team1": "Scotland", "team2": "Brazil", "info": "M49, 18h, 24 June | UTC 22h" },
      { "id": "M50", "team1": "Morocco", "team2": "Haiti", "info": "M50, 18h, 24 June | UTC 22h" }
    ],
    "GroupD": [
      { "id": "M4", "team1": "United States", "team2": "Paraguay", "info": "M4, 18h, 12 June | UTC 01h, 13 June" },
      { "id": "M6", "team1": "Australia", "team2": "Türkiye", "info": "M6, 21h, 13 June | UTC 04h, 14 June" },
      { "id": "M31", "team1": "Türkiye", "team2": "Paraguay", "info": "M31, 21h, 19 June | UTC 04h, 20 June" },
      { "id": "M32", "team1": "United States", "team2": "Australia", "info": "M32, 12h, 19 June | UTC 19h" },
      { "id": "M59", "team1": "Türkiye", "team2": "United States", "info": "M59, 19h, 25 June | UTC 02h, 26 June" },
      { "id": "M60", "team1": "Paraguay", "team2": "Australia", "info": "M60, 19h, 25 June | UTC 02h, 26 June" }
    ],
    "GroupE": [
      { "id": "M9", "team1": "Ivory Coast", "team2": "Ecuador", "info": "M9, 19h, 14 June | UTC 23h" },
      { "id": "M10", "team1": "Germany", "team2": "Curaçao", "info": "M10, 12h, 14 June | UTC 17h" },
      { "id": "M33", "team1": "Germany", "team2": "Ivory Coast", "info": "M33, 16h, 20 June | UTC 20h" },
      { "id": "M34", "team1": "Ecuador", "team2": "Curaçao", "info": "M34, 19h, 20 June | UTC 00h, 21 June" },
      { "id": "M55", "team1": "Curaçao", "team2": "Ivory Coast", "info": "M55, 16h, 25 June | UTC 20h" },
      { "id": "M56", "team1": "Ecuador", "team2": "Germany", "info": "M56, 16h, 25 June | UTC 20h" }
    ],
    "GroupF": [
      { "id": "M11", "team1": "Netherlands", "team2": "Japan", "info": "M11, 15h, 14 June | UTC 20h" },
      { "id": "M12", "team1": "Sweden", "team2": "Tunisia", "info": "M12, 20h, 14 June | UTC 02h, 15 June" },
      { "id": "M35", "team1": "Netherlands", "team2": "Sweden", "info": "M35, 12h, 20 June | UTC 17h" },
      { "id": "M36", "team1": "Tunisia", "team2": "Japan", "info": "M36, 22h, 20 June | UTC 02h, 21 June" },
      { "id": "M57", "team1": "Japan", "team2": "Sweden", "info": "M57, 18h, 25 June | UTC 23h" },
      { "id": "M58", "team1": "Tunisia", "team2": "Netherlands", "info": "M58, 18h, 25 June | UTC 23h" }
    ],
    "GroupG": [
      { "id": "M15", "team1": "Iran", "team2": "New Zealand", "info": "M15, 18h, 15 June | UTC 01h, 16 June" },
      { "id": "M16", "team1": "Belgium", "team2": "Egypt", "info": "M16, 12h, 15 June | UTC 19h" },
      { "id": "M39", "team1": "Belgium", "team2": "Iran", "info": "M39, 12h, 21 June | UTC 19h" },
      { "id": "M40", "team1": "New Zealand", "team2": "Egypt", "info": "M40, 18h, 21 June | UTC 01h, 22 June" },
      { "id": "M63", "team1": "Egypt", "team2": "Iran", "info": "M63, 20h, 26 June | UTC 03h, 27 June" },
      { "id": "M64", "team1": "New Zealand", "team2": "Belgium", "info": "M64, 20h, 26 June | UTC 03h, 27 June" }
    ],
    "GroupH": [
      { "id": "M13", "team1": "Saudi Arabia", "team2": "Uruguay", "info": "M13, 18h, 15 June | UTC 22h" },
      { "id": "M14", "team1": "Spain", "team2": "Cape Verde", "info": "M14, 12h, 15 June | UTC 16h" },
      { "id": "M37", "team1": "Uruguay", "team2": "Cape Verde", "info": "M37, 18h, 21 June | UTC 22h" },
      { "id": "M38", "team1": "Spain", "team2": "Saudi Arabia", "info": "M38, 12h, 21 June | UTC 16h" },
      { "id": "M65", "team1": "Cape Verde", "team2": "Saudi Arabia", "info": "M65, 19h, 26 June | UTC 00h, 27 June" },
      { "id": "M66", "team1": "Uruguay", "team2": "Spain", "info": "M66, 18h, 26 June | UTC 00h, 27 June" }
    ],
    "GroupI": [
      { "id": "M17", "team1": "France", "team2": "Senegal", "info": "M17, 15h, 16 June | UTC 19h" },
      { "id": "M18", "team1": "Iraq", "team2": "Norway", "info": "M18, 18h, 16 June | UTC 22h" },
      { "id": "M41", "team1": "Norway", "team2": "Senegal", "info": "M41, 20h, 22 June | UTC 00h, 23 June" },
      { "id": "M42", "team1": "France", "team2": "Iraq", "info": "M42, 17h, 22 June | UTC 21h" },
      { "id": "M61", "team1": "Norway", "team2": "France", "info": "M61, 15h, 26 June | UTC 19h" },
      { "id": "M62", "team1": "Senegal", "team2": "Iraq", "info": "M62, 15h, 26 June | UTC 19h" }
    ],
    "GroupJ": [
      { "id": "M19", "team1": "Argentina", "team2": "Algeria", "info": "M19, 20h, 16 June | UTC 01h, 17 June" },
      { "id": "M20", "team1": "Austria", "team2": "Jordan", "info": "M20, 21h, 16 June | UTC 04h, 17 June" },
      { "id": "M43", "team1": "Argentina", "team2": "Austria", "info": "M43, 12h, 22 June | UTC 17h" },
      { "id": "M44", "team1": "Jordan", "team2": "Algeria", "info": "M44, 20h, 22 June | UTC 03h, 23 June" },
      { "id": "M69", "team1": "Algeria", "team2": "Austria", "info": "M69, 21h, 27 June | UTC 02h, 28 June" },
      { "id": "M70", "team1": "Jordan", "team2": "Argentina", "info": "M70, 21h, 27 June | UTC 02h, 28 June" }
    ],
    "GroupK": [
      { "id": "M23", "team1": "Portugal", "team2": "DR Congo", "info": "M23, 12h, 17 June | UTC 17h" },
      { "id": "M24", "team1": "Uzbekistan", "team2": "Colombia", "info": "M24, 20h, 17 June | UTC 02h, 18 June" },
      { "id": "M47", "team1": "Portugal", "team2": "Uzbekistan", "info": "M47, 12h, 23 June | UTC 17h" },
      { "id": "M48", "team1": "Colombia", "team2": "DR Congo", "info": "M48, 20h, 23 June | UTC 02h, 24 June" },
      { "id": "M71", "team1": "Colombia", "team2": "Portugal", "info": "M71, 19:30h, 27 June | UTC 23:30h" },
      { "id": "M72", "team1": "DR Congo", "team2": "Uzbekistan", "info": "M72, 19:30h, 27 June | UTC 23:30h" }
    ],
    "GroupL": [
      { "id": "M21", "team1": "Ghana", "team2": "Panama", "info": "M21, 19h, 17 June | UTC 23h" },
      { "id": "M22", "team1": "England", "team2": "Croatia", "info": "M22, 15h, 17 June | UTC 20h" },
      { "id": "M45", "team1": "England", "team2": "Ghana", "info": "M45, 16h, 23 June | UTC 20h" },
      { "id": "M46", "team1": "Panama", "team2": "Croatia", "info": "M46, 19h, 23 June | UTC 23h" },
      { "id": "M67", "team1": "Panama", "team2": "England", "info": "M67, 17h, 27 June | UTC 21h" },
      { "id": "M68", "team1": "Croatia", "team2": "Ghana", "info": "M68, 17h, 27 June | UTC 21h" }
    ]
  },
  "r32": [
    { "node_id": "R32_1", "id": "M74", "team1_placeholder": "1E", "team2_placeholder": "3ABCDF", "info": "M74, 18:30h, 29 June | UTC 22:30h" },
    { "node_id": "R32_2", "id": "M77", "team1_placeholder": "1I", "team2_placeholder": "3CDFGH", "info": "M77, 17h, 30 June | UTC 21h" },
    { "node_id": "R32_3", "id": "M73", "team1_placeholder": "2A", "team2_placeholder": "2B", "info": "M73, 12h, 28 June | UTC 19h" },
    { "node_id": "R32_4", "id": "M75", "team1_placeholder": "1F", "team2_placeholder": "2C", "info": "M75, 19h, 29 June | UTC 01h, 30 June" },
    { "node_id": "R32_5", "id": "M83", "team1_placeholder": "2K", "team2_placeholder": "2L", "info": "M83, 19h, 2 July | UTC 23h" },
    { "node_id": "R32_6", "id": "M84", "team1_placeholder": "1H", "team2_placeholder": "2J", "info": "M84, 12h, 2 July | UTC 19h" },
    { "node_id": "R32_7", "id": "M81", "team1_placeholder": "1D", "team2_placeholder": "3BEFIJ", "info": "M81, 17h, 1 July | UTC 00h, 2 July" },
    { "node_id": "R32_8", "id": "M82", "team1_placeholder": "1G", "team2_placeholder": "3AEHIJ", "info": "M82, 13h, 1 July | UTC 20h" },
    { "node_id": "R32_9", "id": "M76", "team1_placeholder": "1C", "team2_placeholder": "2F", "info": "M76, 12h, 29 June | UTC 18h" },
    { "node_id": "R32_10", "id": "M78", "team1_placeholder": "2E", "team2_placeholder": "2I", "info": "M78, 12h, 30 June | UTC 17h" },
    { "node_id": "R32_11", "id": "M79", "team1_placeholder": "1A", "team2_placeholder": "3CEFHI", "info": "M79, 19h, 30 June | UTC 01h, 1 July" },
    { "node_id": "R32_12", "id": "M80", "team1_placeholder": "1L", "team2_placeholder": "3EHIJK", "info": "M80, 12h, 1 July | UTC 16h" },
    { "node_id": "R32_13", "id": "M86", "team1_placeholder": "1J", "team2_placeholder": "2H", "info": "M86, 18h, 3 July | UTC 22h" },
    { "node_id": "R32_14", "id": "M88", "team1_placeholder": "2D", "team2_placeholder": "2G", "info": "M88, 13h, 3 July | UTC 18h" },
    { "node_id": "R32_15", "id": "M85", "team1_placeholder": "1B", "team2_placeholder": "3EFGIJ", "info": "M85, 20h, 2 July | UTC 03h, 3 July" },
    { "node_id": "R32_16", "id": "M87", "team1_placeholder": "1K", "team2_placeholder": "3DEIJL", "info": "M87, 20:30h, 3 July | UTC 01:30h, 4 July" }
  ],
  "knockouts": [
    { "node_id": "R16_1", "id": "M89", "info": "M89, 17h, 4 July | UTC 21h", "depends_on": ["R32_1", "R32_2"] },
    { "node_id": "R16_2", "id": "M90", "info": "M90, 12h, 4 July | UTC 17h", "depends_on": ["R32_3", "R32_4"] },
    { "node_id": "R16_3", "id": "M93", "info": "M93, 14h, 6 July | UTC 19h", "depends_on": ["R32_5", "R32_6"] },
    { "node_id": "R16_4", "id": "M94", "info": "M94, 17h, 6 July | UTC 00h, 7 July", "depends_on": ["R32_7", "R32_8"] },
    { "node_id": "QF_1", "id": "Match 97", "info": "Match 97, 9 July, 16h | UTC 20h", "depends_on": ["R16_1", "R16_2"] },
    { "node_id": "QF_2", "id": "Match 98", "info": "Match 98, 10 July, 12h | UTC 19h", "depends_on": ["R16_3", "R16_4"] },
    { "node_id": "SF_1", "id": "Match 101", "info": "Match 101, 14 July, 14h | UTC 19h", "depends_on": ["QF_1", "QF_2"] },
    { "node_id": "R16_5", "id": "M91", "info": "M91, 18h, 5 July | UTC 01h", "depends_on": ["R32_9", "R32_10"] },
    { "node_id": "R16_6", "id": "M92", "info": "M92, 18h, 5 July | UTC 01h", "depends_on": ["R32_11", "R32_12"] },
    { "node_id": "R16_7", "id": "M95", "info": "M95, 12h, 7 July | UTC 16h", "depends_on": ["R32_13", "R32_14"] },
    { "node_id": "R16_8", "id": "M96", "info": "M96, 13h, 7 July | UTC 20h", "depends_on": ["R32_15", "R32_16"] },
    { "node_id": "QF_3", "id": "Match 99", "info": "Match 99, 11 July, 17h | UTC 21h", "depends_on": ["R16_5", "R16_6"] },
    { "node_id": "QF_4", "id": "Match 100", "info": "Match 100, 11 July, 20h | UTC 01h, 12 July", "depends_on": ["R16_7", "R16_8"] },
    { "node_id": "SF_2", "id": "Match 102", "info": "Match 102, 15 July, 15h | UTC 19h", "depends_on": ["QF_3", "QF_4"] },
    { "node_id": "FINAL", "id": "M104", "info": "M104, 15h, 19 July | UTC 19h", "depends_on": ["SF_1", "SF_2"] },
    { "node_id": "THIRD", "id": "M103", "info": "M103, 15h, 18 July | UTC 19h", "depends_on": ["SF_1", "SF_2"] }
  ]
};

// Unique Team Names to flagCDN code mappings
const teamFlags = {
  "Mexico": "mx", "South Africa": "za", "South Korea": "kr", "Czech Rep.": "cz",
  "Canada": "ca", "Bosnia-Herz.": "ba", "Qatar": "qa", "Switzerland": "ch",
  "Haiti": "ht", "Scotland": "gb-sct", "Brazil": "br", "Morocco": "ma",
  "United States": "us", "Paraguay": "py", "Australia": "au", "Türkiye": "tr",
  "Ivory Coast": "ci", "Ecuador": "ec", "Germany": "de", "Curaçao": "cw",
  "Netherlands": "nl", "Japan": "jp", "Sweden": "se", "Tunisia": "tn",
  "Iran": "ir", "New Zealand": "nz", "Belgium": "be", "Egypt": "eg",
  "Saudi Arabia": "sa", "Uruguay": "uy", "Spain": "es", "Cape Verde": "cv",
  "France": "fr", "Senegal": "sn", "Norway": "no", "Iraq": "iq",
  "Argentina": "ar", "Algeria": "dz", "Austria": "at", "Jordan": "jo",
  "Portugal": "pt", "DR Congo": "cd", "Uzbekistan": "uz", "Colombia": "co",
  "Ghana": "gh", "Panama": "pa", "England": "gb-eng", "Croatia": "hr"
};

// Eligibility map for qualified 3rd-place teams in the Round of 32
const r32_thirdPlace_eligibility = {
  "R32_1": ["A", "B", "C", "D", "F"],
  "R32_2": ["C", "D", "F", "G", "H"],
  "R32_7": ["B", "E", "F", "I", "J"],
  "R32_8": ["A", "E", "H", "I", "J"],
  "R32_11": ["C", "E", "F", "H", "I"],
  "R32_12": ["E", "H", "I", "J", "K"],
  "R32_15": ["E", "F", "G", "I", "J"],
  "R32_16": ["D", "E", "I", "J", "L"]
};

const config = {
  isLocal: window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1" || window.location.protocol === "file:",
  dataPath: "data.json"
};

// Global Predictions and Actual Results Store
let state = {
  users: ["Default User"],
  currentUser: "Default User",
  userScores: {
    "Default User": {}
  },
  scores: {},
  viewMode: "groups",
  activeGroupTab: "GroupA"
};

let globalKoTeams = {};

// Initialize Dashboard
document.addEventListener("DOMContentLoaded", async () => {
  console.log("Dashboard Mode:", config.isLocal ? "LOCAL/EDITOR" : "ONLINE/STATIC");
  await loadInitialState();
  initTabs();
  initUserDropdown();
  
  if (!config.isLocal) {
    disableEditing();
  } else {
    addLocalOnlyUI();
  }
  
  // Set initial active state on view mode buttons
  const btnGroups = document.getElementById("btn-view-groups");
  const btnSingle = document.getElementById("btn-view-single");
  const btnSeq = document.getElementById("btn-view-seq");
  if (btnGroups && btnSingle && btnSeq) {
    btnGroups.classList.remove("active");
    btnSingle.classList.remove("active");
    btnSeq.classList.remove("active");
    
    if (state.viewMode === "sequence") {
      btnSeq.classList.add("active");
    } else if (state.viewMode === "single") {
      btnSingle.classList.add("active");
    } else {
      btnGroups.classList.add("active");
    }
  }

  renderGroupStage();
  renderKnockoutBracket();
  updateScoresAndStandings();
  initActionHandlers();
});

// Load initial state from data.json (online) or local storage (desktop)
async function loadInitialState() {
  let dataJson = null;
  try {
    const response = await fetch(config.dataPath);
    if (response.ok) {
      dataJson = await response.json();
      console.log("Loaded data.json successfully");
    }
  } catch (e) {
    console.log("data.json not found or fetch failed.");
  }

  const hasLocalData = loadStateFromLocalStorage();

  // Logic: 
  // 1. If Online: Prefer data.json (if exists), otherwise local storage.
  // 2. If Local: Prefer local storage (if exists), otherwise data.json.
  
  if (!config.isLocal) {
    if (dataJson) {
      state = dataJson;
      console.log("Online mode: Using data.json as source of truth.");
    } else {
      console.log("Online mode: No data.json found, using local storage.");
    }
  } else {
    if (hasLocalData) {
      console.log("Local mode: Using existing browser local storage.");
    } else if (dataJson) {
      state = dataJson;
      console.log("Local mode: Local storage empty, falling back to data.json.");
    } else {
      console.log("Local mode: No local storage or data.json found, using defaults.");
    }
  }

  if (!state.users) state.users = ["Default User"];
  if (!state.currentUser) state.currentUser = state.users[0];
  if (!state.userScores) state.userScores = { [state.currentUser]: {} };
  state.scores = state.userScores[state.currentUser];
}

function disableEditing() {
  // Disable all number inputs for scores
  const inputs = document.querySelectorAll('input[type="number"]');
  inputs.forEach(input => {
    input.disabled = true;
    input.style.background = "#f0f0f0";
    input.style.cursor = "not-allowed";
  });

  // Hide buttons that modify data
  const toHide = ["reset-btn", "randomize-btn", "import-btn", "edit-user-btn", "delete-user-btn"];
  toHide.forEach(id => {
    const btn = document.getElementById(id);
    if (btn) btn.style.display = "none";
  });
}

function addLocalOnlyUI() {
  const navActions = document.querySelector('.nav-actions');
  if (navActions) {
    const downloadBtn = document.createElement('button');
    downloadBtn.className = "action-btn secondary";
    downloadBtn.innerHTML = '<i class="fa-solid fa-file-arrow-down"></i> Download data.json';
    downloadBtn.title = "Download current state as data.json to update the website";
    downloadBtn.style.background = "#4a90e2";
    downloadBtn.style.color = "white";
    downloadBtn.onclick = () => {
      const tempScores = state.scores;
      delete state.scores;
      const blob = new Blob([JSON.stringify(state, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "data.json";
      a.click();
      state.scores = tempScores;
    };
    navActions.appendChild(downloadBtn);
  }
}

// Load state from local storage
function loadStateFromLocalStorage() {
  const saved = localStorage.getItem("worldcup_2026_state");
  if (saved) {
    try {
      const parsedState = JSON.parse(saved);
      
      // Migration from old single-user state to multi-user state
      if (parsedState.scores && !parsedState.userScores) {
        state.users = ["Default User"];
        state.currentUser = "Default User";
        state.userScores = {
          "Default User": parsedState.scores
        };
      } else {
        state = parsedState;
      }
      return true;
    } catch (e) {
      console.error("Failed to load local storage state", e);
    }
  }
  return false;
}

// Save state to local storage
function saveStateToLocalStorage() {
  if (!config.isLocal) return;
  const tempScores = state.scores;
  delete state.scores; // Avoid duplicating scores in local storage
  localStorage.setItem("worldcup_2026_state", JSON.stringify(state));
  state.scores = tempScores; // Restore reference
}

// User Dropdown Logic
function initUserDropdown() {
  populateUserDropdown();
  const dropdown = document.getElementById("user-dropdown");
  if (dropdown) {
    dropdown.addEventListener("change", (e) => {
      switchUser(e.target.value);
    });
  }
  
  const editBtn = document.getElementById("edit-user-btn");
  if (editBtn) {
    editBtn.addEventListener("click", () => {
      const currentName = state.currentUser;
      const newName = prompt(`Enter new name for "${currentName}":`, currentName);
      if (newName && newName.trim() !== "" && newName !== currentName) {
        if (state.users.includes(newName)) {
          alert("A user with this name already exists.");
          return;
        }
        const idx = state.users.indexOf(currentName);
        if (idx !== -1) state.users[idx] = newName;
        state.userScores[newName] = state.userScores[currentName];
        delete state.userScores[currentName];
        state.currentUser = newName;
        state.scores = state.userScores[newName];
        
        saveStateToLocalStorage();
        populateUserDropdown();
        renderLeaderboard();
      }
    });
  }

  const deleteBtn = document.getElementById("delete-user-btn");
  if (deleteBtn) {
    deleteBtn.addEventListener("click", () => {
      const currentName = state.currentUser;
      if (confirm(`Are you sure you want to delete the user "${currentName}" and all their predictions?`)) {
        state.users = state.users.filter(u => u !== currentName);
        delete state.userScores[currentName];
        
        if (state.users.length === 0) {
          state.users.push("Default User");
          state.userScores["Default User"] = {};
        }
        
        switchUser(state.users[0]);
      }
    });
  }
}

function populateUserDropdown() {
  const dropdown = document.getElementById("user-dropdown");
  if (!dropdown) return;
  dropdown.innerHTML = "";
  state.users.forEach(u => {
    const opt = document.createElement("option");
    opt.value = u;
    opt.textContent = u;
    if (u === state.currentUser) opt.selected = true;
    dropdown.appendChild(opt);
  });
  
  if (config.isLocal) {
    const addOpt = document.createElement("option");
    addOpt.value = "ADD_NEW";
    addOpt.textContent = "+ Add New User";
    dropdown.appendChild(addOpt);
  }
}

function switchUser(userName) {
  if (userName === "ADD_NEW") {
    const newName = prompt("Enter new user name:");
    if (newName && newName.trim() !== "" && !state.users.includes(newName)) {
      state.users.push(newName);
      state.userScores[newName] = {};
      state.currentUser = newName;
    } else {
      // Revert or alert
      document.getElementById('user-dropdown').value = state.currentUser;
      return;
    }
  } else {
    state.currentUser = userName;
  }
  
  if (!state.userScores[state.currentUser]) {
    state.userScores[state.currentUser] = {};
  }
  
  state.scores = state.userScores[state.currentUser];
  if (config.isLocal) {
    saveStateToLocalStorage();
  }
  
  // Re-render
  renderGroupStage();
  renderKnockoutBracket();
  updateScoresAndStandings();
  populateUserDropdown();
}

// FlagCDN URL Generator
function getFlagUrl(teamName) {
  const code = teamFlags[teamName] || "unknown";
  return `https://flagcdn.com/w40/${code}.png`;
}

// ---------------------- TAB CONTROLLER ----------------------
function initTabs() {
  document.querySelectorAll(".nav-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach(pane => pane.classList.remove("active"));
      
      btn.classList.add("active");
      const targetTab = btn.getAttribute("data-tab");
      const pane = document.getElementById(targetTab);
      if (pane) pane.classList.add("active");
      
      // Redraw bracket connectors when bracket tab becomes visible
      if (targetTab === "bracket-tab") {
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            drawBracketConnectors();
          });
        });
      }
    });
  });
}

// ---------------------- DYNAMIC POINTS CALCULATOR ----------------------
function calculatePoints(predHome, predAway, actHome, actAway) {
  if (predHome === null || predAway === null || actHome === null || actAway === null) return 0;
  
  const predWinner = predHome > predAway ? 'H' : (predHome < predAway ? 'A' : 'D');
  const actWinner = actHome > actAway ? 'H' : (actHome < actAway ? 'A' : 'D');
  
  const correctWinner = (predWinner === actWinner);
  const correctScore = (predHome === actHome && predAway === actAway);
  const correctHomeScore = (predHome === actHome);
  const correctAwayScore = (predAway === actAway);
  const correctAtLeastOneScore = (correctHomeScore || correctAwayScore);
  
  // Rule 1: Correct winner and correct score
  if (correctWinner && correctScore) {
    return 5;
  }
  
  // Rule 6: Draw predicted, draw occurred, but scoreline was wrong
  if (actWinner === 'D' && predWinner === 'D' && !correctScore) {
    const totalPredGoals = predHome + predAway;
    const totalActGoals = actHome + actAway;
    const diff = Math.abs(totalPredGoals - totalActGoals);
    if (diff > 0) {
      return 4 / diff;
    }
    return 5;
  }
  
  // Rule 3: Correct winner, and at least one score was correct
  if (correctWinner && correctAtLeastOneScore) {
    return 3;
  }
  
  // Rule 2: Correct winner, but completely incorrect scoreline
  if (correctWinner) {
    return 2;
  }
  
  // Rule 4: Incorrect winner, but one correct score
  if (!correctWinner && correctAtLeastOneScore) {
    return 1;
  }
  
  // Rule 5: Incorrect winner and incorrect scoreline
  return 0;
}

// Determine which scoring rule was matched (for stats breakdown)
function getRuleMatched(predHome, predAway, actHome, actAway) {
  const points = calculatePoints(predHome, predAway, actHome, actAway);
  if (predHome === null || predAway === null || actHome === null || actAway === null) return null;
  
  if (points === 5) return 'rule1';
  if (points === 3) return 'rule3';
  if (points === 2) return 'rule2';
  if (points === 1) return 'rule4';
  if (points > 0 && points < 4 && (predHome === predAway)) return 'rule6';
  return 'rule5'; // 0 points
}

// ---------------------- GROUP STAGE CONTROLLER ----------------------
function renderMiniStandingsCard(groupId) {
  const groupLetter = groupId.replace("Group", "");
  const matches = initialMatchesData.groups[groupId];
  const teams = Array.from(new Set(matches.flatMap(m => [m.team1, m.team2])));
  
  return `
    <div class="mini-standings-card" id="card-${groupId}">
      <div class="group-header">
        <h4>Group ${groupLetter}</h4>
      </div>
      <table class="standings-table" id="standings-${groupId}">
        <thead>
          <tr>
            <th>Team</th>
            <th class="num">P</th>
            <th class="num">GD</th>
            <th class="pts">Pts</th>
          </tr>
        </thead>
        <tbody>
          ${teams.map(t => `
            <tr data-team="${t}">
              <td class="team-name-cell">
                <img src="${getFlagUrl(t)}" alt="">
                <span class="team-name">${t}</span>
              </td>
              <td class="num val-p">0</td>
              <td class="num val-gd">0</td>
              <td class="pts val-pts">0</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderGroupStage() {
  const container = document.getElementById("groups-container");
  if (!container) return;
  container.innerHTML = "";
  
  if (state.viewMode === "groups") {
    container.classList.remove("sequence-view", "single-view");
    
    Object.keys(initialMatchesData.groups).forEach(groupId => {
      const groupName = groupId.replace("Group", "Group ");
      const matches = initialMatchesData.groups[groupId];
      
      // Extract unique teams in group
      const teams = Array.from(new Set(matches.flatMap(m => [m.team1, m.team2])));
      
      const card = document.createElement("div");
      card.className = "group-card";
      card.id = `card-${groupId}`;
      
      let flagsHTML = teams.map(t => `<img src="${getFlagUrl(t)}" alt="${t}">`).join("");
      
      card.innerHTML = `
        <div class="group-header">
          <h4>${groupName}</h4>
          <div class="group-flags">${flagsHTML}</div>
        </div>
        
        <!-- Standing Table -->
        <table class="standings-table" id="standings-${groupId}">
          <thead>
            <tr>
              <th>Team</th>
              <th class="num">P</th>
              <th class="num">GD</th>
              <th class="pts">Pts</th>
            </tr>
          </thead>
          <tbody>
            ${teams.map(t => `
              <tr data-team="${t}">
                <td class="team-name-cell">
                  <img src="${getFlagUrl(t)}" alt="">
                  <span class="team-name">${t}</span>
                </td>
                <td class="num val-p">0</td>
                <td class="num val-gd">0</td>
                <td class="pts val-pts">0</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
        
        <!-- Matches list -->
        <div class="matches-list">
          ${matches.map(m => `
            <div class="match-item" data-match-id="${m.id}">
              <div class="match-meta">
                <span class="match-id-badge">${m.id}</span>
                <span>${m.info}</span>
              </div>
              <div class="match-content-row">
                <div class="team-display home">
                  <span class="team-name">${m.team1}</span>
                  <img src="${getFlagUrl(m.team1)}" alt="">
                </div>
                
                <div class="score-inputs-container">
                  <div class="score-col">
                    <span>Pred</span>
                    <div class="score-box-pair">
                      <input type="number" min="0" placeholder="-" 
                        class="score-input pred pred-home" data-match-id="${m.id}" data-type="predHome"
                        value="${state.scores[m.id + '_predHome'] !== undefined ? state.scores[m.id + '_predHome'] : ''}">
                      <span class="score-divider">-</span>
                      <input type="number" min="0" placeholder="-" 
                        class="score-input pred pred-away" data-match-id="${m.id}" data-type="predAway"
                        value="${state.scores[m.id + '_predAway'] !== undefined ? state.scores[m.id + '_predAway'] : ''}">
                    </div>
                  </div>
                  <div class="score-col">
                    <span>Act</span>
                    <div class="score-box-pair">
                      <input type="number" min="0" placeholder="-" 
                        class="score-input actual act-home" data-match-id="${m.id}" data-type="actHome"
                        value="${state.scores[m.id + '_actHome'] !== undefined ? state.scores[m.id + '_actHome'] : ''}">
                      <span class="score-divider">-</span>
                      <input type="number" min="0" placeholder="-" 
                        class="score-input actual act-away" data-match-id="${m.id}" data-type="actAway"
                        value="${state.scores[m.id + '_actAway'] !== undefined ? state.scores[m.id + '_actAway'] : ''}">
                    </div>
                  </div>
                </div>
                
                <div class="team-display away">
                  <img src="${getFlagUrl(m.team2)}" alt="">
                  <span class="team-name">${m.team2}</span>
                </div>
                
                <div class="points-badge" id="points-${m.id}">-</div>
              </div>
            </div>
          `).join("")}
        </div>
      `;
      
      container.appendChild(card);
    });
  } else if (state.viewMode === "single") {
    container.classList.remove("sequence-view");
    container.classList.add("single-view");
    
    const groupIds = Object.keys(initialMatchesData.groups);
    const subNavHTML = `
      <div class="group-sub-nav">
        ${groupIds.map(gid => {
          const letter = gid.replace("Group", "");
          const activeClass = state.activeGroupTab === gid ? "active" : "";
          return `<button class="sub-nav-btn ${activeClass}" data-group="${gid}">Group ${letter}</button>`;
        }).join("")}
      </div>
    `;
    
    const groupId = state.activeGroupTab;
    const groupName = groupId.replace("Group", "Group ");
    const matches = initialMatchesData.groups[groupId];
    const teams = Array.from(new Set(matches.flatMap(m => [m.team1, m.team2])));
    
    const singleCardHTML = `
      <div class="single-group-card-wrapper">
        <div class="group-card" id="card-${groupId}">
          <div class="group-header">
            <h4>${groupName}</h4>
            <div class="group-flags">
              ${teams.map(t => `<img src="${getFlagUrl(t)}" alt="${t}">`).join("")}
            </div>
          </div>
          
          <!-- Standing Table -->
          <table class="standings-table" id="standings-${groupId}">
            <thead>
              <tr>
                <th>Team</th>
                <th class="num">P</th>
                <th class="num">GD</th>
                <th class="pts">Pts</th>
              </tr>
            </thead>
            <tbody>
              ${teams.map(t => `
                <tr data-team="${t}">
                  <td class="team-name-cell">
                    <img src="${getFlagUrl(t)}" alt="">
                    <span class="team-name">${t}</span>
                  </td>
                  <td class="num val-p">0</td>
                  <td class="num val-gd">0</td>
                  <td class="pts val-pts">0</td>
                </tr>
              `).join("")}
            </tbody>
          </table>
          
          <!-- Matches list -->
          <div class="matches-list">
            ${matches.map(m => `
              <div class="match-item" data-match-id="${m.id}">
                <div class="match-meta">
                  <span class="match-id-badge">${m.id}</span>
                  <span>${m.info}</span>
                </div>
                <div class="match-content-row">
                  <div class="team-display home">
                    <span class="team-name">${m.team1}</span>
                    <img src="${getFlagUrl(m.team1)}" alt="">
                  </div>
                  
                  <div class="score-inputs-container">
                    <div class="score-col">
                      <span>Pred</span>
                      <div class="score-box-pair">
                        <input type="number" min="0" placeholder="-" 
                          class="score-input pred pred-home" data-match-id="${m.id}" data-type="predHome"
                          value="${state.scores[m.id + '_predHome'] !== undefined ? state.scores[m.id + '_predHome'] : ''}">
                        <span class="score-divider">-</span>
                        <input type="number" min="0" placeholder="-" 
                          class="score-input pred pred-away" data-match-id="${m.id}" data-type="predAway"
                          value="${state.scores[m.id + '_predAway'] !== undefined ? state.scores[m.id + '_predAway'] : ''}">
                      </div>
                    </div>
                    <div class="score-col">
                      <span>Act</span>
                      <div class="score-box-pair">
                        <input type="number" min="0" placeholder="-" 
                          class="score-input actual act-home" data-match-id="${m.id}" data-type="actHome"
                          value="${state.scores[m.id + '_actHome'] !== undefined ? state.scores[m.id + '_actHome'] : ''}">
                        <span class="score-divider">-</span>
                        <input type="number" min="0" placeholder="-" 
                          class="score-input actual act-away" data-match-id="${m.id}" data-type="actAway"
                          value="${state.scores[m.id + '_actAway'] !== undefined ? state.scores[m.id + '_actAway'] : ''}">
                      </div>
                    </div>
                  </div>
                  
                  <div class="team-display away">
                    <img src="${getFlagUrl(m.team2)}" alt="">
                    <span class="team-name">${m.team2}</span>
                  </div>
                  
                  <div class="points-badge" id="points-${m.id}">-</div>
                </div>
              </div>
            `).join("")}
          </div>
        </div>
      </div>
    `;
    
    container.innerHTML = `
      <div class="single-group-layout-container">
        ${subNavHTML}
        ${singleCardHTML}
      </div>
    `;
    
    // Add sub-nav click handlers
    document.querySelectorAll(".sub-nav-btn").forEach(btn => {
      btn.addEventListener("click", (e) => {
        state.activeGroupTab = e.currentTarget.getAttribute("data-group");
        saveStateToLocalStorage();
        renderGroupStage();
        updateScoresAndStandings();
      });
    });
  } else {
    container.classList.remove("single-view");
    container.classList.add("sequence-view");
    
    // Flatten and sort group matches
    const allMatches = [];
    Object.keys(initialMatchesData.groups).forEach(groupId => {
      initialMatchesData.groups[groupId].forEach(m => {
        allMatches.push({ ...m, groupId: groupId });
      });
    });
    
    // Sort by match number in ID: M1, M2, M3 etc.
    allMatches.sort((a, b) => {
      const numA = parseInt(a.id.replace(/\D/g, ''));
      const numB = parseInt(b.id.replace(/\D/g, ''));
      return numA - numB;
    });
    
    const leftGroups = ["GroupA", "GroupB", "GroupC", "GroupD", "GroupE", "GroupF"];
    const leftStandingsHTML = leftGroups.map(gid => renderMiniStandingsCard(gid)).join("");
    
    const rightGroups = ["GroupG", "GroupH", "GroupI", "GroupJ", "GroupK", "GroupL"];
    const rightStandingsHTML = rightGroups.map(gid => renderMiniStandingsCard(gid)).join("");
    
    container.innerHTML = `
      <div class="sequence-layout-container">
        <!-- Left Side Standings -->
        <div class="side-standings left-standings">
          ${leftStandingsHTML}
        </div>
        
        <!-- Center Match List -->
        <div class="center-matches">
          <div class="group-card sequence-card">
            <div class="group-header">
              <h4>Group Stage Matches (Chronological Sequence)</h4>
            </div>
            <div class="matches-list">
              ${allMatches.map(m => {
                const groupLetter = m.groupId.replace("Group", "");
                return `
                  <div class="match-item" data-match-id="${m.id}">
                    <div class="match-meta">
                      <span class="match-id-badge">${m.id}</span>
                      <span class="match-group-badge">Group ${groupLetter}</span>
                      <span>${m.info}</span>
                    </div>
                    <div class="match-content-row">
                      <div class="team-display home">
                        <span class="team-name">${m.team1}</span>
                        <img src="${getFlagUrl(m.team1)}" alt="">
                      </div>
                      
                      <div class="score-inputs-container">
                        <div class="score-col">
                          <span>Pred</span>
                          <div class="score-box-pair">
                            <input type="number" min="0" placeholder="-" 
                              class="score-input pred pred-home" data-match-id="${m.id}" data-type="predHome"
                              value="${state.scores[m.id + '_predHome'] !== undefined ? state.scores[m.id + '_predHome'] : ''}">
                            <span class="score-divider">-</span>
                            <input type="number" min="0" placeholder="-" 
                              class="score-input pred pred-away" data-match-id="${m.id}" data-type="predAway"
                              value="${state.scores[m.id + '_predAway'] !== undefined ? state.scores[m.id + '_predAway'] : ''}">
                          </div>
                        </div>
                        <div class="score-col">
                          <span>Act</span>
                          <div class="score-box-pair">
                            <input type="number" min="0" placeholder="-" 
                              class="score-input actual act-home" data-match-id="${m.id}" data-type="actHome"
                              value="${state.scores[m.id + '_actHome'] !== undefined ? state.scores[m.id + '_actHome'] : ''}">
                            <span class="score-divider">-</span>
                            <input type="number" min="0" placeholder="-" 
                              class="score-input actual act-away" data-match-id="${m.id}" data-type="actAway"
                              value="${state.scores[m.id + '_actAway'] !== undefined ? state.scores[m.id + '_actAway'] : ''}">
                          </div>
                        </div>
                      </div>
                      
                      <div class="team-display away">
                        <img src="${getFlagUrl(m.team2)}" alt="">
                        <span class="team-name">${m.team2}</span>
                      </div>
                      
                      <div class="points-badge" id="points-${m.id}">-</div>
                    </div>
                  </div>
                `;
              }).join("")}
            </div>
          </div>
        </div>
        
        <!-- Right Side Standings -->
        <div class="side-standings right-standings">
          ${rightStandingsHTML}
        </div>
      </div>
    `;
  }
  
  // Register Input Event Listeners
  document.querySelectorAll(".score-input").forEach(input => {
    input.addEventListener("input", (e) => {
      const matchId = e.target.getAttribute("data-match-id");
      const type = e.target.getAttribute("data-type");
      const val = e.target.value === "" ? "" : parseInt(e.target.value);
      
      if (val === "" || isNaN(val)) {
        delete state.scores[matchId + "_" + type];
      } else {
        state.scores[matchId + "_" + type] = val;
      }
      
      saveStateToLocalStorage();
      updateScoresAndStandings();
    });
  });
}

// ---------------------- DYNAMIC STANDINGS CALCULATOR ----------------------
function calculateGroupStandings(groupId, matches, type) {
  const teams = Array.from(new Set(matches.flatMap(m => [m.team1, m.team2])));
  const table = {};
  teams.forEach(t => {
    table[t] = { team: t, played: 0, gd: 0, pts: 0, gs: 0 };
  });
  
  matches.forEach(m => {
    const homeVal = state.scores[m.id + "_" + type + "Home"];
    const awayVal = state.scores[m.id + "_" + type + "Away"];
    
    if (homeVal !== undefined && awayVal !== undefined) {
      table[m.team1].played += 1;
      table[m.team2].played += 1;
      table[m.team1].gs += homeVal;
      table[m.team2].gs += awayVal;
      table[m.team1].gd += (homeVal - awayVal);
      table[m.team2].gd += (awayVal - homeVal);
      
      if (homeVal > awayVal) {
        table[m.team1].pts += 3;
      } else if (homeVal < awayVal) {
        table[m.team2].pts += 3;
      } else {
        table[m.team1].pts += 1;
        table[m.team2].pts += 1;
      }
    }
  });
  
  // Sort teams based on: 1. Points, 2. GD, 3. Goals Scored, 4. Alphabetical
  const sorted = Object.values(table).sort((a, b) => {
    if (b.pts !== a.pts) return b.pts - a.pts;
    if (b.gd !== a.gd) return b.gd - a.gd;
    if (b.gs !== a.gs) return b.gs - a.gs;
    return a.team.localeCompare(b.team);
  });
  
  return sorted;
}

// ---------------------- REAL-TIME GRAPHICS UPDATE ----------------------
function updateScoresAndStandings() {
  let totalPoints = 0;
  let ruleCounts = { rule1: 0, rule2: 0, rule3: 0, rule4: 0, rule5: 0, rule6: 0 };
  let predictedMatches = 0;
  
  // 1. Update Group Match Points and Standing Tables
  Object.keys(initialMatchesData.groups).forEach(groupId => {
    const matches = initialMatchesData.groups[groupId];
    
    // Standings calculation for predictions
    const predStandings = calculateGroupStandings(groupId, matches, "pred");
    // Update Standings table in DOM
    const standingsBody = document.querySelector(`#standings-${groupId} tbody`);
    if (standingsBody) {
      standingsBody.innerHTML = predStandings.map(s => `
        <tr data-team="${s.team}">
          <td class="team-name-cell">
            <img src="${getFlagUrl(s.team)}" alt="">
            <span class="team-name">${s.team}</span>
          </td>
          <td class="num">${s.played}</td>
          <td class="num">${s.gd > 0 ? '+' + s.gd : s.gd}</td>
          <td class="pts">${s.pts}</td>
        </tr>
      `).join("");
    }
    
    // Calculate match scores
    matches.forEach(m => {
      const predHome = state.scores[m.id + "_predHome"];
      const predAway = state.scores[m.id + "_predAway"];
      const actHome = state.scores[m.id + "_actHome"];
      const actAway = state.scores[m.id + "_actAway"];
      
      const badge = document.getElementById(`points-${m.id}`);
      if (badge) {
        if (predHome !== undefined && predAway !== undefined && actHome !== undefined && actAway !== undefined) {
          const pts = calculatePoints(predHome, predAway, actHome, actAway);
          totalPoints += pts;
          
          const rule = getRuleMatched(predHome, predAway, actHome, actAway);
          if (rule) ruleCounts[rule] += 1;
          
          badge.textContent = pts.toFixed(pts % 1 === 0 ? 0 : 2);
          // Set color classes
          badge.className = "points-badge";
          if (pts === 5) badge.classList.add("earned-5");
          else if (pts === 3) badge.classList.add("earned-3");
          else if (pts === 2) badge.classList.add("earned-2");
          else if (pts === 1) badge.classList.add("earned-1");
          else if (pts > 0 && pts < 4) badge.classList.add("earned-draw");
          else badge.classList.add("earned-0");
        } else {
          badge.textContent = "-";
          badge.className = "points-badge";
        }
      }
      
      if (predHome !== undefined && predAway !== undefined) {
        predictedMatches += 1;
      }
    });
  });
  
  // 2. Perform Group Stage Rank Promoters to Round of 32
  const groupStandingsMap = {};
  Object.keys(initialMatchesData.groups).forEach(groupId => {
    groupStandingsMap[groupId] = calculateGroupStandings(groupId, initialMatchesData.groups[groupId], "act");
  });
  
  // Calculate 8 best 3rd-place teams
  const allThirdPlaceTeams = Object.keys(groupStandingsMap).map(groupId => {
    const groupLetter = groupId.replace("Group", "");
    const thirdPlace = groupStandingsMap[groupId][2]; // index 2 is 3rd place
    return {
      group: groupLetter,
      ...thirdPlace
    };
  });
  
  // Sort 3rd places: points, GD, GS, group order
  const sortedThirdPlaces = allThirdPlaceTeams.sort((a, b) => {
    if (b.pts !== a.pts) return b.pts - a.pts;
    if (b.gd !== a.gd) return b.gd - a.gd;
    if (b.gs !== a.gs) return b.gs - a.gs;
    return a.group.localeCompare(b.group);
  });
  
  const qualifiedThirds = sortedThirdPlaces.slice(0, 8);
  const assignedThirds = {};
  const remainingThirds = [...qualifiedThirds];
  
  Object.keys(r32_thirdPlace_eligibility).forEach(nodeId => {
    const eligibleGroups = r32_thirdPlace_eligibility[nodeId];
    // Find best eligible team
    const index = remainingThirds.findIndex(t => eligibleGroups.includes(t.group));
    if (index !== -1) {
      assignedThirds[nodeId] = remainingThirds[index].team;
      remainingThirds.splice(index, 1);
    } else {
      assignedThirds[nodeId] = `3${eligibleGroups.join("")}`; // placeholder fallback
    }
  });

  // 3. Resolve Round of 32 Teams
  const r32Teams = {};
  initialMatchesData.r32.forEach(m => {
    let t1 = m.team1_placeholder;
    let t2 = m.team2_placeholder;
    
    // Resolve placeholder values
    const match1 = t1.match(/(\d)([A-L])/);
    if (match1) {
      const rank = parseInt(match1[1]);
      const groupLetter = match1[2];
      const standings = groupStandingsMap[`Group${groupLetter}`];
      t1 = standings ? standings[rank - 1].team : t1;
    }
    
    const match2 = t2.match(/(\d)([A-L])/);
    if (match2) {
      const rank = parseInt(match2[1]);
      const groupLetter = match2[2];
      const standings = groupStandingsMap[`Group${groupLetter}`];
      t2 = standings ? standings[rank - 1].team : t2;
    }
    
    // Resolve 3rd-place placeholders
    if (t2.startsWith("3")) {
      t2 = assignedThirds[m.node_id] || t2;
    }
    
    r32Teams[m.node_id] = { team1: t1, team2: t2 };
  });

  // Calculate and update all Knockout Node fields recursively
  const koTeams = { ...r32Teams };
  globalKoTeams = koTeams;
  
  const koSequence = [
    ...initialMatchesData.r32.map(m => m.node_id),
    "R16_1", "R16_2", "R16_3", "R16_4", "R16_5", "R16_6", "R16_7", "R16_8",
    "QF_1", "QF_2", "QF_3", "QF_4",
    "SF_1", "SF_2",
    "FINAL", "THIRD"
  ];
  
  // Helper to determine winner of a node
  function getWinnerOfNode(nodeId, prefix) {
    const scoresMap = koTeams[nodeId];
    if (!scoresMap) return null;
    
    const home = state.scores[nodeId + "_" + prefix + "Home"];
    const away = state.scores[nodeId + "_" + prefix + "Away"];
    
    if (home === undefined || away === undefined) return null;
    if (home > away) return scoresMap.team1;
    if (home < away) return scoresMap.team2;
    return null;
  }

  // Build dependency lookup map from the knockout data (single source of truth)
  const koDependsOn = {};
  initialMatchesData.knockouts.forEach(m => {
    if (m.depends_on) {
      koDependsOn[m.node_id] = m.depends_on;
    }
  });

  // Helper to determine loser of a node
  function getLoserOfNode(nodeId, prefix) {
    const scoresMap = koTeams[nodeId];
    if (!scoresMap) return null;
    const home = state.scores[nodeId + "_" + prefix + "Home"];
    const away = state.scores[nodeId + "_" + prefix + "Away"];
    if (home === undefined || away === undefined) return null;
    return home > away ? scoresMap.team2 : scoresMap.team1;
  }

  // Populate winners along the tree using depends_on from knockout data
  koSequence.forEach(nodeId => {
    const deps = koDependsOn[nodeId];
    
    // Resolve teams for non-R32 nodes (R32 teams are already resolved above)
    if (deps) {
      if (nodeId === "THIRD") {
        // Third-place match uses losers instead of winners
        koTeams[nodeId] = {
          team1: getLoserOfNode(deps[0], "act") || `Loser ${deps[0]}`,
          team2: getLoserOfNode(deps[1], "act") || `Loser ${deps[1]}`
        };
      } else {
        // All other knockout matches use winners of their dependencies
        koTeams[nodeId] = {
          team1: getWinnerOfNode(deps[0], "act") || `Winner ${deps[0]}`,
          team2: getWinnerOfNode(deps[1], "act") || `Winner ${deps[1]}`
        };
      }
    }
    
    // Update DOM element for this knockout match card (including R32 nodes)
    updateKoMatchDOM(nodeId, koTeams[nodeId]);
    
    // Points calculation for Knockouts
    const predHome = state.scores[nodeId + "_predHome"];
    const predAway = state.scores[nodeId + "_predAway"];
    const actHome = state.scores[nodeId + "_actHome"];
    const actAway = state.scores[nodeId + "_actAway"];
    
    if (predHome !== undefined && predAway !== undefined) {
      predictedMatches += 1;
    }
    
    const badge = document.getElementById(`ko-points-${nodeId}`);
    if (badge) {
      if (predHome !== undefined && predAway !== undefined && actHome !== undefined && actAway !== undefined) {
        const pts = calculatePoints(predHome, predAway, actHome, actAway);
        totalPoints += pts;
        
        const rule = getRuleMatched(predHome, predAway, actHome, actAway);
        if (rule) ruleCounts[rule] += 1;
        
        badge.textContent = pts.toFixed(pts % 1 === 0 ? 0 : 2) + " pts";
      } else {
        badge.textContent = "- pts";
      }
    }
  });

  // 4. Update Header Dashboard Widget
  document.getElementById("total-points").textContent = totalPoints.toFixed(totalPoints % 1 === 0 ? 0 : 2);
  document.getElementById("predicted-count").textContent = `${predictedMatches} / 104`;
  const acc = predictedMatches > 0 ? Math.round((ruleCounts.rule1 + ruleCounts.rule3 + ruleCounts.rule2 + ruleCounts.rule6) / predictedMatches * 100) : 0;
  document.getElementById("prediction-accuracy").textContent = `${acc}%`;
  
  // 5. Update Tab 3 Stats Page counts
  document.getElementById("count-rule1").textContent = ruleCounts.rule1;
  document.getElementById("count-rule3").textContent = ruleCounts.rule3;
  document.getElementById("count-rule2").textContent = ruleCounts.rule2;
  document.getElementById("count-rule6").textContent = ruleCounts.rule6;
  document.getElementById("count-rule4").textContent = ruleCounts.rule4;
  document.getElementById("count-rule5").textContent = ruleCounts.rule5;
  document.getElementById("avg-points").textContent = predictedMatches > 0 ? (totalPoints / predictedMatches).toFixed(2) : "0.00";
  
  renderLeaderboard();
  
  if (!config.isLocal) {
    disableEditing();
  }
}

function calculateUserStats(scoresObj) {
  let totalPoints = 0;
  let ruleCounts = { rule1: 0, rule2: 0, rule3: 0, rule4: 0, rule5: 0, rule6: 0 };
  let predictedMatches = 0;
  
  Object.keys(initialMatchesData.groups).forEach(groupId => {
    initialMatchesData.groups[groupId].forEach(m => {
      const predHome = scoresObj[m.id + "_predHome"];
      const predAway = scoresObj[m.id + "_predAway"];
      const actHome = scoresObj[m.id + "_actHome"];
      const actAway = scoresObj[m.id + "_actAway"];
      
      if (predHome !== undefined && predAway !== undefined) {
        predictedMatches += 1;
      }
      if (predHome !== undefined && predAway !== undefined && actHome !== undefined && actAway !== undefined) {
        const pts = calculatePoints(predHome, predAway, actHome, actAway);
        totalPoints += pts;
        const rule = getRuleMatched(predHome, predAway, actHome, actAway);
        if (rule) ruleCounts[rule] += 1;
      }
    });
  });
  
  const koMatches = [...initialMatchesData.r32, ...initialMatchesData.knockouts];
  koMatches.forEach(m => {
    const nodeId = m.node_id;
    const predHome = scoresObj[nodeId + "_predHome"];
    const predAway = scoresObj[nodeId + "_predAway"];
    const actHome = scoresObj[nodeId + "_actHome"];
    const actAway = scoresObj[nodeId + "_actAway"];
    
    if (predHome !== undefined && predAway !== undefined) {
      predictedMatches += 1;
    }
    if (predHome !== undefined && predAway !== undefined && actHome !== undefined && actAway !== undefined) {
      const pts = calculatePoints(predHome, predAway, actHome, actAway);
      totalPoints += pts;
      const rule = getRuleMatched(predHome, predAway, actHome, actAway);
      if (rule) ruleCounts[rule] += 1;
    }
  });
  
  const acc = predictedMatches > 0 ? Math.round((ruleCounts.rule1 + ruleCounts.rule3 + ruleCounts.rule2 + ruleCounts.rule6) / predictedMatches * 100) : 0;
  return { totalPoints, predictedMatches, acc, ruleCounts };
}

function renderLeaderboard() {
  const tbody = document.getElementById("leaderboard-body");
  if (!tbody) return;
  
  const userStats = [];
  state.users.forEach(u => {
    const stats = calculateUserStats(state.userScores[u] || {});
    userStats.push({ name: u, ...stats });
  });
  
  userStats.sort((a, b) => {
    if (b.totalPoints !== a.totalPoints) return b.totalPoints - a.totalPoints;
    return a.name.localeCompare(b.name);
  });
  
  tbody.innerHTML = userStats.map((u, idx) => `
    <tr>
      <td style="text-align: center;"><strong>${idx + 1}</strong></td>
      <td><strong>${u.name}</strong>${u.name === state.currentUser ? ' <span style="font-size: 0.8rem; color: #888;">(You)</span>' : ''}</td>
      <td class="num" style="font-weight: 700; color: #e9bc3f;">${u.totalPoints.toFixed(u.totalPoints % 1 === 0 ? 0 : 2)}</td>
      <td class="num">${u.acc}%</td>
      <td class="num">${u.predictedMatches} / 104</td>
    </tr>
  `).join("");
}

// ---------------------- BRACKET DOM INJECTION ----------------------
function renderKnockoutBracket() {
  document.getElementById("r32-left-list").innerHTML = "";
  document.getElementById("r16-left-list").innerHTML = "";
  document.getElementById("qf-left-list").innerHTML = "";
  document.getElementById("sf-left-list").innerHTML = "";
  document.getElementById("finals-list").innerHTML = "";
  document.getElementById("sf-right-list").innerHTML = "";
  document.getElementById("qf-right-list").innerHTML = "";
  document.getElementById("r16-right-list").innerHTML = "";
  document.getElementById("r32-right-list").innerHTML = "";
  
  // Render Round of 32 Left
  for (let i = 1; i <= 8; i++) {
    const m = initialMatchesData.r32[i - 1];
    createKoMatchCard(m, "r32-left-list");
  }
  // Render Round of 32 Right
  for (let i = 9; i <= 16; i++) {
    const m = initialMatchesData.r32[i - 1];
    createKoMatchCard(m, "r32-right-list");
  }
  
  // Render other round placeholders
  initialMatchesData.knockouts.forEach(m => {
    let listId = "";
    if (m.node_id.startsWith("R16_")) {
      const index = parseInt(m.node_id.split("_")[1]);
      listId = index <= 4 ? "r16-left-list" : "r16-right-list";
    } else if (m.node_id.startsWith("QF_")) {
      const index = parseInt(m.node_id.split("_")[1]);
      listId = index <= 2 ? "qf-left-list" : "qf-right-list";
    } else if (m.node_id.startsWith("SF_")) {
      const index = parseInt(m.node_id.split("_")[1]);
      listId = index === 1 ? "sf-left-list" : "sf-right-list";
    } else {
      listId = "finals-list";
    }
    createKoMatchCard(m, listId);
  });
  
  // Register Input Event Listeners for Knockouts
  document.querySelectorAll(".ko-score-input").forEach(input => {
    input.addEventListener("input", (e) => {
      const nodeId = e.target.getAttribute("data-node-id");
      const type = e.target.getAttribute("data-type");
      const val = e.target.value === "" ? "" : parseInt(e.target.value);
      
      if (val === "" || isNaN(val)) {
        delete state.scores[nodeId + "_" + type];
      } else {
        state.scores[nodeId + "_" + type] = val;
      }
      
      saveStateToLocalStorage();
      updateScoresAndStandings();
    });
  });
  
  // Draw connectors after DOM layout settles
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      drawBracketConnectors();
    });
  });
}

// ---------------------- BRACKET CONNECTOR LINES ----------------------
function drawBracketConnectors() {
  const container = document.getElementById("bracket-container");
  if (!container) return;
  
  // Remove old SVG if it exists
  const oldSvg = container.querySelector(".bracket-connectors-svg");
  if (oldSvg) oldSvg.remove();
  
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.classList.add("bracket-connectors-svg");
  svg.setAttribute("width", container.scrollWidth);
  svg.setAttribute("height", container.scrollHeight);
  
  // Define arrowhead marker
  const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
  
  // Right-pointing arrow (for left bracket)
  const markerRight = document.createElementNS("http://www.w3.org/2000/svg", "marker");
  markerRight.setAttribute("id", "arrow-right");
  markerRight.setAttribute("markerWidth", "8");
  markerRight.setAttribute("markerHeight", "6");
  markerRight.setAttribute("refX", "8");
  markerRight.setAttribute("refY", "3");
  markerRight.setAttribute("orient", "auto");
  const pathRight = document.createElementNS("http://www.w3.org/2000/svg", "path");
  pathRight.setAttribute("d", "M0,0 L8,3 L0,6 Z");
  pathRight.setAttribute("fill", "rgba(212,175,55,0.5)");
  markerRight.appendChild(pathRight);
  defs.appendChild(markerRight);
  
  // Left-pointing arrow (for right bracket)
  const markerLeft = document.createElementNS("http://www.w3.org/2000/svg", "marker");
  markerLeft.setAttribute("id", "arrow-left");
  markerLeft.setAttribute("markerWidth", "8");
  markerLeft.setAttribute("markerHeight", "6");
  markerLeft.setAttribute("refX", "0");
  markerLeft.setAttribute("refY", "3");
  markerLeft.setAttribute("orient", "auto");
  const pathLeft = document.createElementNS("http://www.w3.org/2000/svg", "path");
  pathLeft.setAttribute("d", "M8,0 L0,3 L8,6 Z");
  pathLeft.setAttribute("fill", "rgba(212,175,55,0.5)");
  markerLeft.appendChild(pathLeft);
  defs.appendChild(markerLeft);
  
  svg.appendChild(defs);
  
  const containerRect = container.getBoundingClientRect();
  
  // Helper to get card center position relative to container
  function getCardPos(nodeId) {
    const card = document.getElementById(`ko-card-${nodeId}`);
    if (!card) return null;
    const rect = card.getBoundingClientRect();
    return {
      left: rect.left - containerRect.left,
      right: rect.right - containerRect.left,
      top: rect.top - containerRect.top,
      bottom: rect.bottom - containerRect.top,
      centerY: (rect.top + rect.bottom) / 2 - containerRect.top,
      centerX: (rect.left + rect.right) / 2 - containerRect.left
    };
  }
  
  // Determine if a node is on the left or right side of the bracket
  function isLeftSide(nodeId) {
    if (nodeId.startsWith("R32_")) {
      const idx = parseInt(nodeId.split("_")[1]);
      return idx <= 8;
    }
    if (nodeId.startsWith("R16_")) {
      const idx = parseInt(nodeId.split("_")[1]);
      return idx <= 4;
    }
    if (nodeId.startsWith("QF_")) {
      const idx = parseInt(nodeId.split("_")[1]);
      return idx <= 2;
    }
    if (nodeId === "SF_1") return true;
    if (nodeId === "SF_2") return false;
    return null; // Finals/Third - center
  }
  
  // Draw a connector from two feeder nodes into one target node
  function drawConnector(feeder1Id, feeder2Id, targetId) {
    const f1 = getCardPos(feeder1Id);
    const f2 = getCardPos(feeder2Id);
    const t = getCardPos(targetId);
    if (!f1 || !f2 || !t) return;
    
    const leftSide = isLeftSide(feeder1Id);
    const lineColor = "rgba(212,175,55,0.35)";
    const lineWidth = 1.5;
    
    if (leftSide === true) {
      // Left bracket: lines go RIGHT
      const startX = f1.right + 2;
      const midX = (f1.right + t.left) / 2;
      const endX = t.left - 2;
      
      // Feeder 1 → merge point
      const path1 = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path1.setAttribute("d", `M${startX},${f1.centerY} H${midX} V${t.centerY} H${endX}`);
      path1.setAttribute("stroke", lineColor);
      path1.setAttribute("stroke-width", lineWidth);
      path1.setAttribute("fill", "none");
      path1.setAttribute("marker-end", "url(#arrow-right)");
      svg.appendChild(path1);
      
      // Feeder 2 → merge point
      const path2 = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path2.setAttribute("d", `M${startX},${f2.centerY} H${midX} V${t.centerY}`);
      path2.setAttribute("stroke", lineColor);
      path2.setAttribute("stroke-width", lineWidth);
      path2.setAttribute("fill", "none");
      svg.appendChild(path2);
      
    } else if (leftSide === false) {
      // Right bracket: lines go LEFT
      const startX = f1.left - 2;
      const midX = (f1.left + t.right) / 2;
      const endX = t.right + 2;
      
      // Feeder 1 → merge point
      const path1 = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path1.setAttribute("d", `M${startX},${f1.centerY} H${midX} V${t.centerY} H${endX}`);
      path1.setAttribute("stroke", lineColor);
      path1.setAttribute("stroke-width", lineWidth);
      path1.setAttribute("fill", "none");
      path1.setAttribute("marker-end", "url(#arrow-left)");
      svg.appendChild(path1);
      
      // Feeder 2 → merge point
      const path2 = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path2.setAttribute("d", `M${startX},${f2.centerY} H${midX} V${t.centerY}`);
      path2.setAttribute("stroke", lineColor);
      path2.setAttribute("stroke-width", lineWidth);
      path2.setAttribute("fill", "none");
      svg.appendChild(path2);
      
    } else {
      // Center (Finals) - draw from both sides
      // SF_1 (left) feeds into Final from left
      const sf1Pos = getCardPos(feeder1Id);
      const sf2Pos = getCardPos(feeder2Id);
      if (!sf1Pos || !sf2Pos) return;
      
      const path1 = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path1.setAttribute("d", `M${sf1Pos.right + 2},${sf1Pos.centerY} H${t.left - 2}`);
      path1.setAttribute("stroke", lineColor);
      path1.setAttribute("stroke-width", lineWidth);
      path1.setAttribute("fill", "none");
      path1.setAttribute("marker-end", "url(#arrow-right)");
      svg.appendChild(path1);
      
      const path2 = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path2.setAttribute("d", `M${sf2Pos.left - 2},${sf2Pos.centerY} H${t.right + 2}`);
      path2.setAttribute("stroke", lineColor);
      path2.setAttribute("stroke-width", lineWidth);
      path2.setAttribute("fill", "none");
      path2.setAttribute("marker-end", "url(#arrow-left)");
      svg.appendChild(path2);
    }
  }
  
  // Draw all connections using the depends_on data
  initialMatchesData.knockouts.forEach(m => {
    if (m.depends_on && m.depends_on.length === 2) {
      drawConnector(m.depends_on[0], m.depends_on[1], m.node_id);
    }
  });
  
  container.appendChild(svg);
}

function createKoMatchCard(m, listId) {
  const container = document.getElementById(listId);
  if (!container) return;
  const card = document.createElement("div");
  card.className = "ko-match-card";
  card.id = `ko-card-${m.node_id}`;
  
  card.innerHTML = `
    <div class="ko-match-header">
      <span>${m.id}</span>
      <span>${m.info.split("|")[0]}</span>
    </div>
    
    <!-- Team 1 -->
    <div class="ko-match-team-row" id="ko-row-${m.node_id}-team1">
      <div class="ko-team-info" id="ko-info-${m.node_id}-team1">
        <span class="ko-placeholder-text">Pending...</span>
      </div>
      <div class="ko-score-inputs">
        <input type="number" min="0" placeholder="P" class="ko-score-input pred ko-pred-home" 
          data-node-id="${m.node_id}" data-type="predHome"
          value="${state.scores[m.node_id + '_predHome'] !== undefined ? state.scores[m.node_id + '_predHome'] : ''}">
        <input type="number" min="0" placeholder="A" class="ko-score-input actual ko-act-home" 
          data-node-id="${m.node_id}" data-type="actHome"
          value="${state.scores[m.node_id + '_actHome'] !== undefined ? state.scores[m.node_id + '_actHome'] : ''}">
      </div>
    </div>
    
    <!-- Team 2 -->
    <div class="ko-match-team-row" id="ko-row-${m.node_id}-team2">
      <div class="ko-team-info" id="ko-info-${m.node_id}-team2">
        <span class="ko-placeholder-text">Pending...</span>
      </div>
      <div class="ko-score-inputs">
        <input type="number" min="0" placeholder="P" class="ko-score-input pred ko-pred-away" 
          data-node-id="${m.node_id}" data-type="predAway"
          value="${state.scores[m.node_id + '_predAway'] !== undefined ? state.scores[m.node_id + '_predAway'] : ''}">
        <input type="number" min="0" placeholder="A" class="ko-score-input actual ko-act-away" 
          data-node-id="${m.node_id}" data-type="actAway"
          value="${state.scores[m.node_id + '_actAway'] !== undefined ? state.scores[m.node_id + '_actAway'] : ''}">
      </div>
    </div>
    
    <div class="ko-points-footer">
      <span class="match-info-txt">${m.info.includes("|") ? m.info.split("|")[1].trim() : ''}</span>
      <span class="ko-points-badge" id="ko-points-${m.node_id}">- pts</span>
    </div>
  `;
  container.appendChild(card);
}

function updateKoMatchDOM(nodeId, teamsObj) {
  const team1Info = document.getElementById(`ko-info-${nodeId}-team1`);
  const team2Info = document.getElementById(`ko-info-${nodeId}-team2`);
  
  if (team1Info && team2Info) {
    // Team 1
    const t1 = teamsObj.team1;
    const isPlaceholder1 = t1.startsWith("Winner") || t1.startsWith("Loser") || t1.match(/^\d/) || t1.startsWith("3");
    if (isPlaceholder1) {
      team1Info.innerHTML = `<span class="ko-placeholder-text">${t1}</span>`;
    } else {
      team1Info.innerHTML = `
        <img src="${getFlagUrl(t1)}" alt="">
        <span class="team-name">${t1}</span>
      `;
    }
    
    // Team 2
    const t2 = teamsObj.team2;
    const isPlaceholder2 = t2.startsWith("Winner") || t2.startsWith("Loser") || t2.match(/^\d/) || t2.startsWith("3");
    if (isPlaceholder2) {
      team2Info.innerHTML = `<span class="ko-placeholder-text">${t2}</span>`;
    } else {
      team2Info.innerHTML = `
        <img src="${getFlagUrl(t2)}" alt="">
        <span class="team-name">${t2}</span>
      `;
    }
    
    // Highlight predicted winners
    const row1 = document.getElementById(`ko-row-${nodeId}-team1`);
    const row2 = document.getElementById(`ko-row-${nodeId}-team2`);
    if (row1 && row2) {
      row1.classList.remove("winner-predicted");
      row2.classList.remove("winner-predicted");
      
      const predHome = state.scores[nodeId + "_predHome"];
      const predAway = state.scores[nodeId + "_predAway"];
      if (predHome !== undefined && predAway !== undefined) {
        if (predHome > predAway) row1.classList.add("winner-predicted");
        else if (predHome < predAway) row2.classList.add("winner-predicted");
      }
    }
  }
}

// ---------------------- LEADERBOARD IMPORT/EXPORT CONTROLLER ----------------------
function initActionHandlers() {
  const btnGroups = document.getElementById("btn-view-groups");
  const btnSingle = document.getElementById("btn-view-single");
  const btnSeq = document.getElementById("btn-view-seq");
  
  const setBtnActive = (activeBtn) => {
    [btnGroups, btnSingle, btnSeq].forEach(btn => {
      if (btn) btn.classList.remove("active");
    });
    if (activeBtn) activeBtn.classList.add("active");
  };
  
  if (btnGroups) {
    btnGroups.addEventListener("click", () => {
      if (state.viewMode !== 'groups') {
        state.viewMode = 'groups';
        setBtnActive(btnGroups);
        saveStateToLocalStorage();
        renderGroupStage();
        updateScoresAndStandings();
      }
    });
  }
  
  if (btnSingle) {
    btnSingle.addEventListener("click", () => {
      if (state.viewMode !== 'single') {
        state.viewMode = 'single';
        setBtnActive(btnSingle);
        saveStateToLocalStorage();
        renderGroupStage();
        updateScoresAndStandings();
      }
    });
  }
  
  if (btnSeq) {
    btnSeq.addEventListener("click", () => {
      if (state.viewMode !== 'sequence') {
        state.viewMode = 'sequence';
        setBtnActive(btnSeq);
        saveStateToLocalStorage();
        renderGroupStage();
        updateScoresAndStandings();
      }
    });
  }

  // Reset Scores Handler
  document.getElementById("reset-btn").addEventListener("click", () => {
    if (confirm("Are you sure you want to delete all predictions and actual scores? This cannot be undone.")) {
      state.userScores[state.currentUser] = {};
      state.scores = state.userScores[state.currentUser];
      saveStateToLocalStorage();
      // Re-render
      renderGroupStage();
      renderKnockoutBracket();
      updateScoresAndStandings();
    }
  });
  
  // Randomize Scores Handler
  document.getElementById("randomize-btn").addEventListener("click", () => {
    if (confirm("This will overwrite ALL predictions and actual results with random scores (0-10). Continue?")) {
      const randomScore = () => Math.floor(Math.random() * 11); // 0 to 10
      
      // Randomize all group stage matches
      Object.keys(initialMatchesData.groups).forEach(groupId => {
        initialMatchesData.groups[groupId].forEach(m => {
          state.scores[m.id + "_predHome"] = randomScore();
          state.scores[m.id + "_predAway"] = randomScore();
          state.scores[m.id + "_actHome"] = randomScore();
          state.scores[m.id + "_actAway"] = randomScore();
        });
      });
      
      // Randomize all R32 knockout matches
      initialMatchesData.r32.forEach(m => {
        let h = randomScore(), a = randomScore();
        // Knockout matches can't draw — re-roll if equal
        while (h === a) { h = randomScore(); a = randomScore(); }
        state.scores[m.node_id + "_predHome"] = h;
        state.scores[m.node_id + "_predAway"] = a;
        h = randomScore(); a = randomScore();
        while (h === a) { h = randomScore(); a = randomScore(); }
        state.scores[m.node_id + "_actHome"] = h;
        state.scores[m.node_id + "_actAway"] = a;
      });
      
      // Randomize all other knockout matches (R16, QF, SF, Final, Third)
      initialMatchesData.knockouts.forEach(m => {
        let h = randomScore(), a = randomScore();
        while (h === a) { h = randomScore(); a = randomScore(); }
        state.scores[m.node_id + "_predHome"] = h;
        state.scores[m.node_id + "_predAway"] = a;
        h = randomScore(); a = randomScore();
        while (h === a) { h = randomScore(); a = randomScore(); }
        state.scores[m.node_id + "_actHome"] = h;
        state.scores[m.node_id + "_actAway"] = a;
      });
      
      saveStateToLocalStorage();
      // Re-render everything
      renderGroupStage();
      renderKnockoutBracket();
      updateScoresAndStandings();
    }
  });
  // Export CSV Handler
  document.getElementById("export-csv-btn").addEventListener("click", () => {
    let csvContent = "Date,Group,Team 1,Predicted Score for team 1,Team 2,Predicted score for team 2\n";
    
    Object.keys(initialMatchesData.groups).forEach(groupId => {
      const groupName = groupId.replace("Group", "Group ");
      initialMatchesData.groups[groupId].forEach(m => {
        let date = m.info;
        if (date.includes(",")) {
          const parts = date.split(",");
          if (parts.length >= 3) {
            date = parts[2].split("|")[0].trim();
          }
        }
        
        const predHome = state.scores[m.id + "_predHome"] !== undefined ? state.scores[m.id + "_predHome"] : "";
        const predAway = state.scores[m.id + "_predAway"] !== undefined ? state.scores[m.id + "_predAway"] : "";
        
        csvContent += `"${date}","${groupName}","${m.team1}","${predHome}","${m.team2}","${predAway}"\n`;
      });
    });
    
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `worldcup_predictions_${state.currentUser.replace(/\s+/g, '_')}.csv`);
    link.click();
  });
  
  // Export Handler
  document.getElementById("export-btn").addEventListener("click", () => {
    const tempScores = state.scores;
    delete state.scores; // Avoid duplicating scores
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(state));
    state.scores = tempScores; // Restore reference
    const dlAnchorElem = document.createElement('a');
    dlAnchorElem.setAttribute("href",     dataStr     );
    dlAnchorElem.setAttribute("download", "worldcup_2026_predictions.json");
    dlAnchorElem.click();
  });
  
  // Import Handlers
  const fileInput = document.getElementById("import-file-input");
  document.getElementById("import-btn").addEventListener("click", () => {
    fileInput.click();
  });
  
  fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const imported = JSON.parse(event.target.result);
        if (imported) {
          if (imported.userScores) {
            state.users = imported.users || ["Default User"];
            state.currentUser = imported.currentUser || "Default User";
            state.userScores = imported.userScores;
            state.scores = state.userScores[state.currentUser];
          } else {
            // Legacy import
            state.userScores[state.currentUser] = imported.scores || {};
            state.scores = state.userScores[state.currentUser];
          }
          state.viewMode = imported.viewMode || 'groups';
          state.activeGroupTab = imported.activeGroupTab || 'GroupA';
          saveStateToLocalStorage();
          populateUserDropdown();
          if (btnGroups && btnSingle && btnSeq) {
            btnGroups.classList.remove("active");
            btnSingle.classList.remove("active");
            btnSeq.classList.remove("active");
            
            if (state.viewMode === 'sequence') {
              btnSeq.classList.add("active");
            } else if (state.viewMode === 'single') {
              btnSingle.classList.add("active");
            } else {
              btnGroups.classList.add("active");
            }
          }
          
          // Re-render and recalculate
          renderGroupStage();
          renderKnockoutBracket();
          updateScoresAndStandings();
          alert("Predictions successfully imported!");
        } else {
          alert("Invalid file structure. Make sure this is a valid predictions export file.");
        }
      } catch (err) {
        alert("Failed to parse file. Make sure the file is in JSON format.");
      }
    };
    reader.readAsText(file);
  });
}

// Register Modal Close Events
document.addEventListener("DOMContentLoaded", () => {
  const modalClose = document.getElementById("modal-close-btn");
  const modal = document.getElementById("country-modal");
  
  if (modalClose && modal) {
    modalClose.addEventListener("click", () => {
      modal.classList.remove("active");
    });
    modal.addEventListener("click", (e) => {
      if (e.target === modal) {
        modal.classList.remove("active");
      }
    });
  }
});

// Click listener on team names
document.addEventListener("click", (e) => {
  const teamNameSpan = e.target.closest(".team-name");
  if (teamNameSpan) {
    const countryName = teamNameSpan.textContent.trim();
    // Verify it's a real country (not placeholder like "Winner R32_1" or "Pending...")
    const isPlaceholder = countryName.startsWith("Winner") || 
                          countryName.startsWith("Loser") || 
                          countryName.match(/^\d/) || 
                          countryName.startsWith("3") || 
                          countryName.startsWith("Pending") ||
                          countryName.includes("Winner") ||
                          countryName.includes("Loser");
    if (!isPlaceholder && teamFlags[countryName]) {
      showCountryDetailsModal(countryName);
    }
  }
});

// Show country details inside modal overlay
function showCountryDetailsModal(country) {
  // Update modal header
  const titleElem = document.getElementById("modal-country-name");
  const flagElem = document.getElementById("modal-country-flag");
  if (titleElem) titleElem.textContent = country;
  if (flagElem) flagElem.src = getFlagUrl(country);
  
  // Find all matches for this country
  const countryMatches = [];
  
  // 1. Check Group Stage matches
  Object.keys(initialMatchesData.groups).forEach(groupId => {
    initialMatchesData.groups[groupId].forEach(m => {
      if (m.team1 === country || m.team2 === country) {
        countryMatches.push({
          id: m.id,
          type: "Group Stage",
          team1: m.team1,
          team2: m.team2,
          info: m.info
        });
      }
    });
  });
  
  // 2. Check Knockout matches
  const koSequence = [
    ...initialMatchesData.r32.map(m => ({ node_id: m.node_id, id: m.id, info: m.info, type: "Round of 32" })),
    ...initialMatchesData.knockouts.map(m => {
      let type = "Knockout";
      if (m.node_id.startsWith("R16_")) type = "Round of 16";
      else if (m.node_id.startsWith("QF_")) type = "Quarter-Final";
      else if (m.node_id.startsWith("SF_")) type = "Semi-Final";
      else if (m.node_id === "FINAL") type = "Final";
      else if (m.node_id === "THIRD") type = "Third Place Match";
      return { node_id: m.node_id, id: m.id, info: m.info, type: type };
    })
  ];
  
  koSequence.forEach(k => {
    const resolved = globalKoTeams[k.node_id];
    if (resolved && (resolved.team1 === country || resolved.team2 === country)) {
      countryMatches.push({
        id: k.id,
        node_id: k.node_id,
        type: k.type,
        team1: resolved.team1,
        team2: resolved.team2,
        info: k.info
      });
    }
  });
  
  // Build Matches HTML
  const container = document.getElementById("modal-matches-container");
  if (!container) return;
  container.innerHTML = "";
  
  let totalPointsEarned = 0;
  let wins = 0, draws = 0, losses = 0;
  
  countryMatches.forEach(m => {
    const keyId = m.node_id || m.id;
    const predHome = state.scores[keyId + "_predHome"];
    const predAway = state.scores[keyId + "_predAway"];
    const actHome = state.scores[keyId + "_actHome"];
    const actAway = state.scores[keyId + "_actAway"];
    
    let pointsHTML = "-";
    let scoreDisplayHTML = "";
    
    const hasPredictions = (predHome !== undefined && predAway !== undefined);
    const hasActuals = (actHome !== undefined && actAway !== undefined);
    
    if (hasPredictions) {
      scoreDisplayHTML += `<span class="score-lbl">Pred:</span> <strong>${predHome} - ${predAway}</strong>`;
      
      const isTeam1 = (m.team1 === country);
      if (predHome === predAway) {
        draws++;
      } else if ((predHome > predAway && isTeam1) || (predHome < predAway && !isTeam1)) {
        wins++;
      } else {
        losses++;
      }
    } else {
      scoreDisplayHTML += `<span class="score-lbl">Pred:</span> <em>Not entered</em>`;
    }
    
    if (hasActuals) {
      scoreDisplayHTML += `<br><span class="score-lbl">Act:</span> <strong>${actHome} - ${actAway}</strong>`;
    } else {
      scoreDisplayHTML += `<br><span class="score-lbl">Act:</span> <em>Pending</em>`;
    }
    
    if (hasPredictions && hasActuals) {
      const pts = calculatePoints(predHome, predAway, actHome, actAway);
      totalPointsEarned += pts;
      pointsHTML = `${pts.toFixed(pts % 1 === 0 ? 0 : 2)} pts`;
    }
    
    const item = document.createElement("div");
    item.className = "modal-match-item";
    
    item.innerHTML = `
      <div class="m-meta">
        <span class="m-type-badge">${m.type}</span>
        <span class="m-id-badge">${m.id}</span>
      </div>
      <div class="m-row">
        <div class="m-team home ${m.team1 === country ? 'active-country' : ''}">
          <span class="team-name">${m.team1}</span>
          <img src="${getFlagUrl(m.team1)}" alt="">
        </div>
        <div class="m-scores">
          ${scoreDisplayHTML}
        </div>
        <div class="m-team away ${m.team2 === country ? 'active-country' : ''}">
          <img src="${getFlagUrl(m.team2)}" alt="">
          <span class="team-name">${m.team2}</span>
        </div>
        <div class="m-points">${pointsHTML}</div>
      </div>
    `;
    
    container.appendChild(item);
  });
  
  // Update stats summary in DOM
  const recordElem = document.getElementById("modal-pred-record");
  const pointsElem = document.getElementById("modal-country-points");
  if (recordElem) recordElem.textContent = `${wins}W - ${draws}D - ${losses}L`;
  if (pointsElem) pointsElem.textContent = totalPointsEarned.toFixed(totalPointsEarned % 1 === 0 ? 0 : 2);
  
  // Show modal
  const modal = document.getElementById("country-modal");
  if (modal) modal.classList.add("active");
}
