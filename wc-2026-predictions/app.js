// Embedded Matches Database (extracted from main.tex)
const initialMatchesData = {
  "groups": {
    "GroupA": [
      {
        "id": "M1",
        "team1": "Mexico",
        "team2": "South Africa",
        "info": "M1, 15:00, 11 June | UTC 19:00"
      },
      {
        "id": "M2",
        "team1": "South Korea",
        "team2": "Czech Rep.",
        "info": "M2, 15:00, 11 June | UTC 19:00"
      },
      {
        "id": "M25",
        "team1": "Czech Rep.",
        "team2": "South Africa",
        "info": "M25, 12:00, 18 June | UTC 16:00"
      },
      {
        "id": "M28",
        "team1": "Mexico",
        "team2": "South Korea",
        "info": "M28, 21:00, 18 June | UTC 01:00, 19 June"
      },
      {
        "id": "M53",
        "team1": "Czech Rep.",
        "team2": "Mexico",
        "info": "M53, 21:00, 24 June | UTC 01:00, 25 June"
      },
      {
        "id": "M54",
        "team1": "South Africa",
        "team2": "South Korea",
        "info": "M54, 21:00, 24 June | UTC 01:00, 25 June"
      }
    ],
    "GroupB": [
      {
        "id": "M3",
        "team1": "Canada",
        "team2": "Bosnia-Herz.",
        "info": "M3, 15:00, 12 June | UTC 19:00"
      },
      {
        "id": "M5",
        "team1": "Qatar",
        "team2": "Switzerland",
        "info": "M5, 15:00, 13 June | UTC 19:00"
      },
      {
        "id": "M26",
        "team1": "Switzerland",
        "team2": "Bosnia-Herz.",
        "info": "M26, 15:00, 18 June | UTC 19:00"
      },
      {
        "id": "M27",
        "team1": "Canada",
        "team2": "Qatar",
        "info": "M27, 18:00, 18 June | UTC 22:00"
      },
      {
        "id": "M49",
        "team1": "Switzerland",
        "team2": "Canada",
        "info": "M49, 15:00, 24 June | UTC 19:00"
      },
      {
        "id": "M50",
        "team1": "Bosnia-Herz.",
        "team2": "Qatar",
        "info": "M50, 15:00, 24 June | UTC 19:00"
      }
    ],
    "GroupC": [
      {
        "id": "M6",
        "team1": "Brazil",
        "team2": "Morocco",
        "info": "M6, 18:00, 13 June | UTC 22:00"
      },
      {
        "id": "M7",
        "team1": "Haiti",
        "team2": "Scotland",
        "info": "M7, 21:00, 13 June | UTC 01:00, 14 June"
      },
      {
        "id": "M30",
        "team1": "Scotland",
        "team2": "Morocco",
        "info": "M30, 18:00, 19 June | UTC 22:00"
      },
      {
        "id": "M31",
        "team1": "Brazil",
        "team2": "Haiti",
        "info": "M31, 20:30, 19 June | UTC 00:30, 20 June"
      },
      {
        "id": "M51",
        "team1": "Scotland",
        "team2": "Brazil",
        "info": "M51, 18:00, 24 June | UTC 22:00"
      },
      {
        "id": "M52",
        "team1": "Morocco",
        "team2": "Haiti",
        "info": "M52, 18:00, 24 June | UTC 22:00"
      }
    ],
    "GroupD": [
      {
        "id": "M4",
        "team1": "United States",
        "team2": "Paraguay",
        "info": "M4, 21:00, 12 June | UTC 01:00, 13 June"
      },
      {
        "id": "M8",
        "team1": "Australia",
        "team2": "Türkiye",
        "info": "M8, 00:00, 14 June | UTC 04:00"
      },
      {
        "id": "M29",
        "team1": "United States",
        "team2": "Australia",
        "info": "M29, 15:00, 19 June | UTC 19:00"
      },
      {
        "id": "M32",
        "team1": "Türkiye",
        "team2": "Paraguay",
        "info": "M32, 23:00, 19 June | UTC 03:00, 20 June"
      },
      {
        "id": "M59",
        "team1": "Türkiye",
        "team2": "United States",
        "info": "M59, 22:00, 25 June | UTC 02:00, 26 June"
      },
      {
        "id": "M60",
        "team1": "Paraguay",
        "team2": "Australia",
        "info": "M60, 22:00, 25 June | UTC 02:00, 26 June"
      }
    ],
    "GroupE": [
      {
        "id": "M9",
        "team1": "Germany",
        "team2": "Curaçao",
        "info": "M9, 13:00, 14 June | UTC 17:00"
      },
      {
        "id": "M11",
        "team1": "Ivory Coast",
        "team2": "Ecuador",
        "info": "M11, 19:00, 14 June | UTC 23:00"
      },
      {
        "id": "M34",
        "team1": "Germany",
        "team2": "Ivory Coast",
        "info": "M34, 16:00, 20 June | UTC 20:00"
      },
      {
        "id": "M35",
        "team1": "Ecuador",
        "team2": "Curaçao",
        "info": "M35, 20:00, 20 June | UTC 00:00, 21 June"
      },
      {
        "id": "M55",
        "team1": "Curaçao",
        "team2": "Ivory Coast",
        "info": "M55, 16:00, 25 June | UTC 20:00"
      },
      {
        "id": "M56",
        "team1": "Ecuador",
        "team2": "Germany",
        "info": "M56, 16:00, 25 June | UTC 20:00"
      }
    ],
    "GroupF": [
      {
        "id": "M10",
        "team1": "Netherlands",
        "team2": "Japan",
        "info": "M10, 16:00, 14 June | UTC 20:00"
      },
      {
        "id": "M12",
        "team1": "Sweden",
        "team2": "Tunisia",
        "info": "M12, 22:00, 14 June | UTC 02:00, 15 June"
      },
      {
        "id": "M33",
        "team1": "Netherlands",
        "team2": "Sweden",
        "info": "M33, 13:00, 20 June | UTC 17:00"
      },
      {
        "id": "M36",
        "team1": "Tunisia",
        "team2": "Japan",
        "info": "M36, 00:00, 21 June | UTC 04:00"
      },
      {
        "id": "M57",
        "team1": "Japan",
        "team2": "Sweden",
        "info": "M57, 19:00, 25 June | UTC 23:00"
      },
      {
        "id": "M58",
        "team1": "Tunisia",
        "team2": "Netherlands",
        "info": "M58, 19:00, 25 June | UTC 23:00"
      }
    ],
    "GroupG": [
      {
        "id": "M14",
        "team1": "Belgium",
        "team2": "Egypt",
        "info": "M14, 15:00, 15 June | UTC 19:00"
      },
      {
        "id": "M16",
        "team1": "Iran",
        "team2": "New Zealand",
        "info": "M16, 21:00, 15 June | UTC 01:00, 16 June"
      },
      {
        "id": "M38",
        "team1": "Belgium",
        "team2": "Iran",
        "info": "M38, 15:00, 21 June | UTC 19:00"
      },
      {
        "id": "M40",
        "team1": "New Zealand",
        "team2": "Egypt",
        "info": "M40, 21:00, 21 June | UTC 01:00, 22 June"
      },
      {
        "id": "M65",
        "team1": "Egypt",
        "team2": "Iran",
        "info": "M65, 23:00, 26 June | UTC 03:00, 27 June"
      },
      {
        "id": "M66",
        "team1": "New Zealand",
        "team2": "Belgium",
        "info": "M66, 23:00, 26 June | UTC 03:00, 27 June"
      }
    ],
    "GroupH": [
      {
        "id": "M13",
        "team1": "Spain",
        "team2": "Cape Verde",
        "info": "M13, 12:00, 15 June | UTC 16:00"
      },
      {
        "id": "M15",
        "team1": "Saudi Arabia",
        "team2": "Uruguay",
        "info": "M15, 18:00, 15 June | UTC 22:00"
      },
      {
        "id": "M37",
        "team1": "Spain",
        "team2": "Saudi Arabia",
        "info": "M37, 12:00, 21 June | UTC 16:00"
      },
      {
        "id": "M39",
        "team1": "Uruguay",
        "team2": "Cape Verde",
        "info": "M39, 18:00, 21 June | UTC 22:00"
      },
      {
        "id": "M63",
        "team1": "Cape Verde",
        "team2": "Saudi Arabia",
        "info": "M63, 20:00, 26 June | UTC 00:00, 27 June"
      },
      {
        "id": "M64",
        "team1": "Uruguay",
        "team2": "Spain",
        "info": "M64, 20:00, 26 June | UTC 00:00, 27 June"
      }
    ],
    "GroupI": [
      {
        "id": "M17",
        "team1": "France",
        "team2": "Senegal",
        "info": "M17, 15:00, 16 June | UTC 19:00"
      },
      {
        "id": "M18",
        "team1": "Iraq",
        "team2": "Norway",
        "info": "M18, 18:00, 16 June | UTC 22:00"
      },
      {
        "id": "M42",
        "team1": "France",
        "team2": "Iraq",
        "info": "M42, 17:00, 22 June | UTC 21:00"
      },
      {
        "id": "M43",
        "team1": "Norway",
        "team2": "Senegal",
        "info": "M43, 20:00, 22 June | UTC 00:00, 23 June"
      },
      {
        "id": "M61",
        "team1": "Norway",
        "team2": "France",
        "info": "M61, 15:00, 26 June | UTC 19:00"
      },
      {
        "id": "M62",
        "team1": "Senegal",
        "team2": "Iraq",
        "info": "M62, 15:00, 26 June | UTC 19:00"
      }
    ],
    "GroupJ": [
      {
        "id": "M19",
        "team1": "Argentina",
        "team2": "Algeria",
        "info": "M19, 21:00, 16 June | UTC 01:00, 17 June"
      },
      {
        "id": "M20",
        "team1": "Austria",
        "team2": "Jordan",
        "info": "M20, 00:00, 17 June | UTC 04:00"
      },
      {
        "id": "M41",
        "team1": "Argentina",
        "team2": "Austria",
        "info": "M41, 13:00, 22 June | UTC 17:00"
      },
      {
        "id": "M44",
        "team1": "Jordan",
        "team2": "Algeria",
        "info": "M44, 23:00, 22 June | UTC 03:00, 23 June"
      },
      {
        "id": "M71",
        "team1": "Algeria",
        "team2": "Austria",
        "info": "M71, 22:00, 27 June | UTC 02:00, 28 June"
      },
      {
        "id": "M72",
        "team1": "Jordan",
        "team2": "Argentina",
        "info": "M72, 22:00, 27 June | UTC 02:00, 28 June"
      }
    ],
    "GroupK": [
      {
        "id": "M21",
        "team1": "Portugal",
        "team2": "DR Congo",
        "info": "M21, 13:00, 17 June | UTC 17:00"
      },
      {
        "id": "M24",
        "team1": "Uzbekistan",
        "team2": "Colombia",
        "info": "M24, 22:00, 17 June | UTC 02:00, 18 June"
      },
      {
        "id": "M45",
        "team1": "Portugal",
        "team2": "Uzbekistan",
        "info": "M45, 13:00, 23 June | UTC 17:00"
      },
      {
        "id": "M48",
        "team1": "Colombia",
        "team2": "DR Congo",
        "info": "M48, 22:00, 23 June | UTC 02:00, 24 June"
      },
      {
        "id": "M69",
        "team1": "Colombia",
        "team2": "Portugal",
        "info": "M69, 19:30, 27 June | UTC 23:30"
      },
      {
        "id": "M70",
        "team1": "DR Congo",
        "team2": "Uzbekistan",
        "info": "M70, 19:30, 27 June | UTC 23:30"
      }
    ],
    "GroupL": [
      {
        "id": "M22",
        "team1": "England",
        "team2": "Croatia",
        "info": "M22, 16:00, 17 June | UTC 20:00"
      },
      {
        "id": "M23",
        "team1": "Ghana",
        "team2": "Panama",
        "info": "M23, 19:00, 17 June | UTC 23:00"
      },
      {
        "id": "M46",
        "team1": "England",
        "team2": "Ghana",
        "info": "M46, 16:00, 23 June | UTC 20:00"
      },
      {
        "id": "M47",
        "team1": "Panama",
        "team2": "Croatia",
        "info": "M47, 19:00, 23 June | UTC 23:00"
      },
      {
        "id": "M67",
        "team1": "Panama",
        "team2": "England",
        "info": "M67, 17:00, 27 June | UTC 21:00"
      },
      {
        "id": "M68",
        "team1": "Croatia",
        "team2": "Ghana",
        "info": "M68, 17:00, 27 June | UTC 21:00"
      }
    ]
  },
  "r32": [
    {
      "node_id": "R32_1",
      "id": "M74",
      "team1_placeholder": "1E",
      "team2_placeholder": "3ABCDF",
      "info": "M74, 13:00, 29 June | UTC 17:00"
    },
    {
      "node_id": "R32_2",
      "id": "M77",
      "team1_placeholder": "1I",
      "team2_placeholder": "3CDFGH",
      "info": "M77, 13:00, 30 June | UTC 17:00"
    },
    {
      "node_id": "R32_3",
      "id": "M73",
      "team1_placeholder": "2A",
      "team2_placeholder": "2B",
      "info": "M73, 15:00, 28 June | UTC 19:00"
    },
    {
      "node_id": "R32_4",
      "id": "M75",
      "team1_placeholder": "1F",
      "team2_placeholder": "2C",
      "info": "M75, 16:00, 29 June | UTC 20:00"
    },
    {
      "node_id": "R32_5",
      "id": "M83",
      "team1_placeholder": "2K",
      "team2_placeholder": "2L",
      "info": "M83, 15:00, 2 July | UTC 19:00"
    },
    {
      "node_id": "R32_6",
      "id": "M84",
      "team1_placeholder": "1H",
      "team2_placeholder": "2J",
      "info": "M84, 19:00, 2 July | UTC 23:00"
    },
    {
      "node_id": "R32_7",
      "id": "M81",
      "team1_placeholder": "1D",
      "team2_placeholder": "3BEFIJ",
      "info": "M81, 16:00, 1 July | UTC 20:00"
    },
    {
      "node_id": "R32_8",
      "id": "M82",
      "team1_placeholder": "1G",
      "team2_placeholder": "3AEHIJ",
      "info": "M82, 20:00, 1 July | UTC 00:00, 2 July"
    },
    {
      "node_id": "R32_9",
      "id": "M76",
      "team1_placeholder": "1C",
      "team2_placeholder": "2F",
      "info": "M76, 21:00, 29 June | UTC 01:00, 30 June"
    },
    {
      "node_id": "R32_10",
      "id": "M78",
      "team1_placeholder": "2E",
      "team2_placeholder": "2I",
      "info": "M78, 17:00, 30 June | UTC 21:00"
    },
    {
      "node_id": "R32_11",
      "id": "M79",
      "team1_placeholder": "1A",
      "team2_placeholder": "3CEFHI",
      "info": "M79, 21:00, 30 June | UTC 01:00, 1 July"
    },
    {
      "node_id": "R32_12",
      "id": "M80",
      "team1_placeholder": "1L",
      "team2_placeholder": "3EHIJK",
      "info": "M80, 12:00, 1 July | UTC 16:00"
    },
    {
      "node_id": "R32_13",
      "id": "M86",
      "team1_placeholder": "1J",
      "team2_placeholder": "2H",
      "info": "M86, 14:00, 3 July | UTC 18:00"
    },
    {
      "node_id": "R32_14",
      "id": "M88",
      "team1_placeholder": "2D",
      "team2_placeholder": "2G",
      "info": "M88, 21:30, 3 July | UTC 01:30, 4 July"
    },
    {
      "node_id": "R32_15",
      "id": "M85",
      "team1_placeholder": "1B",
      "team2_placeholder": "3EFGIJ",
      "info": "M85, 21:00, 2 July | UTC 01:00, 3 July"
    },
    {
      "node_id": "R32_16",
      "id": "M87",
      "team1_placeholder": "1K",
      "team2_placeholder": "3DEIJL",
      "info": "M87, 18:00, 3 July | UTC 22:00"
    }
  ],
  "knockouts": [
    {
      "node_id": "R16_1",
      "id": "M89",
      "info": "M89, 13:00, 4 July | UTC 17:00",
      "depends_on": [
        "R32_1",
        "R32_2"
      ]
    },
    {
      "node_id": "R16_2",
      "id": "M90",
      "info": "M90, 17:00, 4 July | UTC 21:00",
      "depends_on": [
        "R32_3",
        "R32_4"
      ]
    },
    {
      "node_id": "R16_3",
      "id": "M93",
      "info": "M93, 15:00, 6 July | UTC 19:00",
      "depends_on": [
        "R32_5",
        "R32_6"
      ]
    },
    {
      "node_id": "R16_4",
      "id": "M94",
      "info": "M94, 20:00, 6 July | UTC 00:00, 7 July",
      "depends_on": [
        "R32_7",
        "R32_8"
      ]
    },
    {
      "node_id": "QF_1",
      "id": "Match 97",
      "info": "M97, 16:00, 9 July | UTC 20:00",
      "depends_on": [
        "R16_1",
        "R16_2"
      ]
    },
    {
      "node_id": "QF_2",
      "id": "Match 98",
      "info": "M98, 15:00, 10 July | UTC 19:00",
      "depends_on": [
        "R16_3",
        "R16_4"
      ]
    },
    {
      "node_id": "SF_1",
      "id": "Match 101",
      "info": "M101, 15:00, 14 July | UTC 19:00",
      "depends_on": [
        "QF_1",
        "QF_2"
      ]
    },
    {
      "node_id": "R16_5",
      "id": "M91",
      "info": "M91, 16:00, 5 July | UTC 20:00",
      "depends_on": [
        "R32_9",
        "R32_10"
      ]
    },
    {
      "node_id": "R16_6",
      "id": "M92",
      "info": "M92, 20:00, 5 July | UTC 00:00, 6 July",
      "depends_on": [
        "R32_11",
        "R32_12"
      ]
    },
    {
      "node_id": "R16_7",
      "id": "M95",
      "info": "M95, 12:00, 6 July | UTC 16:00",
      "depends_on": [
        "R32_13",
        "R32_14"
      ]
    },
    {
      "node_id": "R16_8",
      "id": "M96",
      "info": "M96, 16:00, 7 July | UTC 20:00",
      "depends_on": [
        "R32_15",
        "R32_16"
      ]
    },
    {
      "node_id": "QF_3",
      "id": "Match 99",
      "info": "M99, 17:00, 11 July | UTC 21:00",
      "depends_on": [
        "R16_5",
        "R16_6"
      ]
    },
    {
      "node_id": "QF_4",
      "id": "Match 100",
      "info": "M100, 21:00, 11 July | UTC 01:00, 12 July",
      "depends_on": [
        "R16_7",
        "R16_8"
      ]
    },
    {
      "node_id": "SF_2",
      "id": "Match 102",
      "info": "M102, 15:00, 15 July | UTC 19:00",
      "depends_on": [
        "QF_3",
        "QF_4"
      ]
    },
    {
      "node_id": "FINAL",
      "id": "M104",
      "info": "M104, 15:00, 19 July | UTC 19:00",
      "depends_on": [
        "SF_1",
        "SF_2"
      ]
    },
    {
      "node_id": "THIRD",
      "id": "M103",
      "info": "M103, 15:00, 18 July | UTC 19:00",
      "depends_on": [
        "SF_1",
        "SF_2"
      ]
    }
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
  users: ["Actual Results"],
  currentUser: "Actual Results",
  userScores: {
    "Actual Results": {}
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
  
  // Sync all actual scores to match the default user
  const defaultScores = state.userScores[state.users[0]];
  state.users.forEach(u => {
    if (u === state.users[0]) return;
    Object.keys(state.userScores[u]).forEach(key => {
      if (key.endsWith("_actHome") || key.endsWith("_actAway")) {
        delete state.userScores[u][key];
      }
    });
    Object.keys(defaultScores).forEach(key => {
      if (key.endsWith("_actHome") || key.endsWith("_actAway")) {
        state.userScores[u][key] = defaultScores[key];
      }
    });
  });
  
  initTabs();
  initBracketTabs();
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
  // 1. If we have data.json, ALWAYS use it as the source of truth (especially since we can now save to it directly).
  // 2. Otherwise, fall back to browser local storage.
  
  if (dataJson) {
    state = dataJson;
    console.log("Using data.json from disk/server as source of truth.");
  } else if (hasLocalData) {
    console.log("data.json not found, using existing browser local storage.");
  } else {
    console.log("No local storage or data.json found, using defaults.");
  }

  // Migrate "Default User" to "Actual Results" in loaded state
  if (state.users) {
    const idx = state.users.indexOf("Default User");
    if (idx !== -1) {
      state.users[idx] = "Actual Results";
    }
  }
  if (state.currentUser === "Default User") {
    state.currentUser = "Actual Results";
  }
  if (state.userScores && state.userScores["Default User"]) {
    state.userScores["Actual Results"] = state.userScores["Default User"];
    delete state.userScores["Default User"];
  }

  if (!state.users) state.users = ["Actual Results"];
  if (state.userScores) {
    Object.keys(state.userScores).forEach(u => {
      if (!state.users.includes(u)) {
        state.users.push(u);
      }
    });
  }
  
  if (!state.currentUser) state.currentUser = state.users[0];
  if (!state.userScores) state.userScores = { [state.currentUser]: {} };
  state.scores = state.userScores[state.currentUser];

  // Set body class for actual results view
  document.body.classList.toggle('user-actual-results', state.currentUser === 'Actual Results');
}

function disableEditing() {
  // Disable all number inputs for scores and set colors
  const inputs = document.querySelectorAll('input[type="number"]');
  inputs.forEach(input => {
    input.disabled = true;
    input.style.cursor = "not-allowed";
    
    // Set colors based on whether it's a predicted or actual score
    if (input.classList.contains("pred")) {
      input.style.color = "#ef4444"; // Red for predictions
      input.style.fontWeight = "bold";
    } else if (input.classList.contains("act") || input.classList.contains("actual")) {
      input.style.color = "#000000"; // Black for actual results
      input.style.fontWeight = "bold";
    }
    
    if (input.closest('.match-played')) {
      input.style.background = "#dcfce7"; // Light green for played matches
    } else {
      input.style.background = "#ffffff"; // White background
    }
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

    const fetchScoresBtn = document.createElement('button');
    fetchScoresBtn.className = "action-btn primary";
    fetchScoresBtn.innerHTML = '<i class="fa-solid fa-download"></i> Fetch Live Scores';
    fetchScoresBtn.title = "Fetch live scores from API-Football";
    fetchScoresBtn.style.background = "#3b82f6";
    fetchScoresBtn.style.color = "white";
    fetchScoresBtn.style.marginLeft = "10px";
    fetchScoresBtn.onclick = async () => {
      await fetchLiveScores();
    };
    navActions.appendChild(fetchScoresBtn);

    const autoFetchLabel = document.createElement('label');
    autoFetchLabel.style.color = "white";
    autoFetchLabel.style.marginLeft = "10px";
    autoFetchLabel.style.fontSize = "0.8rem";
    autoFetchLabel.style.display = "flex";
    autoFetchLabel.style.alignItems = "center";
    autoFetchLabel.style.gap = "5px";
    autoFetchLabel.style.cursor = "pointer";
    const autoFetchCheckbox = document.createElement('input');
    autoFetchCheckbox.type = "checkbox";
    autoFetchCheckbox.checked = localStorage.getItem('autoFetchEnabled') === 'true';
    
    // Auto Fetch Logic
    let autoFetchInterval = null;
    const startAutoFetch = () => {
      if (autoFetchInterval) clearInterval(autoFetchInterval);
      // Fetch every hour
      autoFetchInterval = setInterval(async () => {
        const updated = await fetchLiveScores(true); // true = silent mode
        if (updated) {
           updatePushBtn.click(); // auto push after successful fetch
        }
      }, 60 * 60 * 1000);
    };
    const stopAutoFetch = () => {
      if (autoFetchInterval) clearInterval(autoFetchInterval);
    };

    autoFetchCheckbox.onchange = (e) => {
      localStorage.setItem('autoFetchEnabled', e.target.checked);
      if (e.target.checked) {
        startAutoFetch();
        alert("Auto-Fetch enabled. It will fetch scores and auto-push every hour.");
      } else {
        stopAutoFetch();
        alert("Auto-Fetch disabled.");
      }
    };
    autoFetchLabel.appendChild(autoFetchCheckbox);
    autoFetchLabel.appendChild(document.createTextNode("Auto (1hr)"));
    navActions.appendChild(autoFetchLabel);

    if (autoFetchCheckbox.checked) {
      startAutoFetch();
    }

    const updatePushBtn = document.createElement('button');
    updatePushBtn.className = "action-btn primary";
    updatePushBtn.innerHTML = '<i class="fa-solid fa-cloud-arrow-up"></i> Update & Push';
    updatePushBtn.title = "Directly update data.json locally and push to GitHub";
    updatePushBtn.style.background = "#2ecc71";
    updatePushBtn.style.color = "white";
    updatePushBtn.style.marginLeft = "10px";
    updatePushBtn.onclick = async () => {
      const originalText = updatePushBtn.innerHTML;
      updatePushBtn.disabled = true;
      updatePushBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Pushing...';
      updatePushBtn.style.opacity = "0.7";

      await pushToGithub(false);

      updatePushBtn.disabled = false;
      updatePushBtn.innerHTML = originalText;
      updatePushBtn.style.opacity = "1";
    };
    navActions.appendChild(updatePushBtn);
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
        state.users = ["Actual Results"];
        state.currentUser = "Actual Results";
        state.userScores = {
          "Actual Results": parsedState.scores
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

// Push to GitHub helper
async function pushToGithub(silent = false) {
  const tempScores = state.scores;
  delete state.scores;
  try {
    const response = await fetch('http://localhost:3000/api/save-and-push', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(state, null, 2)
    });
    state.scores = tempScores;
    const result = await response.json();
    if (response.ok && result.success) {
      if (!silent) alert('Success! ' + result.message);
      return true;
    } else {
      const detailMessage = result.details ? `\n\nDetails: ${result.details}` : '';
      if (!silent) alert('Error: ' + (result.error || 'Failed to save and push.') + detailMessage);
      return false;
    }
  } catch (error) {
    state.scores = tempScores;
    console.error('Failed to communicate with helper server:', error);
    if (!silent) alert('Failed to connect to the local helper server. Please make sure "node server.js" is running in your terminal!');
    return false;
  }
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
          state.users.push("Actual Results");
          state.userScores["Actual Results"] = {};
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
      const defaultScores = state.userScores[state.users[0]];
      Object.keys(defaultScores).forEach(key => {
        if (key.endsWith("_actHome") || key.endsWith("_actAway")) {
          state.userScores[newName][key] = defaultScores[key];
        }
      });
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

  // Toggle body class for Actual Results view
  document.body.classList.toggle('user-actual-results', state.currentUser === 'Actual Results');
  
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

function initBracketTabs() {
  const leftBracket = document.getElementById("bracket-left-half");
  const rightBracket = document.getElementById("bracket-right-half");
  
  document.querySelectorAll(".bracket-layout-header .toggle-btn").forEach(btn => {
    btn.addEventListener("click", (e) => {
      // Handle active state on buttons
      document.querySelectorAll(".bracket-layout-header .toggle-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      
      const viewMode = btn.id;
      
      if (viewMode === "btn-bracket-both") {
        if (leftBracket) leftBracket.style.display = "block";
        if (rightBracket) rightBracket.style.display = "block";
      } else if (viewMode === "btn-bracket-left") {
        if (leftBracket) leftBracket.style.display = "block";
        if (rightBracket) rightBracket.style.display = "none";
      } else if (viewMode === "btn-bracket-right") {
        if (leftBracket) leftBracket.style.display = "none";
        if (rightBracket) rightBracket.style.display = "block";
      }
      
      // Redraw connectors
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          drawBracketConnectors();
        });
      });
    });
  });
}

// ---------------------- DYNAMIC POINTS CALCULATOR ----------------------
function calculatePoints(predHome, predAway, actHome, actAway, predHomePens, predAwayPens, actHomePens, actAwayPens) {
  if (predHome === null || predAway === null || actHome === null || actAway === null) return 0;
  
  const predWinner = predHome > predAway ? 'H' : (predHome < predAway ? 'A' : 'D');
  const actWinner = actHome > actAway ? 'H' : (actHome < actAway ? 'A' : 'D');
  
  const isKo = arguments.length > 4;

  let predAdvancing = predWinner;
  let actAdvancing = actWinner;

  if (isKo) {
    if (predWinner === 'D' && predHomePens !== undefined && predAwayPens !== undefined) {
      predAdvancing = predHomePens > predAwayPens ? 'H' : (predHomePens < predAwayPens ? 'A' : 'D');
    }
    if (actWinner === 'D' && actHomePens !== undefined && actAwayPens !== undefined) {
      actAdvancing = actHomePens > actAwayPens ? 'H' : (actHomePens < actAwayPens ? 'A' : 'D');
    }
  }

  const correctWinner = (predWinner === actWinner) || (isKo && predAdvancing === actAdvancing && predAdvancing !== 'D');
  const correctScore = (predHome === actHome && predAway === actAway);
  const correctHomeScore = (predHome === actHome);
  const correctAwayScore = (predAway === actAway);
  const correctAtLeastOneScore = (correctHomeScore || correctAwayScore);
  
  let basePoints = 0;

  // Rule 1: Correct winner and correct score
  if (correctWinner && correctScore) {
    basePoints = 5;
  }
  // Rule 6: Draw predicted, draw occurred, but scoreline was wrong
  else if (actWinner === 'D' && predWinner === 'D' && !correctScore) {
    const totalPredGoals = predHome + predAway;
    const totalActGoals = actHome + actAway;
    const diff = Math.abs(totalPredGoals - totalActGoals);
    if (diff > 0) {
      basePoints = 4 / diff;
    } else {
      basePoints = 5;
    }
  }
  // Rule 3: Correct winner, and at least one score was correct
  else if (correctWinner && correctAtLeastOneScore) {
    basePoints = 3;
  }
  // Rule 2: Correct winner, but completely incorrect scoreline
  else if (correctWinner) {
    basePoints = 2;
  }
  // Rule 4: Incorrect winner, but one correct score
  else if (!correctWinner && correctAtLeastOneScore) {
    basePoints = 1;
  }
  // Rule 5: Incorrect winner and incorrect scoreline
  else {
    basePoints = 0;
  }

  // Knockout Penalty Bonus: +1 point if both predict draw, actual is draw, and penalty winner is correct
  if (isKo && predWinner === 'D' && actWinner === 'D' && predAdvancing === actAdvancing && predAdvancing !== 'D') {
    basePoints += 1;
  }

  return basePoints;
}

// Determine which scoring rule was matched (for stats breakdown)
function getRuleMatched(predHome, predAway, actHome, actAway, predHomePens, predAwayPens, actHomePens, actAwayPens) {
  const points = calculatePoints(predHome, predAway, actHome, actAway, predHomePens, predAwayPens, actHomePens, actAwayPens);
  if (predHome === null || predAway === null || actHome === null || actAway === null) return null;
  
  const predWinner = predHome > predAway ? 'H' : (predHome < predAway ? 'A' : 'D');
  const actWinner = actHome > actAway ? 'H' : (actHome < actAway ? 'A' : 'D');

  const isKo = arguments.length > 4;
  let predAdvancing = predWinner;
  let actAdvancing = actWinner;

  if (isKo) {
    if (predWinner === 'D' && predHomePens !== undefined && predAwayPens !== undefined) {
      predAdvancing = predHomePens > predAwayPens ? 'H' : (predHomePens < predAwayPens ? 'A' : 'D');
    }
    if (actWinner === 'D' && actHomePens !== undefined && actAwayPens !== undefined) {
      actAdvancing = actHomePens > actAwayPens ? 'H' : (actHomePens < actAwayPens ? 'A' : 'D');
    }
  }

  const hasBonus = (isKo && predWinner === 'D' && actWinner === 'D' && predAdvancing === actAdvancing && predAdvancing !== 'D');
  
  // Strip off the penalty bonus to figure out the base rule
  const basePoints = hasBonus ? points - 1 : points;
  
  if (basePoints === 5) return 'rule1';
  if (basePoints === 3) return 'rule3';
  if (basePoints === 2) return 'rule2';
  if (basePoints === 1) return 'rule4';
  if (basePoints > 0 && basePoints < 5 && (predHome === predAway)) return 'rule6';
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
    
    Object.keys(initialMatchesData.groups).forEach((groupId, idx) => {
      const groupName = groupId.replace("Group", "Group ");
      const matches = initialMatchesData.groups[groupId];
      
      // Extract unique teams in group
      const teams = Array.from(new Set(matches.flatMap(m => [m.team1, m.team2])));
      
      const card = document.createElement("div");
      card.className = "group-card";
      card.id = `card-${groupId}`;
      card.style.animationDelay = `${idx * 0.05}s`;
      
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
              ${(() => {
                let lastDate = "";
                return allMatches.map(m => {
                  const groupLetter = m.groupId.replace("Group", "");
                  // Extract date from info string like "M1, 15:00, 11 June | UTC 19:00"
                  const dateMatch = m.info ? m.info.match(/(\d{1,2}\s+\w+)/) : null;
                  const matchDate = dateMatch ? dateMatch[1].trim() : "";
                  let separator = "";
                  if (matchDate && matchDate !== lastDate) {
                    lastDate = matchDate;
                    separator = `<div class="match-day-separator"><span>📅 ${matchDate}</span></div>`;
                  }
                  return `
                    ${separator}
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
                }).join("");
              })()}
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
      
      if (type === "actHome" || type === "actAway") {
        state.users.forEach(u => {
          if (val === "" || isNaN(val)) {
            delete state.userScores[u][matchId + "_" + type];
          } else {
            state.userScores[u][matchId + "_" + type] = val;
          }
        });
        state.scores = state.userScores[state.currentUser];
      } else {
        if (val === "" || isNaN(val)) {
          delete state.scores[matchId + "_" + type];
        } else {
          state.scores[matchId + "_" + type] = val;
        }
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
  let evaluatedMatches = 0;
  let totalGoalError = 0;
  
  // 1. Update Group Match Points and Standing Tables
  Object.keys(initialMatchesData.groups).forEach(groupId => {
    const matches = initialMatchesData.groups[groupId];
    
    // Standings calculation
    const stTypeElem = document.getElementById("standings-type-select");
    const stType = stTypeElem ? stTypeElem.value : "act";
    const predStandings = calculateGroupStandings(groupId, matches, stType);
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
      
      const matchElem = document.querySelector(`.match-item[data-match-id="${m.id}"]`);
      if (matchElem) {
        if (actHome !== undefined && actAway !== undefined) {
          matchElem.classList.add('match-played');
        } else {
          matchElem.classList.remove('match-played');
        }
      }
      
      const badge = document.getElementById(`points-${m.id}`);
      if (badge) {
        if (predHome !== undefined && predAway !== undefined && actHome !== undefined && actAway !== undefined) {
          evaluatedMatches += 1;
          const pts = calculatePoints(predHome, predAway, actHome, actAway);
          totalPoints += pts;
          totalGoalError += (Math.abs(predHome - actHome) + Math.abs(predAway - actAway));
          
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
    const match1 = t1.match(/^(\d)([A-L])$/);
    if (match1) {
      const rank = parseInt(match1[1]);
      const groupLetter = match1[2];
      const standings = groupStandingsMap[`Group${groupLetter}`];
      t1 = standings ? standings[rank - 1].team : t1;
    }
    
    const match2 = t2.match(/^(\d)([A-L])$/);
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

    t1 = state.userScores[state.users[0]][m.node_id + "_overrideTeam1"] || t1;
    t2 = state.userScores[state.users[0]][m.node_id + "_overrideTeam2"] || t2;
    
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
    
    // Check penalties if it's a draw
    const homePens = state.scores[nodeId + "_" + prefix + "HomePens"];
    const awayPens = state.scores[nodeId + "_" + prefix + "AwayPens"];
    if (homePens !== undefined && awayPens !== undefined) {
      if (homePens > awayPens) return scoresMap.team1;
      if (homePens < awayPens) return scoresMap.team2;
    }
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
    
    if (home > away) return scoresMap.team2;
    if (home < away) return scoresMap.team1;
    
    // Check penalties if it's a draw
    const homePens = state.scores[nodeId + "_" + prefix + "HomePens"];
    const awayPens = state.scores[nodeId + "_" + prefix + "AwayPens"];
    if (homePens !== undefined && awayPens !== undefined) {
      if (homePens > awayPens) return scoresMap.team2;
      if (homePens < awayPens) return scoresMap.team1;
    }
    return null;
  }

  // Populate winners along the tree using depends_on from knockout data
  koSequence.forEach(nodeId => {
    const deps = koDependsOn[nodeId];
    
    // Resolve teams for non-R32 nodes (R32 teams are already resolved above)
    if (deps) {
      if (nodeId === "THIRD") {
        // Third-place match uses losers instead of winners
        koTeams[nodeId] = {
          team1: state.userScores[state.users[0]][nodeId + "_overrideTeam1"] || getLoserOfNode(deps[0], "act") || `Loser ${deps[0]}`,
          team2: state.userScores[state.users[0]][nodeId + "_overrideTeam2"] || getLoserOfNode(deps[1], "act") || `Loser ${deps[1]}`
        };
      } else {
        // All other knockout matches use winners of their dependencies
        koTeams[nodeId] = {
          team1: state.userScores[state.users[0]][nodeId + "_overrideTeam1"] || getWinnerOfNode(deps[0], "act") || `Winner ${deps[0]}`,
          team2: state.userScores[state.users[0]][nodeId + "_overrideTeam2"] || getWinnerOfNode(deps[1], "act") || `Winner ${deps[1]}`
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
    
    const koCard = document.getElementById(`ko-card-${nodeId}`);
    if (koCard) {
      if (actHome !== undefined && actAway !== undefined) {
        koCard.classList.add('match-played');
      } else {
        koCard.classList.remove('match-played');
      }
    }
    
    if (predHome !== undefined && predAway !== undefined) {
      predictedMatches += 1;
    }
    
    const badge = document.getElementById(`ko-points-${nodeId}`);
    if (badge) {
      if (predHome !== undefined && predAway !== undefined && actHome !== undefined && actAway !== undefined) {
        evaluatedMatches += 1;
        const predHomePens = state.scores[nodeId + "_predHomePens"];
        const predAwayPens = state.scores[nodeId + "_predAwayPens"];
        const actHomePens = state.scores[nodeId + "_actHomePens"];
        const actAwayPens = state.scores[nodeId + "_actAwayPens"];
        
        const pts = calculatePoints(predHome, predAway, actHome, actAway, predHomePens, predAwayPens, actHomePens, actAwayPens);
        totalPoints += pts;
        totalGoalError += (Math.abs(predHome - actHome) + Math.abs(predAway - actAway));
        
        const rule = getRuleMatched(predHome, predAway, actHome, actAway, predHomePens, predAwayPens, actHomePens, actAwayPens);
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
  const acc = evaluatedMatches > 0 ? Math.round((ruleCounts.rule1 + ruleCounts.rule3 + ruleCounts.rule2 + ruleCounts.rule6) / evaluatedMatches * 100) : 0;
  document.getElementById("prediction-accuracy").textContent = `${acc}%`;
  const errorElem = document.getElementById("avg-goal-error");
  if (errorElem) errorElem.textContent = evaluatedMatches > 0 ? (totalGoalError / evaluatedMatches).toFixed(2) : "0.00";
  
  // 5. Update Tab 3 Stats Page counts
  document.getElementById("count-rule1").textContent = ruleCounts.rule1;
  document.getElementById("count-rule3").textContent = ruleCounts.rule3;
  document.getElementById("count-rule2").textContent = ruleCounts.rule2;
  document.getElementById("count-rule6").textContent = ruleCounts.rule6;
  document.getElementById("count-rule4").textContent = ruleCounts.rule4;
  document.getElementById("count-rule5").textContent = ruleCounts.rule5;
  document.getElementById("avg-points").textContent = evaluatedMatches > 0 ? (totalPoints / evaluatedMatches).toFixed(2) : "0.00";
  
  renderLeaderboard();
  renderPredictionsMatrix();
  
  // Disable actual score inputs for non-default users
  if (config.isLocal) {
    const isDefaultUser = state.currentUser === state.users[0];
    document.querySelectorAll('.score-input.actual, .ko-score-input.actual, .ko-score-input-pens.actual').forEach(input => {
      if (!isDefaultUser) {
        input.disabled = true;
        input.title = "Switch to " + state.users[0] + " to edit actual scores";
        input.style.cursor = "not-allowed";
        input.style.background = "rgba(16, 185, 129, 0.02)";
      } else {
        input.disabled = false;
        input.title = "";
        input.style.cursor = "text";
        input.style.background = ""; // Restore default via CSS
      }
    });
  }
  
  if (!config.isLocal) {
    disableEditing();
  }
}

function calculateUserStats(scoresObj) {
  let totalPoints = 0;
  let ruleCounts = { rule1: 0, rule2: 0, rule3: 0, rule4: 0, rule5: 0, rule6: 0, ruleMinus1: 0 };
  let predictedMatches = 0;
  let evaluatedMatches = 0;
  let totalGoalError = 0;
  
  let groupPoints = 0;
  let koPoints = 0;
  let overUnderTotal = 0;
  let overUnderCorrect = 0;
  let cleanSheetTotal = 0;
  let cleanSheetCorrect = 0;
  let scorelineCounts = {};
  let shootoutsTotal = 0;
  let shootoutsCorrect = 0;
  
  let predictedMaxGoals = [];
  let predictedMinGoals = [];
  
  let totalGoalsPredicted = 0;
  let drawsPredicted = 0;
  let maxGoalsInMatch = 0;
  let craziestPrediction = "-";
  
  Object.keys(initialMatchesData.groups).forEach(groupId => {
    initialMatchesData.groups[groupId].forEach(m => {
      const predHome = scoresObj[m.id + "_predHome"];
      const predAway = scoresObj[m.id + "_predAway"];
      const actHome = scoresObj[m.id + "_actHome"];
      const actAway = scoresObj[m.id + "_actAway"];
      
      if (predHome !== undefined && predAway !== undefined) {
        predictedMatches += 1;
        predictedMaxGoals.push(Math.max(predHome, predAway));
        predictedMinGoals.push(Math.min(predHome, predAway));
        
        const matchTotal = predHome + predAway;
        totalGoalsPredicted += matchTotal;
        if (predHome === predAway) drawsPredicted++;
        if (matchTotal > maxGoalsInMatch) {
          maxGoalsInMatch = matchTotal;
          craziestPrediction = `${predHome}-${predAway} (${matchTotal}g)`;
        }
      }
      if (predHome !== undefined && predAway !== undefined && actHome !== undefined && actAway !== undefined) {
        evaluatedMatches += 1;
        const pts = calculatePoints(predHome, predAway, actHome, actAway);
        totalPoints += pts;
        totalGoalError += (Math.abs(predHome - actHome) + Math.abs(predAway - actAway));
        const rule = getRuleMatched(predHome, predAway, actHome, actAway);
        if (rule) ruleCounts[rule] += 1;
        
        groupPoints += pts;
        
        const predTotal = predHome + predAway;
        const actTotal = actHome + actAway;
        if ((predTotal > 2.5 && actTotal > 2.5) || (predTotal <= 2.5 && actTotal <= 2.5)) overUnderCorrect++;
        overUnderTotal++;
        
        if (predHome === 0 || predAway === 0) {
          cleanSheetTotal++;
          if (predHome === 0 && actHome === 0) cleanSheetCorrect++;
          else if (predAway === 0 && actAway === 0) cleanSheetCorrect++;
        }
        
        const scoreStr = `${predHome}-${predAway}`;
        scorelineCounts[scoreStr] = (scorelineCounts[scoreStr] || 0) + 1;
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
      predictedMaxGoals.push(Math.max(predHome, predAway));
      predictedMinGoals.push(Math.min(predHome, predAway));
      
      const matchTotal = predHome + predAway;
      totalGoalsPredicted += matchTotal;
      if (predHome === predAway) drawsPredicted++;
      if (matchTotal > maxGoalsInMatch) {
        maxGoalsInMatch = matchTotal;
        craziestPrediction = `${predHome}-${predAway} (${matchTotal}g)`;
      }
    }
    if (predHome !== undefined && predAway !== undefined && actHome !== undefined && actAway !== undefined) {
      evaluatedMatches += 1;
      const predHomePens = scoresObj[nodeId + "_predHomePens"];
      const predAwayPens = scoresObj[nodeId + "_predAwayPens"];
      const actHomePens = scoresObj[nodeId + "_actHomePens"];
      const actAwayPens = scoresObj[nodeId + "_actAwayPens"];

      const pts = calculatePoints(predHome, predAway, actHome, actAway, predHomePens, predAwayPens, actHomePens, actAwayPens);
      totalPoints += pts;
      totalGoalError += (Math.abs(predHome - actHome) + Math.abs(predAway - actAway));
      const rule = getRuleMatched(predHome, predAway, actHome, actAway, predHomePens, predAwayPens, actHomePens, actAwayPens);
      if (rule) ruleCounts[rule] += 1;
      
      koPoints += pts;
      
      const predTotal = predHome + predAway;
      const actTotal = actHome + actAway;
      if ((predTotal > 2.5 && actTotal > 2.5) || (predTotal <= 2.5 && actTotal <= 2.5)) overUnderCorrect++;
      overUnderTotal++;
      
      if (predHome === 0 || predAway === 0) {
        cleanSheetTotal++;
        if (predHome === 0 && actHome === 0) cleanSheetCorrect++;
        else if (predAway === 0 && actAway === 0) cleanSheetCorrect++;
      }
      
      const scoreStr = `${predHome}-${predAway}`;
      scorelineCounts[scoreStr] = (scorelineCounts[scoreStr] || 0) + 1;
      
      if (actHome === actAway) {
        shootoutsTotal++;
        if (predHome === predAway) shootoutsCorrect++;
      }
    }
  });
  
  const acc = evaluatedMatches > 0 ? Math.round((ruleCounts.rule1 + ruleCounts.rule3 + ruleCounts.rule2 + ruleCounts.rule6) / evaluatedMatches * 100) : 0;
  const avgGoalError = evaluatedMatches > 0 ? (totalGoalError / evaluatedMatches).toFixed(2) : "0.00";
  
  let favoriteScoreline = "-";
  let maxCount = 0;
  for (const [score, count] of Object.entries(scorelineCounts)) {
    if (count > maxCount) {
      maxCount = count;
      favoriteScoreline = score;
    }
  }
  
  let medianScoreline = "-";
  if (predictedMaxGoals.length > 0) {
    predictedMaxGoals.sort((a, b) => a - b);
    predictedMinGoals.sort((a, b) => a - b);
    const mid = Math.floor(predictedMaxGoals.length / 2);
    let medianMax = predictedMaxGoals[mid];
    let medianMin = predictedMinGoals[mid];
    if (predictedMaxGoals.length % 2 === 0) {
      medianMax = (predictedMaxGoals[mid - 1] + predictedMaxGoals[mid]) / 2;
      medianMin = (predictedMinGoals[mid - 1] + predictedMinGoals[mid]) / 2;
    }
    medianMax = medianMax % 1 === 0 ? medianMax : medianMax.toFixed(1);
    medianMin = medianMin % 1 === 0 ? medianMin : medianMin.toFixed(1);
    medianScoreline = `${medianMax}-${medianMin}`;
  }
  
  const drawRate = predictedMatches > 0 ? Math.round((drawsPredicted / predictedMatches) * 100) : 0;
  
  return { 
    totalPoints, predictedMatches, acc, ruleCounts, avgGoalError,
    groupPoints, koPoints, overUnderCorrect, overUnderTotal,
    cleanSheetCorrect, cleanSheetTotal, favoriteScoreline, medianScoreline,
    shootoutsCorrect, shootoutsTotal,
    totalGoalsPredicted, drawRate, craziestPrediction
  };
}

function renderPredictionsMatrix() {
  const headerElem = document.getElementById("matrix-header");
  const bodyElem = document.getElementById("matrix-body");
  if (!headerElem || !bodyElem) return;

  const filterCheckboxes = document.querySelectorAll('#matrix-stage-filters input[type="checkbox"]');
  filterCheckboxes.forEach(cb => {
    if (!cb.dataset.listenerAttached) {
      const savedState = localStorage.getItem(`matrix_filter_${cb.value}`);
      if (savedState !== null) {
        cb.checked = savedState === 'true';
      }
      cb.addEventListener('change', () => {
        localStorage.setItem(`matrix_filter_${cb.value}`, cb.checked);
        renderPredictionsMatrix();
      });
      cb.dataset.listenerAttached = "true";
    }
  });

  const stageFilters = Array.from(filterCheckboxes).filter(cb => cb.checked).map(cb => cb.value);

  const usersToShow = state.users.filter(u => u !== "Actual Results");
  
  const distinctColors = [
    "#60a5fa", // Blue
    "#f87171", // Red
    "#34d399", // Emerald
    "#a78bfa", // Purple
    "#fb923c", // Orange
    "#f472b6", // Pink
    "#2dd4bf", // Teal
    "#facc15", // Yellow
    "#bef264", // Lime
    "#38bdf8", // Light Blue
    "#fb7185", // Rose
    "#d946ef"  // Fuchsia
  ];
  
  const userColors = {};
  usersToShow.forEach((u, i) => {
    const lowerName = u.toLowerCase();
    const presetColors = {
      "oneiros": "#facc15", // Bright yellow
      "nithin": "#f87171",  // Red
      "abhi": "#34d399",    // Emerald
      "anuj": "#a78bfa",    // Purple
      "karthik": "#fb923c", // Orange
      "tej": "#f472b6",     // Pink
      "divyank": "#38bdf8"  // Light Blue
    };

    if (presetColors[lowerName]) {
      userColors[u] = presetColors[lowerName];
    } else {
      // Find the next available distinct color not already assigned
      let colorAssigned = false;
      for (let j = 0; j < distinctColors.length; j++) {
        const candidateColor = distinctColors[(i + j) % distinctColors.length];
        if (!Object.values(userColors).includes(candidateColor)) {
          userColors[u] = candidateColor;
          colorAssigned = true;
          break;
        }
      }
      if (!colorAssigned) {
        const hue = (i * 137.508) % 360; 
        userColors[u] = `hsl(${hue}, 80%, 60%)`;
      }
    }
  });

  // Pre-calculate final totals and played matches count
  const finalTotals = {};
  usersToShow.forEach(u => finalTotals[u] = 0);
  let playedCount = 0;

  // Collect group and knockout matches to calculate totals
  let allMatchesForTotals = [];
  Object.keys(initialMatchesData.groups).forEach(groupId => {
    initialMatchesData.groups[groupId].forEach(m => {
      allMatchesForTotals.push({ ...m, isKo: false });
    });
  });
  const koMatchesForTotals = [...(initialMatchesData.r32 || []), ...(initialMatchesData.knockouts || [])];
  koMatchesForTotals.forEach(m => {
    allMatchesForTotals.push({ ...m, isKo: true });
  });

  allMatchesForTotals.forEach(m => {
    const mId = m.isKo ? m.node_id : m.id;
    const actHome = state.userScores[state.users[0]][mId + "_actHome"];
    const actAway = state.userScores[state.users[0]][mId + "_actAway"];
    const isPlayed = (actHome !== undefined && actAway !== undefined);
    if (isPlayed) {
      playedCount++;
      usersToShow.forEach(u => {
        const pHome = state.userScores[u][mId + "_predHome"];
        const pAway = state.userScores[u][mId + "_predAway"];
        if (pHome !== undefined && pAway !== undefined) {
          if (m.isKo) {
            const pHomePens = state.userScores[u][mId + "_predHomePens"];
            const pAwayPens = state.userScores[u][mId + "_predAwayPens"];
            const aHomePens = state.userScores[state.users[0]][mId + "_actHomePens"];
            const aAwayPens = state.userScores[state.users[0]][mId + "_actAwayPens"];
            finalTotals[u] += calculatePoints(pHome, pAway, actHome, actAway, pHomePens, pAwayPens, aHomePens, aAwayPens);
          } else {
            finalTotals[u] += calculatePoints(pHome, pAway, actHome, actAway);
          }
        }
      });
    }
  });

  const maxPossibleScore = playedCount * 5;

  let headerHTML = `
    <tr>
      <th style="min-width: 80px; text-align: left; white-space: nowrap; position: sticky; top: 0; z-index: 10; background: var(--bg-deep);">Match</th>
      <th style="width: 30%; text-align: left; position: sticky; top: 0; z-index: 10; background: var(--bg-deep);">Teams</th>
      <th style="text-align: center; width: 10%; position: sticky; top: 0; z-index: 10; background: var(--bg-deep);">Actual<br><span style="font-size: 0.7rem; font-weight: normal; color: #9ca3af;">Max: ${maxPossibleScore}</span></th>
  `;
  usersToShow.forEach(u => {
    const pts = finalTotals[u];
    const isCurrentUser = (u === state.currentUser);
    const highlightBg = isCurrentUser ? '#1a1432' : 'var(--bg-deep)';
    headerHTML += `<th style="text-align: center; color: ${userColors[u]}; position: sticky; top: 0; z-index: 10; background: ${highlightBg}; box-shadow: 0 1px 0 rgba(255,255,255,0.1);">${u}<br><span style="font-size: 0.7rem; font-weight: normal; color: #9ca3af;">${pts.toFixed(pts % 1 === 0 ? 0 : 2)} pts</span></th>`;
  });
  headerHTML += `</tr>`;
  headerElem.innerHTML = headerHTML;

  let bodyHTML = "";

  let allMatches = [];
  
  // Collect group matches
  Object.keys(initialMatchesData.groups).forEach(groupId => {
    initialMatchesData.groups[groupId].forEach(m => {
      allMatches.push({
        ...m,
        isKo: false
      });
    });
  });
  
  // Collect knockout matches
  const koMatches = [...(initialMatchesData.r32 || []), ...(initialMatchesData.knockouts || [])];
  koMatches.forEach(m => {
    allMatches.push({
      ...m,
      isKo: true
    });
  });
  
  // Sort by match number parsed from id (e.g. "M1", "Match 97")
  allMatches.sort((a, b) => {
    const numA = parseInt(a.id.replace(/[^\d]/g, '')) || 0;
    const numB = parseInt(b.id.replace(/[^\d]/g, '')) || 0;
    return numA - numB;
  });
  
  const plotlyData = usersToShow.map(u => {
    return {
      x: [0],
      y: [0],
      text: [`<span style="color: ${userColors[u]};"><b>${u}</b>: Start: 0 pts</span>`],
      mode: 'lines+markers',
      name: u,
      line: { color: userColors[u], width: 3 },
      marker: { size: 6 },
      hovertemplate: '%{text}<extra></extra>'
    };
  });
  
  const runningTotals = {};
  usersToShow.forEach(u => runningTotals[u] = 0);

  let currentPhase = '';

  // Render sorted list
  allMatches.forEach(m => {
    const matchNum = parseInt(m.id.replace(/[^\d]/g, '')) || m.id;
    const displayStr = `Match ${matchNum}`;
    const mId = m.isKo ? m.node_id : m.id;
    const mInfoOriginal = m.info || mId;
    
    let t1 = "?", t2 = "?";
    if (m.isKo) {
      if (globalKoTeams && globalKoTeams[m.node_id]) {
        t1 = globalKoTeams[m.node_id].team1;
        t2 = globalKoTeams[m.node_id].team2;
      } else {
        t1 = m.team1 || m.source1 || m.team1_placeholder || "?";
        t2 = m.team2 || m.source2 || m.team2_placeholder || "?";
      }
    } else {
      t1 = m.team1;
      t2 = m.team2;
    }
    
    const actHome = state.userScores[state.users[0]][mId + "_actHome"];
    const actAway = state.userScores[state.users[0]][mId + "_actAway"];
    const actualStr = (actHome !== undefined && actAway !== undefined) ? `<span style="color: #6ee7b7;">${actHome} - ${actAway}</span>` : "-";
    const actualHoverStr = (actHome !== undefined && actAway !== undefined) ? `${actHome} - ${actAway}` : "TBD";
    const isPlayed = (actHome !== undefined && actAway !== undefined);

    let rowBgColor = '';
    let phase = 'Group Stage';
    if (m.isKo) {
      if (m.node_id.startsWith('R32')) { rowBgColor = 'background-color: rgba(59, 130, 246, 0.05);'; phase = 'Round of 32'; }
      else if (m.node_id.startsWith('R16')) { rowBgColor = 'background-color: rgba(59, 130, 246, 0.1);'; phase = 'Round of 16'; }
      else if (m.node_id.startsWith('QF')) { rowBgColor = 'background-color: rgba(59, 130, 246, 0.15);'; phase = 'Quarter-Finals'; }
      else if (m.node_id.startsWith('SF')) { rowBgColor = 'background-color: rgba(139, 92, 246, 0.15);'; phase = 'Semi-Finals'; }
      else if (m.node_id === 'THIRD') { rowBgColor = 'background-color: rgba(245, 158, 11, 0.15);'; phase = 'Third Place Match'; }
      else if (m.node_id === 'FINAL') { rowBgColor = 'background-color: rgba(212, 175, 55, 0.2);'; phase = 'Final'; }
    }

    let headerRowHTML = '';
    if (phase !== currentPhase) {
      currentPhase = phase;
      const colspan = 3 + usersToShow.length;
      headerRowHTML = `
        <tr>
          <td colspan="${colspan}" style="background: rgba(255,255,255,0.05); color: #f6e093; text-align: center; font-weight: bold; padding: 8px; font-size: 0.9rem; letter-spacing: 1px; border-top: 1px solid rgba(255,255,255,0.1); border-bottom: 1px solid rgba(255,255,255,0.1);">
            ${phase.toUpperCase()}
          </td>
        </tr>
      `;
    }

    let rowHTML = `${headerRowHTML}
      <tr style="border-bottom: 1px solid rgba(255,255,255,0.05); ${rowBgColor}">
        <td style="color: var(--text-muted); font-size: 0.75rem; white-space: nowrap;">${displayStr}</td>
        <td><strong>${t1}</strong> <span style="color: var(--text-muted); font-size: 0.8rem;">vs</span> <strong>${t2}</strong></td>
        <td style="text-align: center; font-weight: bold; background: rgba(16, 185, 129, 0.05); border-left: 1px solid rgba(255,255,255,0.05); border-right: 1px solid rgba(255,255,255,0.05);">${actualStr}</td>
    `;

    let fullHoverTxt = `<b>${displayStr}</b><br>${mInfoOriginal.split('|')[0].trim()}<br>${t1} vs ${t2}<br><b>Actual: ${actualHoverStr}</b>`;
    
    // First pass: compute points and build unified hover string
    const matchUserData = [];
    usersToShow.forEach((u, uIdx) => {
      const pHome = state.userScores[u][mId + "_predHome"];
      const pAway = state.userScores[u][mId + "_predAway"];
      const predStr = (pHome !== undefined && pAway !== undefined) ? `${pHome} - ${pAway}` : "-";
      
      let pts = 0;
      let ptsStr = "";
      if (pHome !== undefined && pAway !== undefined && isPlayed) {
        if (m.isKo) {
          const pHomePens = state.userScores[u][mId + "_predHomePens"];
          const pAwayPens = state.userScores[u][mId + "_predAwayPens"];
          const aHomePens = state.userScores[state.users[0]][mId + "_actHomePens"];
          const aAwayPens = state.userScores[state.users[0]][mId + "_actAwayPens"];
          pts = calculatePoints(pHome, pAway, actHome, actAway, pHomePens, pAwayPens, aHomePens, aAwayPens);
        } else {
          pts = calculatePoints(pHome, pAway, actHome, actAway);
        }
        ptsStr = `<br><span style="font-size: 0.65rem; color: #9ca3af;">(${pts.toFixed(pts % 1 === 0 ? 0 : 2)}p)</span>`;
      }
      
      if (isPlayed) {
        runningTotals[u] += pts;
        fullHoverTxt += `<br><span style="color: ${userColors[u]};"><b>${u}</b>: Pred: ${predStr} | +${pts.toFixed(2)}p | Total: ${runningTotals[u].toFixed(2)}</span>`;
      }
      
      matchUserData.push({ predStr, ptsStr });
    });

    // Second pass: apply to plotly traces and table
    usersToShow.forEach((u, uIdx) => {
      const d = matchUserData[uIdx];
      
      if (isPlayed) {
        plotlyData[uIdx].x.push(matchNum);
        plotlyData[uIdx].y.push(runningTotals[u]);
        plotlyData[uIdx].text.push(fullHoverTxt);
      }
      
      const isCurrentUser = (u === state.currentUser);
      const bgStyle = isCurrentUser ? 'background-color: rgba(212, 175, 55, 0.05);' : '';
      rowHTML += `<td style="text-align: center; color: ${userColors[u]}; ${bgStyle}">${d.predStr}${d.ptsStr}</td>`;
    });
    rowHTML += `</tr>`;
    
    if (stageFilters.includes(phase)) {
      bodyHTML += rowHTML;
    }
  });

  bodyElem.innerHTML = bodyHTML;
  
  // Determine X-axis range with a buffer
  let minMatch = Infinity;
  let maxMatch = -Infinity;
  plotlyData.forEach(trace => {
    if (trace.x.length > 0) {
      minMatch = Math.min(minMatch, ...trace.x);
      maxMatch = Math.max(maxMatch, ...trace.x);
    }
  });
  if (minMatch === Infinity) { minMatch = 0; maxMatch = 104; }
  else { minMatch = -1; maxMatch = 104; }

  // Build Animation Frames
  const frames = [];
  const maxPoints = Math.max(...plotlyData.map(t => t.x.length));

  for (let k = 1; k <= maxPoints; k++) {
    const frameData = plotlyData.map(t => ({
      x: t.x.slice(0, k),
      y: t.y.slice(0, k),
      text: t.text.slice(0, k)
    }));

    let frameAnnotations = usersToShow.map((u, uIdx) => {
      const t = plotlyData[uIdx];
      const lastIdx = Math.min(k - 1, t.x.length - 1);
      if (lastIdx < 0 || t.x.length === 0) return null;
      
      return {
        user: u,
        x: t.x[lastIdx],
        y: t.y[lastIdx],
        rawY: t.y[lastIdx],
        text: `<b>${u}</b>`,
        showarrow: false,
        xanchor: 'left',
        yanchor: 'middle',
        font: { color: userColors[u], family: 'Inter, sans-serif', size: 10 },
        bgcolor: 'rgba(0,0,0,0.6)',
        bordercolor: userColors[u],
        borderwidth: 1,
        borderpad: 2,
        yshift: 0
      };
    }).filter(a => a !== null);

    if (frameAnnotations.length > 0) {
      // Find the maximum points in this frame to highlight the leader
      const maxY = Math.max(...frameAnnotations.map(a => a.rawY));
      
      // Sort ascending by rawY to stack overlapping labels upward
      frameAnnotations.sort((a, b) => a.rawY - b.rawY);
      
      let prevRawY = -Infinity;
      let prevShift = 0;
      
      for (let i = 0; i < frameAnnotations.length; i++) {
        const ann = frameAnnotations[i];
        
        // Approximate pixels per point: ~600px height / ~120 max points = ~5px per point
        const pixelDiff = (ann.rawY - prevRawY) * 5;
        const currentDistance = pixelDiff - prevShift;
        
        if (currentDistance < 25) {
          ann.yshift = 25 - currentDistance;
        } else {
          ann.yshift = 0;
        }
        
        prevRawY = ann.rawY;
        prevShift = ann.yshift;
        
        // Highlight leader
        if (ann.rawY === maxY && maxY > 0) {
          ann.text = `👑 <b>${ann.user}</b>`;
          ann.font.size = 12;
          ann.bgcolor = 'rgba(212, 175, 55, 0.4)'; // Gold background
          ann.bordercolor = '#d4af37';
          ann.borderwidth = 2;
        }
      }

      // Add a global banner for the leader at the top of the graph
      let banner = null;
      if (maxY > 0) {
        const leaders = frameAnnotations.filter(a => a.rawY === maxY).map(a => a.user);
        const currentMatch = frameAnnotations[0].x;
        banner = {
          xref: 'paper',
          yref: 'paper',
          x: 0.02,
          y: 0.98,
          text: `<b>Match ${currentMatch}</b><br>👑 Leader: <b>${leaders.join(', ')}</b>`,
          showarrow: false,
          xanchor: 'left',
          yanchor: 'top',
          font: { color: '#ffffff', family: 'Inter, sans-serif', size: 14 },
          bgcolor: 'rgba(212, 175, 55, 0.4)',
          bordercolor: '#d4af37',
          borderwidth: 2,
          borderpad: 6
        };
      }
      
      if (k === maxPoints) {
        frameAnnotations = []; // Clear individual labels on the final frame
      }
      
      if (banner) {
        frameAnnotations.push(banner);
      }
    }

    frames.push({
      name: 'f' + k,
      data: frameData,
      layout: { annotations: frameAnnotations }
    });
  }

  window.plotlyFrames = frames;

  const maxTotals = Math.max(0, ...plotlyData.flatMap(t => t.y));

  // Render Plotly Graph
  const layout = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#e5e7eb', family: 'Inter, sans-serif' },
    margin: { t: 50, r: 30, b: 50, l: 50 },
    xaxis: { 
      title: 'Match Number', 
      gridcolor: 'rgba(255,255,255,0.1)',
      zerolinecolor: 'rgba(255,255,255,0.2)',
      range: [minMatch, maxMatch],
      showspikes: document.getElementById('toggle-spikeline') ? document.getElementById('toggle-spikeline').checked : true,
      spikemode: 'across',
      spikedash: 'dash',
      spikecolor: '#ffffff',
      spikethickness: 1
    },
    yaxis: { 
      title: 'Cumulative Points', 
      gridcolor: 'rgba(255,255,255,0.1)',
      zerolinecolor: 'rgba(255,255,255,0.2)',
      range: [0, maxTotals * 1.15 + 15]
    },
    legend: { orientation: 'h', y: -0.2 },
    hovermode: document.getElementById('toggle-spikeline') && !document.getElementById('toggle-spikeline').checked ? false : 'closest',
    hoverdistance: -1,
    hoverlabel: {
      bgcolor: '#0b0125',
      font: { color: '#ffffff', family: 'Inter, sans-serif' },
      bordercolor: 'rgba(212, 175, 55, 0.4)'
    },
    annotations: window.plotlyFrames && window.plotlyFrames.length > 0 ? window.plotlyFrames[window.plotlyFrames.length - 1].layout.annotations : []
  };
  
  if (typeof Plotly !== 'undefined') {
    Plotly.newPlot('plotly-graph', plotlyData, layout, {responsive: true, displayModeBar: true, displaylogo: false}).then(() => {
      Plotly.addFrames('plotly-graph', window.plotlyFrames);
    });
  }

  // --- GAP ANALYSIS GRAPH ---
  const gapData = usersToShow.map(u => ({
    x: [],
    y: [],
    text: [],
    mode: 'lines',
    name: u,
    line: { color: userColors[u], width: 2 },
    hovertemplate: '%{text}<extra></extra>'
  }));

  for (let k = 0; k < maxPoints; k++) {
    const yValues = plotlyData.map(t => t.y[k]);
    const maxVal = Math.max(...yValues);
    
    usersToShow.forEach((u, uIdx) => {
      const matchNum = plotlyData[uIdx].x[k];
      const pt = plotlyData[uIdx].y[k];
      const gap = pt - maxVal; // will be <= 0
      
      gapData[uIdx].x.push(matchNum);
      gapData[uIdx].y.push(gap);
      gapData[uIdx].text.push(`<span style="color: ${userColors[u]};"><b>${u}</b>: Gap to leader: ${gap.toFixed(2)} pts</span>`);
    });
  }

  const minGap = Math.min(0, ...gapData.flatMap(t => t.y));

  const gapLayout = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#e5e7eb', family: 'Inter, sans-serif' },
    margin: { t: 20, r: 30, b: 50, l: 50 },
    xaxis: { 
      title: 'Match Number', 
      gridcolor: 'rgba(255,255,255,0.1)',
      zerolinecolor: 'rgba(255,255,255,0.2)',
      range: [minMatch, maxMatch],
      showspikes: document.getElementById('toggle-spikeline') ? document.getElementById('toggle-spikeline').checked : true,
      spikemode: 'across',
      spikedash: 'dash',
      spikecolor: '#ffffff',
      spikethickness: 1
    },
    yaxis: { 
      title: 'Points Behind Leader', 
      gridcolor: 'rgba(255,255,255,0.1)',
      zerolinecolor: 'rgba(255,255,255,0.2)',
      range: [minGap - 5, 2] // padding on bottom and top
    },
    showlegend: false,
    hovermode: document.getElementById('toggle-spikeline') && !document.getElementById('toggle-spikeline').checked ? false : 'closest',
    hoverdistance: -1,
    hoverlabel: {
      bgcolor: '#0b0125',
      font: { color: '#ffffff', family: 'Inter, sans-serif' },
      bordercolor: 'rgba(255, 255, 255, 0.4)'
    }
  };

  if (typeof Plotly !== 'undefined' && document.getElementById('plotly-graph-gap')) {
    const gapFrames = [];
    for (let k = 1; k <= maxPoints; k++) {
      const frameData = gapData.map(t => ({
        x: t.x.slice(0, k),
        y: t.y.slice(0, k),
        text: t.text.slice(0, k)
      }));
      gapFrames.push({
        name: 'f' + k,
        data: frameData
      });
    }
    window.plotlyGapFrames = gapFrames;

    Plotly.newPlot('plotly-graph-gap', gapData, gapLayout, {responsive: true, displayModeBar: true, displaylogo: false}).then(() => {
      Plotly.addFrames('plotly-graph-gap', window.plotlyGapFrames);
    });
  }

  // Helper to convert hex colors to RGBA
  const hexToRgbA = (hex, alpha) => {
    let c;
    if(/^#([A-Fa-f0-9]{3}){1,2}$/.test(hex)){
      c= hex.substring(1).split('');
      if(c.length== 3){
        c= [c[0], c[0], c[1], c[1], c[2], c[2]];
      }
      c= '0x' + c.join('');
      return 'rgba('+[(c>>16)&255, (c>>8)&255, c&255].join(',')+','+alpha+')';
    }
    return hex;
  };

  // 1. Prediction Style Breakdown (Grouped Bar Chart)
  const styleData = usersToShow.map(u => {
    const stats = calculateUserStats(state.userScores[u] || {});
    const rc = stats.ruleCounts;
    return {
      x: ["Exact Score", "Correct Draw", "Winner + 1", "Winner Only", "Loser + 1", "Incorrect"],
      y: [rc.rule1, rc.rule6, rc.rule3, rc.rule2, rc.rule4, rc.rule5],
      name: u,
      type: 'bar',
      marker: { color: userColors[u] }
    };
  });

  const styleLayout = {
    barmode: 'group',
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#e5e7eb', family: 'Inter, sans-serif' },
    margin: { t: 20, r: 20, b: 40, l: 40 },
    xaxis: { 
      gridcolor: 'rgba(255,255,255,0.1)',
      zerolinecolor: 'rgba(255,255,255,0.2)'
    },
    yaxis: { 
      title: 'Match Count',
      gridcolor: 'rgba(255,255,255,0.1)',
      zerolinecolor: 'rgba(255,255,255,0.2)'
    },
    legend: { orientation: 'h', y: -0.2 },
    hoverlabel: {
      bgcolor: '#0b0125',
      font: { color: '#ffffff', family: 'Inter, sans-serif' },
      bordercolor: 'rgba(212, 175, 55, 0.4)'
    }
  };

  // 2. Group-by-Group Performance (Radar Chart)
  const groupIds = ["GroupA", "GroupB", "GroupC", "GroupD", "GroupE", "GroupF", "GroupG", "GroupH", "GroupI", "GroupJ", "GroupK", "GroupL"];
  const radarData = usersToShow.map(u => {
    const rValues = groupIds.map(groupId => {
      let ptsInGroup = 0;
      initialMatchesData.groups[groupId].forEach(m => {
        const actHome = state.userScores[state.users[0]][m.id + "_actHome"];
        const actAway = state.userScores[state.users[0]][m.id + "_actAway"];
        const pHome = state.userScores[u][m.id + "_predHome"];
        const pAway = state.userScores[u][m.id + "_predAway"];
        
        if (actHome !== undefined && actAway !== undefined && pHome !== undefined && pAway !== undefined) {
          ptsInGroup += calculatePoints(pHome, pAway, actHome, actAway);
        }
      });
      return ptsInGroup;
    });

    const rClosed = [...rValues, rValues[0]];
    const thetaClosed = [...groupIds.map(g => g.replace("Group", "Group ")), groupIds[0].replace("Group", "Group ")];

    return {
      type: 'scatterpolar',
      r: rClosed,
      theta: thetaClosed,
      fill: 'toself',
      fillcolor: hexToRgbA(userColors[u], 0.04),
      name: u,
      line: { color: userColors[u], width: 2 },
      marker: { size: 4 }
    };
  });

  const radarLayout = {
    polar: {
      radialaxis: {
        visible: true,
        gridcolor: 'rgba(255,255,255,0.1)',
        linecolor: 'rgba(255,255,255,0.2)',
        tickfont: { size: 9, color: '#9ca3af' },
        bgcolor: 'rgba(0,0,0,0)'
      },
      angularaxis: {
        gridcolor: 'rgba(255,255,255,0.1)',
        linecolor: 'rgba(255,255,255,0.2)',
        tickfont: { size: 10, color: '#e5e7eb' }
      },
      bgcolor: 'rgba(0,0,0,0)'
    },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#e5e7eb', family: 'Inter, sans-serif' },
    margin: { t: 30, r: 30, b: 30, l: 30 },
    showlegend: true,
    legend: { orientation: 'h', y: -0.2 },
    hoverlabel: {
      bgcolor: '#0b0125',
      font: { color: '#ffffff', family: 'Inter, sans-serif' },
      bordercolor: 'rgba(212, 175, 55, 0.4)'
    }
  };

  // 3. Scatter Plot for Accuracy vs Goal Error
  const errors = usersToShow.map(u => parseFloat(calculateUserStats(state.userScores[u] || {}).avgGoalError));
  const accuracies = usersToShow.map(u => calculateUserStats(state.userScores[u] || {}).acc);
  const meanError = errors.reduce((a, b) => a + b, 0) / (errors.length || 1);
  const meanAccuracy = accuracies.reduce((a, b) => a + b, 0) / (accuracies.length || 1);

  const scatterData = usersToShow.map(u => {
    const stats = calculateUserStats(state.userScores[u] || {});
    return {
      x: [parseFloat(stats.avgGoalError)],
      y: [stats.acc],
      mode: 'markers+text',
      name: u,
      text: [u],
      textposition: 'top center',
      marker: { 
        color: userColors[u], 
        size: 12,
        line: { color: '#1e1e1e', width: 1.5 }
      },
      textfont: {
        family: 'Inter, sans-serif',
        color: '#e5e7eb',
        size: 9
      },
      hovertemplate: `<b>${u}</b><br>Accuracy: ${stats.acc}%<br>Avg Goal Error: ${stats.avgGoalError}<extra></extra>`
    };
  });

  const minErr = errors.length > 0 ? Math.min(...errors) - 0.2 : 0;
  const maxErr = errors.length > 0 ? Math.max(...errors) + 0.2 : 4;
  const minAcc = accuracies.length > 0 ? Math.min(...accuracies) - 5 : 0;
  const maxAcc = accuracies.length > 0 ? Math.max(...accuracies) + 5 : 100;

  const scatterLayout = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#e5e7eb', family: 'Inter, sans-serif' },
    margin: { t: 20, r: 20, b: 45, l: 45 },
    xaxis: { 
      title: 'Average Goal Error (lower is better)', 
      gridcolor: 'rgba(255,255,255,0.1)',
      zerolinecolor: 'rgba(255,255,255,0.2)',
      range: [minErr, maxErr]
    },
    yaxis: { 
      title: 'Accuracy % (higher is better)', 
      gridcolor: 'rgba(255,255,255,0.1)',
      zerolinecolor: 'rgba(255,255,255,0.2)',
      range: [minAcc, maxAcc]
    },
    shapes: [
      {
        type: 'line',
        x0: meanError,
        y0: minAcc,
        x1: meanError,
        y1: maxAcc,
        line: {
          color: 'rgba(255,255,255,0.25)',
          width: 1.5,
          dash: 'dashdot'
        }
      },
      {
        type: 'line',
        x0: minErr,
        y0: meanAccuracy,
        x1: maxErr,
        y1: meanAccuracy,
        line: {
          color: 'rgba(255,255,255,0.25)',
          width: 1.5,
          dash: 'dashdot'
        }
      }
    ],
    showlegend: false,
    hoverlabel: {
      bgcolor: '#0b0125',
      font: { color: '#ffffff', family: 'Inter, sans-serif' },
      bordercolor: 'rgba(212, 175, 55, 0.4)'
    }
  };

  // 5. Advanced Accuracy
  const advData = [];
  const overUnderPct = [];
  const cleanSheetPct = [];
  usersToShow.forEach(u => {
    const stats = calculateUserStats(state.userScores[u] || {});
    overUnderPct.push(stats.overUnderTotal > 0 ? Math.round(stats.overUnderCorrect / stats.overUnderTotal * 100) : 0);
    cleanSheetPct.push(stats.cleanSheetTotal > 0 ? Math.round(stats.cleanSheetCorrect / stats.cleanSheetTotal * 100) : 0);
  });
  
  advData.push({ x: usersToShow, y: overUnderPct, name: 'Over/Under 2.5 (%)', type: 'bar', marker: { color: '#0ea5e9' } });
  advData.push({ x: usersToShow, y: cleanSheetPct, name: 'Clean Sheet (%)', type: 'bar', marker: { color: '#10b981' } });

  const advLayout = {
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    font: { color: '#ffffff', family: 'Inter, sans-serif' },
    barmode: 'group',
    xaxis: { gridcolor: 'rgba(255,255,255,0.1)' },
    yaxis: { title: 'Accuracy (%)', gridcolor: 'rgba(255,255,255,0.1)', range: [0, 100] },
    legend: { orientation: 'h', y: -0.2 }
  };

  // 6. Group vs KO Points
  const phaseData = [];
  const groupPts = [];
  const koPts = [];
  usersToShow.forEach(u => {
    const stats = calculateUserStats(state.userScores[u] || {});
    groupPts.push(stats.groupPoints);
    koPts.push(stats.koPoints);
  });
  
  phaseData.push({ x: usersToShow, y: groupPts, name: 'Group Stage Pts', type: 'bar', marker: { color: '#8b5cf6' } });
  phaseData.push({ x: usersToShow, y: koPts, name: 'Knockout Stage Pts', type: 'bar', marker: { color: '#ec4899' } });

  const phaseLayout = {
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    font: { color: '#ffffff', family: 'Inter, sans-serif' },
    barmode: 'stack',
    xaxis: { gridcolor: 'rgba(255,255,255,0.1)' },
    yaxis: { title: 'Points', gridcolor: 'rgba(255,255,255,0.1)' },
    legend: { orientation: 'h', y: -0.2 }
  };

  // 7. Fun Stats Showcase
  let funStatsHTML = `
    <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 0.85rem; color: #cbd5e1; background: rgba(0,0,0,0.3); border-radius: 6px; overflow: hidden;">
      <thead>
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.1); background: rgba(0,0,0,0.2);">
          <th style="padding: 12px 10px; font-weight: 500;">User</th>
          <th style="padding: 12px 10px; font-weight: 500;">Accuracy</th>
          <th style="padding: 12px 10px; font-weight: 500;">Fav Scoreline</th>
          <th style="padding: 12px 10px; font-weight: 500;">Median Scoreline</th>
          <th style="padding: 12px 10px; font-weight: 500;">Shootouts</th>
          <th style="padding: 12px 10px; font-weight: 500;">Total Goals</th>
          <th style="padding: 12px 10px; font-weight: 500;">Draw Rate</th>
          <th style="padding: 12px 10px; font-weight: 500;">Craziest Pred</th>
        </tr>
      </thead>
      <tbody>
  `;
  
  usersToShow.forEach(u => {
    const stats = calculateUserStats(state.userScores[u] || {});
    funStatsHTML += `
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.05); transition: background-color 0.2s;">
          <td style="padding: 10px; color: ${userColors[u]}; font-weight: 600; font-size: 0.95rem;">${u}</td>
          <td style="padding: 10px; color: #4ade80;">${stats.acc}%</td>
          <td style="padding: 10px; color: #fcd34d;">${stats.favoriteScoreline}</td>
          <td style="padding: 10px; color: #38bdf8;">${stats.medianScoreline}</td>
          <td style="padding: 10px; color: #6ee7b7;">${stats.shootoutsCorrect}/${stats.shootoutsTotal}</td>
          <td style="padding: 10px; color: #fb7185;">${stats.totalGoalsPredicted}</td>
          <td style="padding: 10px; color: #c084fc;">${stats.drawRate}%</td>
          <td style="padding: 10px; color: #fca5a5;">${stats.craziestPrediction}</td>
        </tr>
    `;
  });
  
  funStatsHTML += `
      </tbody>
    </table>
  `;
  const funStatsElem = document.getElementById("fun-stats-container");
  if (funStatsElem) funStatsElem.innerHTML = funStatsHTML;


  if (typeof Plotly !== 'undefined') {
    Plotly.newPlot('plotly-graph-style', styleData, styleLayout, {responsive: true, displayModeBar: true, displaylogo: false});
    Plotly.newPlot('plotly-graph-groups', radarData, radarLayout, {responsive: true, displayModeBar: true, displaylogo: false});
    Plotly.newPlot('plotly-graph-error', scatterData, scatterLayout, {responsive: true, displayModeBar: true, displaylogo: false});
    Plotly.newPlot('plotly-graph-adv-acc', advData, advLayout, {responsive: true, displayModeBar: true, displaylogo: false});
    Plotly.newPlot('plotly-graph-phase-points', phaseData, phaseLayout, {responsive: true, displayModeBar: true, displaylogo: false});
  }

  // View Checkboxes Toggle
  const chkSimple = document.getElementById("simplified-view-chk");
  const chkGraphOnly = document.getElementById("graph-only-chk");
  const graphsCard = document.getElementById("graphs-card");
  const tableCard = document.getElementById("table-card");

  if (chkSimple && chkGraphOnly && graphsCard && tableCard) {
    const savedSimple = localStorage.getItem("matrix_simplified_view") === "true";
    const savedGraphOnly = localStorage.getItem("matrix_graph_only_view") === "true";

    chkSimple.checked = savedSimple;
    chkGraphOnly.checked = savedGraphOnly;
    
    if (chkSimple.checked && chkGraphOnly.checked) {
      chkGraphOnly.checked = false;
    }

    const updateVisibility = () => {
      if (chkSimple.checked) {
        graphsCard.style.display = "none";
        tableCard.style.display = "block";
        tableCard.style.flex = "1 1 100%";
      } else if (chkGraphOnly.checked) {
        tableCard.style.display = "none";
        graphsCard.style.display = "block";
        graphsCard.style.flex = "1 1 100%";
        setTimeout(resizeGraphs, 50);
      } else {
        tableCard.style.display = "block";
        graphsCard.style.display = "block";
        tableCard.style.flex = "1";
        graphsCard.style.flex = "1";
        setTimeout(resizeGraphs, 50);
      }
    };

    const resizeGraphs = () => {
      ["plotly-graph", "plotly-graph-style", "plotly-graph-groups", "plotly-graph-error", "plotly-graph-adv-acc", "plotly-graph-phase-points"].forEach(id => {
        const el = document.getElementById(id);
        if (el && typeof Plotly !== 'undefined') {
          Plotly.Plots.resize(el);
        }
      });
    };

    chkSimple.onchange = () => {
      if (chkSimple.checked) chkGraphOnly.checked = false;
      localStorage.setItem("matrix_simplified_view", chkSimple.checked);
      localStorage.setItem("matrix_graph_only_view", chkGraphOnly.checked);
      updateVisibility();
    };

    chkGraphOnly.onchange = () => {
      if (chkGraphOnly.checked) chkSimple.checked = false;
      localStorage.setItem("matrix_simplified_view", chkSimple.checked);
      localStorage.setItem("matrix_graph_only_view", chkGraphOnly.checked);
      updateVisibility();
    };

    updateVisibility();
  }

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
      <td class="num">${u.avgGoalError}</td>
      <td class="num">${u.predictedMatches} / 104</td>
    </tr>
  `).join("");
  
  const podiumContainer = document.getElementById("podium-container");
  if (podiumContainer) {
    const realUsers = userStats.filter(u => u.name !== "Actual Results");
    if (realUsers.length >= 3) {
      const top1 = realUsers[0];
      const top2 = realUsers[1];
      const top3 = realUsers[2];
      
      const formatPts = (pts) => pts.toFixed(pts % 1 === 0 ? 0 : 2);
      
      const getUserColor = (u) => {
        if (!u) return '#fff';
        const presetColors = {
          "oneiros": "#facc15", "nithin": "#f87171", "abhi": "#34d399",
          "anuj": "#a78bfa", "karthik": "#fb923c", "tej": "#f472b6", "divyank": "#38bdf8"
        };
        return presetColors[u.toLowerCase()] || '#fff';
      };
      
      podiumContainer.innerHTML = `
        <div style="display: flex; flex-direction: column; align-items: center; width: 30%; animation: fadeInGroup 0.5s ease backwards; animation-delay: 0.2s;">
          <span style="font-size: 1.2rem; margin-bottom: 5px;">🥈</span>
          <span style="color: ${getUserColor(top2.name)}; font-weight: bold; margin-bottom: 5px; text-align: center;">${top2.name}</span>
          <span style="color: #e9bc3f; font-size: 0.9rem; margin-bottom: 10px;">${formatPts(top2.totalPoints)} pts</span>
          <div style="width: 100%; height: 120px; background: linear-gradient(to top, rgba(14, 43, 92, 0.8), rgba(192, 192, 192, 0.6)); border-radius: 8px 8px 0 0; border: 1px solid silver; border-bottom: none; box-shadow: 0 0 15px rgba(192,192,192,0.2);"></div>
        </div>
        <div style="display: flex; flex-direction: column; align-items: center; width: 35%; animation: fadeInGroup 0.5s ease backwards; animation-delay: 0.4s;">
          <span style="font-size: 1.5rem; margin-bottom: 5px;">🥇</span>
          <span style="color: ${getUserColor(top1.name)}; font-weight: bold; margin-bottom: 5px; text-align: center; font-size: 1.1rem;">${top1.name}</span>
          <span style="color: #e9bc3f; font-weight: bold; margin-bottom: 10px;">${formatPts(top1.totalPoints)} pts</span>
          <div style="width: 100%; height: 160px; background: linear-gradient(to top, rgba(14, 43, 92, 0.8), rgba(212, 175, 55, 0.8)); border-radius: 8px 8px 0 0; border: 1px solid var(--gold); border-bottom: none; box-shadow: 0 0 20px rgba(212,175,55,0.4);"></div>
        </div>
        <div style="display: flex; flex-direction: column; align-items: center; width: 30%; animation: fadeInGroup 0.5s ease backwards; animation-delay: 0.0s;">
          <span style="font-size: 1.2rem; margin-bottom: 5px;">🥉</span>
          <span style="color: ${getUserColor(top3.name)}; font-weight: bold; margin-bottom: 5px; text-align: center;">${top3.name}</span>
          <span style="color: #e9bc3f; font-size: 0.9rem; margin-bottom: 10px;">${formatPts(top3.totalPoints)} pts</span>
          <div style="width: 100%; height: 90px; background: linear-gradient(to top, rgba(14, 43, 92, 0.8), rgba(205, 127, 50, 0.6)); border-radius: 8px 8px 0 0; border: 1px solid #cd7f32; border-bottom: none; box-shadow: 0 0 15px rgba(205,127,50,0.2);"></div>
        </div>
      `;
    }
  }
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
  document.querySelectorAll(".ko-score-input, .ko-score-input-pens").forEach(input => {
    input.addEventListener("input", (e) => {
      const nodeId = e.target.getAttribute("data-node-id");
      const type = e.target.getAttribute("data-type");
      const val = e.target.value === "" ? "" : parseInt(e.target.value);
      
      if (type === "actHome" || type === "actAway" || type === "actHomePens" || type === "actAwayPens") {
        state.users.forEach(u => {
          if (val === "" || isNaN(val)) {
            delete state.userScores[u][nodeId + "_" + type];
          } else {
            state.userScores[u][nodeId + "_" + type] = val;
          }
        });
        state.scores = state.userScores[state.currentUser];
      } else {
        if (val === "" || isNaN(val)) {
          delete state.scores[nodeId + "_" + type];
        } else {
          state.scores[nodeId + "_" + type] = val;
        }
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
    if (nodeId === "SF_1" || nodeId === "SF_2") return null; // These feed the final
    // Everything else flows left to right
    return true; 
  }
  
  // Draw a connector from two feeder nodes into one target node
  function drawConnector(feeder1Id, feeder2Id, targetId) {
    const f1 = getCardPos(feeder1Id);
    const f2 = getCardPos(feeder2Id);
    const t = getCardPos(targetId);
    if (!f1 || !f2 || !t) return;
    
    const leftSide = isLeftSide(targetId === "Final" ? feeder1Id : targetId); // wait, for finals, targetId is Final.
    const lineColor = "rgba(212,175,55,0.35)";
    const lineWidth = 1.5;
    
    if (targetId !== "FINAL" && targetId !== "THIRD") {
      // Left to Right flow for all matches in the halves
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
      path1.classList.add("animated-connector");
      svg.appendChild(path1);
      
      // Feeder 2 → merge point
      const path2 = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path2.setAttribute("d", `M${startX},${f2.centerY} H${midX} V${t.centerY}`);
      path2.setAttribute("stroke", lineColor);
      path2.setAttribute("stroke-width", lineWidth);
      path2.setAttribute("fill", "none");
      path2.classList.add("animated-connector");
      svg.appendChild(path2);
      
    } else {
      // User requested to remove the lines going from semi finals to the finals
      // So we don't draw anything here.
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
        <input type="number" min="0" placeholder="p" class="ko-score-input-pens pred ko-pred-home-pens" 
          data-node-id="${m.node_id}" data-type="predHomePens" style="display: none;" title="Penalties (Prediction)"
          value="${state.scores[m.node_id + '_predHomePens'] !== undefined ? state.scores[m.node_id + '_predHomePens'] : ''}">
        <input type="number" min="0" placeholder="A" class="ko-score-input actual ko-act-home" 
          data-node-id="${m.node_id}" data-type="actHome"
          value="${state.scores[m.node_id + '_actHome'] !== undefined ? state.scores[m.node_id + '_actHome'] : ''}">
        <input type="number" min="0" placeholder="p" class="ko-score-input-pens actual ko-act-home-pens" 
          data-node-id="${m.node_id}" data-type="actHomePens" style="display: none;" title="Penalties (Actual)"
          value="${state.scores[m.node_id + '_actHomePens'] !== undefined ? state.scores[m.node_id + '_actHomePens'] : ''}">
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
        <input type="number" min="0" placeholder="p" class="ko-score-input-pens pred ko-pred-away-pens" 
          data-node-id="${m.node_id}" data-type="predAwayPens" style="display: none;" title="Penalties (Prediction)"
          value="${state.scores[m.node_id + '_predAwayPens'] !== undefined ? state.scores[m.node_id + '_predAwayPens'] : ''}">
        <input type="number" min="0" placeholder="A" class="ko-score-input actual ko-act-away" 
          data-node-id="${m.node_id}" data-type="actAway"
          value="${state.scores[m.node_id + '_actAway'] !== undefined ? state.scores[m.node_id + '_actAway'] : ''}">
        <input type="number" min="0" placeholder="p" class="ko-score-input-pens actual ko-act-away-pens" 
          data-node-id="${m.node_id}" data-type="actAwayPens" style="display: none;" title="Penalties (Actual)"
          value="${state.scores[m.node_id + '_actAwayPens'] !== undefined ? state.scores[m.node_id + '_actAwayPens'] : ''}">
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
  const renderTeamInfo = (t, nodeId, teamIdx) => {
    const isPlaceholder = t.startsWith("Winner") || t.startsWith("Loser") || t.match(/^\d/) || t.startsWith("3");
    const overrideKey = `${nodeId}_overrideTeam${teamIdx}`;
    const hasOverride = state.userScores[state.users[0]][overrideKey] !== undefined;

    let options = `<option value="">-- Auto${isPlaceholder ? ` (${t})` : ''} --</option>`;
    Object.keys(teamFlags).sort().forEach(teamName => {
      options += `<option value="${teamName}" ${teamName === t && !isPlaceholder ? 'selected' : ''}>${teamName}</option>`;
    });

    const isEditable = config.isLocal && state.currentUser === state.users[0];
    const selectHtml = `
      <select class="ko-team-override-select" data-node-id="${nodeId}" data-team-idx="${teamIdx}" ${!isEditable ? 'disabled' : ''}
        title="${!isEditable ? 'Switch to ' + state.users[0] + ' to override' : 'Override Team'}"
        style="background: transparent; border: 1px dashed ${hasOverride ? '#3b82f6' : 'rgba(255,255,255,0.2)'}; 
        color: ${isPlaceholder ? '#94a3b8' : 'white'}; outline: none; padding: 2px; border-radius: 4px; 
        font-family: inherit; font-size: 0.9rem; cursor: ${!isEditable ? 'not-allowed' : 'pointer'}; max-width: 100%; text-overflow: ellipsis;">
        ${options}
      </select>
    `;

    if (isPlaceholder) {
      return selectHtml;
    } else {
      return `
        <img src="${getFlagUrl(t)}" alt="">
        ${selectHtml}
      `;
    }
  };

    team1Info.innerHTML = renderTeamInfo(teamsObj.team1, nodeId, 1);
    team2Info.innerHTML = renderTeamInfo(teamsObj.team2, nodeId, 2);
    
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
        else {
          const predHomePens = state.scores[nodeId + "_predHomePens"];
          const predAwayPens = state.scores[nodeId + "_predAwayPens"];
          if (predHomePens !== undefined && predAwayPens !== undefined) {
            if (predHomePens > predAwayPens) row1.classList.add("winner-predicted");
            else if (predHomePens < predAwayPens) row2.classList.add("winner-predicted");
          }
        }
      }
      
      // Toggle penalty inputs visibility
      const predHomePensInput = document.querySelector(`.ko-pred-home-pens[data-node-id="${nodeId}"]`);
      const predAwayPensInput = document.querySelector(`.ko-pred-away-pens[data-node-id="${nodeId}"]`);
      if (predHomePensInput && predAwayPensInput) {
        const showPredPens = predHome !== undefined && predAway !== undefined && predHome === predAway;
        predHomePensInput.style.display = showPredPens ? "inline-block" : "none";
        predAwayPensInput.style.display = showPredPens ? "inline-block" : "none";
      }

      const actHome = state.scores[nodeId + "_actHome"];
      const actAway = state.scores[nodeId + "_actAway"];
      const actHomePensInput = document.querySelector(`.ko-act-home-pens[data-node-id="${nodeId}"]`);
      const actAwayPensInput = document.querySelector(`.ko-act-away-pens[data-node-id="${nodeId}"]`);
      if (actHomePensInput && actAwayPensInput) {
        const showActPens = actHome !== undefined && actAway !== undefined && actHome === actAway;
        actHomePensInput.style.display = showActPens ? "inline-block" : "none";
        actAwayPensInput.style.display = showActPens ? "inline-block" : "none";
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
            state.users = imported.users || ["Actual Results"];
            state.currentUser = imported.currentUser || "Actual Results";
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
  
  // Add delegated listener for knockout team overrides
  document.addEventListener("change", (e) => {
    if (e.target && e.target.classList.contains("ko-team-override-select")) {
      const nodeId = e.target.getAttribute("data-node-id");
      const teamIdx = e.target.getAttribute("data-team-idx");
      const val = e.target.value;
      const overrideKey = `${nodeId}_overrideTeam${teamIdx}`;
      
      if (val === "") {
        delete state.scores[overrideKey];
      } else {
        state.scores[overrideKey] = val;
      }
      
      saveStateToLocalStorage();
      updateScoresAndStandings();
    }
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

// --- UI Event Listeners ---
document.addEventListener('DOMContentLoaded', () => {
  const toggle = document.getElementById('toggle-spikeline');
  if (toggle) {
    toggle.addEventListener('change', (e) => {
      if (typeof Plotly !== 'undefined') {
        Plotly.relayout('plotly-graph', { 
          'xaxis.showspikes': e.target.checked,
          'hovermode': e.target.checked ? 'closest' : false
        });
        if (document.getElementById('plotly-graph-gap')) {
          Plotly.relayout('plotly-graph-gap', {
            'xaxis.showspikes': e.target.checked,
            'hovermode': e.target.checked ? 'closest' : false
          });
        }
      }
    });
  }

  const stSelect = document.getElementById('standings-type-select');
  if (stSelect) {
    stSelect.addEventListener('change', () => {
      updateScoresAndStandings();
    });
  }

  const btnAnimate = document.getElementById('btn-animate-graph');
  if (btnAnimate) {
    btnAnimate.addEventListener('click', () => {
      if (typeof Plotly !== 'undefined' && window.plotlyFrames && window.plotlyFrames.length > 0) {
        const frameNames = window.plotlyFrames.map(f => f.name);
        
        // Reset to first frame instantly
        const animateMain = Plotly.animate('plotly-graph', [frameNames[0]], {
          transition: { duration: 0 },
          frame: { duration: 0, redraw: true },
          mode: 'immediate'
        });

        let animateGap = Promise.resolve();
        if (document.getElementById('plotly-graph-gap') && window.plotlyGapFrames) {
          animateGap = Plotly.animate('plotly-graph-gap', [frameNames[0]], {
            transition: { duration: 0 },
            frame: { duration: 0, redraw: false },
            mode: 'immediate'
          });
        }

        Promise.all([animateMain, animateGap]).then(() => {
          // Play sequence
          Plotly.animate('plotly-graph', frameNames, {
            transition: { duration: 10000 / frameNames.length, easing: 'linear' },
            frame: { duration: 10000 / frameNames.length, redraw: true },
            mode: 'immediate'
          });

          if (document.getElementById('plotly-graph-gap') && window.plotlyGapFrames) {
            Plotly.animate('plotly-graph-gap', frameNames, {
              transition: { duration: 10000 / frameNames.length, easing: 'linear' },
              frame: { duration: 10000 / frameNames.length, redraw: false },
              mode: 'immediate'
            });
          }
        });
      }
    });
  }
});

// --- API-Football Integration ---
function normalizeTeamName(name) {
  if (!name) return "";
  const n = name.toLowerCase().trim();
  const map = {
    "united states": "usa",
    "korea republic": "south korea",
    "south korea": "south korea",
    "czech republic": "czech rep.",
    "czech rep": "czech rep.",
    "iran": "ir iran",
    "saudi arabia": "saudi arabia",
    "congo dr": "dr congo",
    "dr congo": "dr congo",
    "bosnia and herzegovina": "bosnia-herz.",
    "bosnia-herz.": "bosnia-herz.",
    "curacao": "curaçao",
    "curaçao": "curaçao",
    "turkey": "türkiye",
    "türkiye": "türkiye"
  };
  return map[n] || n;
}

async function fetchLiveScores(silent = false) {
  if (!silent) {
    console.log("Fetching live scores from ESPN API...");
  }

  try {
    // Switch to ESPN API (100% Free, No Key Required, Fetches whole tournament range)
    const res = await fetch("https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard?dates=20260611-20260719&limit=1000");
    
    if (!res.ok) {
      throw new Error(`ESPN API returned status ${res.status}`);
    }

    const data = await res.json();
    
    if (!data.events || data.events.length === 0) {
       if (!silent) alert("ESPN API returned 0 fixtures for the current window.");
       return false;
    }

    let updatedCount = 0;
    
    // Flatten our initialMatchesData
    const allMatches = [];
    Object.keys(initialMatchesData.groups).forEach(g => {
       initialMatchesData.groups[g].forEach(m => allMatches.push(m));
    });
    if (initialMatchesData.r32) initialMatchesData.r32.forEach(m => allMatches.push(m));
    if (initialMatchesData.knockouts) initialMatchesData.knockouts.forEach(m => allMatches.push(m));
    
    data.events.forEach(event => {
      const comp = event.competitions[0];
      const statusName = comp.status.type.name; // e.g., "STATUS_FINAL"
      
      // Look for matches that have started (live or finished)
      if (statusName !== 'STATUS_SCHEDULED' && statusName !== 'STATUS_POSTPONED' && statusName !== 'STATUS_CANCELED') {
        let homeTeam = null;
        let awayTeam = null;
        let homeScore = 0;
        let awayScore = 0;

        comp.competitors.forEach(team => {
          if (team.homeAway === 'home') {
            homeTeam = team.team.displayName;
            homeScore = parseInt(team.score, 10) || 0;
          } else {
            awayTeam = team.team.displayName;
            awayScore = parseInt(team.score, 10) || 0;
          }
        });

        const hNameAPI = normalizeTeamName(homeTeam);
        const aNameAPI = normalizeTeamName(awayTeam);
        
        let matchId = null;
        for (const mData of allMatches) {
          let dashboardHome = mData.team1 || mData.team1_placeholder || "";
          let dashboardAway = mData.team2 || mData.team2_placeholder || "";
          
          if (mData.node_id && globalKoTeams[mData.node_id]) {
            dashboardHome = globalKoTeams[mData.node_id].team1 || dashboardHome;
            dashboardAway = globalKoTeams[mData.node_id].team2 || dashboardAway;
          }
          
          if (!dashboardHome || !dashboardAway) continue;
          
          if (normalizeTeamName(dashboardHome) === hNameAPI && normalizeTeamName(dashboardAway) === aNameAPI) {
            matchId = mData.id;
            break;
          }
        }
        
        if (matchId) {
           let wasUpdated = false;
           Object.keys(state.userScores).forEach(u => {
             if (state.userScores[u][matchId + "_actHome"] !== homeScore ||
                 state.userScores[u][matchId + "_actAway"] !== awayScore) {
                 
                 state.userScores[u][matchId + "_actHome"] = homeScore;
                 state.userScores[u][matchId + "_actAway"] = awayScore;
                 wasUpdated = true;
             }
           });
           
           if (wasUpdated) {
             updatedCount++;
             if (state.scores) {
               state.scores[matchId + "_actHome"] = homeScore;
               state.scores[matchId + "_actAway"] = awayScore;
             }
             const domHome = document.querySelector(`input.act-home[data-match-id="${matchId}"]`);
             const domAway = document.querySelector(`input.act-away[data-match-id="${matchId}"]`);
             if (domHome) domHome.value = homeScore;
             if (domAway) domAway.value = awayScore;
           }
        }
      }
    });
    
    if (updatedCount > 0) {
       updateScoresAndStandings();
       if (!silent) alert(`Successfully fetched and updated ${updatedCount} matches from ESPN!`);
       await pushToGithub(silent); // <--- Added Auto Push
       return true;
    } else {
       if (!silent) alert("No new finished matches found on ESPN.");
       return false;
    }
    
  } catch (err) {
    if (!silent) alert("Error fetching scores: " + err.message);
    return false;
  }
}

// ---------------------- BACK TO TOP BUTTON ----------------------
(function() {
  const btn = document.getElementById("back-to-top-btn");
  if (!btn) return;
  
  window.addEventListener("scroll", () => {
    if (window.scrollY > 400) {
      btn.style.display = "flex";
    } else {
      btn.style.display = "none";
    }
  });
  
  btn.addEventListener("click", () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
})();
