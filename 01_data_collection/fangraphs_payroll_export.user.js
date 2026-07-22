// ==UserScript==
// @name         FanGraphs RosterResource Payroll CSV Export
// @namespace    mlb-salary-homework
// @version      1.0.1
// @description  Save one public FanGraphs RosterResource team-season payroll page as CSV.
// @match        https://www.fangraphs.com/roster-resource/payroll/*
// @grant        none
// ==/UserScript==

(function () {
  "use strict";

  const PANEL_ID = "mlb-salary-payroll-export-panel";
  const PLAYER_GROUPS = new Set([
    "Players With Guaranteed Salaries",
    "Players Eligible For Arbitration",
    "Notable Players Not Yet Eligible For Arbitration",
    "Players No Longer On 40-Man Roster or No Longer In Organization",
  ]);
  const TEAM_CODES = {
    angels: "LAA", astros: "HOU", athletics: "ATH", "blue-jays": "TOR",
    braves: "ATL", brewers: "MIL", cardinals: "STL", cubs: "CHC",
    diamondbacks: "ARI", dodgers: "LAD", giants: "SFG", guardians: "CLE",
    mariners: "SEA", marlins: "MIA", mets: "NYM", nationals: "WSN",
    orioles: "BAL", padres: "SDP", phillies: "PHI", pirates: "PIT",
    rangers: "TEX", rays: "TBR", "red-sox": "BOS", reds: "CIN",
    rockies: "COL", royals: "KCR", tigers: "DET", twins: "MIN",
    "white-sox": "CHW", yankees: "NYY",
  };
  const TEAM_HEADING_WORDS = {
    angels: ["angels"], astros: ["astros"], athletics: ["athletics"], "blue-jays": ["blue jays"],
    braves: ["braves"], brewers: ["brewers"], cardinals: ["cardinals"], cubs: ["cubs"],
    diamondbacks: ["diamondbacks"], dodgers: ["dodgers"], giants: ["giants"], guardians: ["guardians", "indians"],
    mariners: ["mariners"], marlins: ["marlins"], mets: ["mets"], nationals: ["nationals"],
    orioles: ["orioles"], padres: ["padres"], phillies: ["phillies"], pirates: ["pirates"],
    rangers: ["rangers"], rays: ["rays"], "red-sox": ["red sox"], reds: ["reds"],
    rockies: ["rockies"], royals: ["royals"], tigers: ["tigers"], twins: ["twins"],
    "white-sox": ["white sox"], yankees: ["yankees"],
  };

  function clean(value) {
    return String(value ?? "")
      .replace(/\u00a0/g, " ")
      .replace(/\s+/g, " ")
      .replace(/▼/g, "")
      .trim();
  }

  function csvEscape(value) {
    const text = String(value ?? "");
    return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  }

  function isVisible(element) {
    const style = window.getComputedStyle(element);
    const rect = element.getBoundingClientRect();
    return style.display !== "none" && style.visibility !== "hidden" && rect.width > 0 && rect.height > 0;
  }

  function pageInfo() {
    const url = new URL(window.location.href);
    const parts = url.pathname.split("/").filter(Boolean);
    const slug = parts[parts.length - 1] || "unknown-team";
    const season = url.searchParams.get("season") || clean(document.querySelector('[class*="season"]')?.textContent).match(/20\d{2}/)?.[0] || "";
    return { slug, team: TEAM_CODES[slug] || slug.toUpperCase(), season };
  }

  function tableHeaders(table) {
    const row = table.querySelector("thead tr") || table.querySelector("tr");
    return row ? Array.from(row.querySelectorAll("th,td")).map((cell) => clean(cell.textContent)) : [];
  }

  function precedingSection(table) {
    let current = "";
    for (const element of document.querySelectorAll("h2,h3,table")) {
      if (element === table) return current;
      if (element.matches("h2,h3")) current = clean(element.textContent);
    }
    return current;
  }

  function playerIdFromLink(link) {
    if (!link) return "";
    const href = link.getAttribute("href") || "";
    const modern = href.match(/\/players\/[^/]+\/(\d+)(?:\/|$)/i);
    if (modern) return modern[1];
    try {
      return new URL(href, window.location.origin).searchParams.get("playerid") || "";
    } catch (_) {
      return "";
    }
  }

  function moneyToNumber(raw) {
    const text = clean(raw).toUpperCase();
    if (!text || !/[0-9]/.test(text)) return "";
    const match = text.match(/-?\$?\s*([0-9,.]+)\s*([KM])?/);
    if (!match) return "";
    let value = Number(match[1].replace(/,/g, ""));
    if (!Number.isFinite(value)) return "";
    if (match[2] === "K") value *= 1_000;
    if (match[2] === "M") value *= 1_000_000;
    return Math.round(value);
  }

  function extractRows() {
    const { slug, team, season } = pageInfo();
    if (!/^20\d{2}$/.test(season)) {
      throw new Error("URL에서 시즌을 찾지 못했습니다. season=2019 같은 주소인지 확인하세요.");
    }
    const payrollHeading = Array.from(document.querySelectorAll("h2"))
      .map((heading) => clean(heading.textContent).toLowerCase())
      .find((text) => text.includes("player payroll")) || "";
    const expectedWords = TEAM_HEADING_WORDS[slug] || [];
    if (expectedWords.length && !expectedWords.some((word) => payrollHeading.includes(word))) {
      throw new Error("팀 주소와 표 내용이 다릅니다. 새 팀 표가 완전히 로딩된 뒤 다시 누르세요.");
    }

    const output = [];
    const seen = new Set();
    const tables = Array.from(document.querySelectorAll("table")).filter(isVisible);

    for (const table of tables) {
      const headers = tableHeaders(table);
      const group = precedingSection(table);
      if (!PLAYER_GROUPS.has(group) || !headers.includes("Player") || !headers.includes(season)) continue;

      const indices = Object.fromEntries(headers.map((header, index) => [header, index]));
      for (const row of table.querySelectorAll("tbody tr")) {
        const cells = Array.from(row.querySelectorAll("td"));
        if (cells.length < headers.length) continue;

        const playerCell = cells[indices.Player];
        const playerLink = playerCell?.querySelector('a[href*="/players/"], a[href*="playerid="]');
        const playerid = playerIdFromLink(playerLink);
        const player = clean(playerLink?.textContent || playerCell?.textContent);
        if (!player || !playerid) continue;

        const salaryCell = cells[indices[season]];
        const salaryRaw = clean(salaryCell?.textContent);
        const key = `${season}|${team}|${playerid}|${group}|${salaryRaw}`;
        if (seen.has(key)) continue;
        seen.add(key);

        const valueAt = (name) => indices[name] === undefined ? "" : clean(cells[indices[name]]?.textContent);
        output.push({
          Season: season,
          Team: team,
          Team_Slug: slug,
          Player: player,
          playerid,
          contract_group: group,
          Age: valueAt("Age"),
          Service_Time: valueAt("Service Time"),
          Contract: valueAt("Contract"),
          Info: valueAt("Info"),
          AAV: moneyToNumber(valueAt("AAV")),
          AAV_raw: valueAt("AAV"),
          Salary: moneyToNumber(salaryRaw),
          Salary_raw: salaryRaw,
          Salary_type: clean(salaryCell?.dataset.contractColor),
          _source_url: window.location.href,
          _collected_at: new Date().toISOString(),
        });
      }
    }
    return { rows: output, slug, team, season };
  }

  function saveCsv() {
    try {
      const result = extractRows();
      if (!result.rows.length) {
        throw new Error("선수 급여 행을 찾지 못했습니다. 페이지 로딩이 끝난 뒤 다시 누르세요.");
      }
      const headers = Object.keys(result.rows[0]);
      const lines = [headers.map(csvEscape).join(",")];
      for (const row of result.rows) lines.push(headers.map((header) => csvEscape(row[header])).join(","));

      const filename = `fangraphs_payroll_${result.team}_${result.season}.csv`;
      const blob = new Blob([`\uFEFF${lines.join("\r\n")}`], { type: "text/csv;charset=utf-8" });
      const objectUrl = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = objectUrl;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(objectUrl);
      setStatus(`${result.rows.length.toLocaleString()}명 저장 완료: ${filename}`, false);
    } catch (error) {
      setStatus(error.message || String(error), true);
    }
  }

  function setStatus(message, isError) {
    const status = document.querySelector(`#${PANEL_ID} [data-role="status"]`);
    if (!status) return;
    status.textContent = message;
    status.style.color = isError ? "#b91c1c" : "#166534";
  }

  function addPanel() {
    if (document.getElementById(PANEL_ID)) return;
    const panel = document.createElement("div");
    panel.id = PANEL_ID;
    panel.style.cssText = [
      "position:fixed", "right:18px", "bottom:18px", "z-index:2147483647",
      "width:310px", "padding:12px", "background:#fff", "border:1px solid #9ca3af",
      "border-radius:8px", "box-shadow:0 4px 16px rgba(0,0,0,.18)",
      "font:13px/1.4 Arial,sans-serif", "color:#111827",
    ].join(";");

    const button = document.createElement("button");
    button.type = "button";
    button.textContent = "FG 현재 팀·시즌 Payroll CSV 저장";
    button.style.cssText = [
      "width:100%", "padding:9px 10px", "border:0", "border-radius:6px",
      "background:#2563eb", "color:white", "font-weight:700", "cursor:pointer",
    ].join(";");
    button.addEventListener("click", saveCsv);

    const note = document.createElement("div");
    note.textContent = "현재 열어 둔 팀과 시즌 1개만 저장합니다. 비회원 공개 화면에서 작동합니다.";
    note.style.cssText = "margin-top:8px;color:#4b5563";

    const status = document.createElement("div");
    status.dataset.role = "status";
    status.textContent = "표 로딩 완료 후 저장 버튼을 누르세요.";
    status.style.cssText = "margin-top:7px;font-size:12px;color:#4b5563";

    panel.append(button, note, status);
    document.body.appendChild(panel);
  }

  addPanel();
})();
