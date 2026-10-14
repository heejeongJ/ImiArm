"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const JOINT_KO = { shoulder_pan: "어깨 회전", shoulder_lift: "어깨 들기", elbow_flex: "팔꿈치", wrist_flex: "손목 굽힘", wrist_roll: "손목 회전", gripper: "그리퍼" };
const POLICY_KO = { act: "ACT", smolvla: "SmolVLA", diffusion: "Diffusion" };
const pad3 = (n) => String(n).padStart(3, "0");
const sec = (s) => (s == null ? "-" : `${s.toFixed(1)}초`);
const num = (n) => (n == null ? "-" : Number(n).toLocaleString("ko-KR"));

// API 호출, 실패 시 서버 오류 메시지로 예외
async function api(path, body) {
  const opt = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const r = await fetch(path, opt);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || `요청 실패 (${r.status})`);
  return data;
}

// 버튼 동작용 API 호출, 실패 시 알림 후 null 반환
async function act(path, body = {}) {
  try {
    return await api(path, body);
  } catch (e) {
    toast(e.message);
    return null;
  }
}

// 화면 하단 알림
let toastTimer;
function toast(msg) {
  let t = $(".toast");
  if (!t) {
    t = document.createElement("div");
    t.className = "toast";
    t.setAttribute("role", "status");
    document.body.append(t);
  }
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), 4000);
}

// 겹치지 않는 주기 호출, 화면 이탈 시 중지
function poll(fn, ms) {
  let stop = false, timer;
  const loop = async () => {
    if (stop) return;
    try { await fn(); } catch (e) { console.warn(e); }
    if (!stop) timer = setTimeout(loop, ms);
  };
  loop();
  onLeave(() => { stop = true; clearTimeout(timer); });
}

// 값 배열을 SVG path로 변환, null은 선 끊김
function linePath(xs, ys, { w, h, x0, x1, y0, y1 }) {
  let d = "", pen = false;
  for (let i = 0; i < xs.length; i++) {
    const y = ys[i];
    if (y == null || !isFinite(y)) { pen = false; continue; }
    const X = ((xs[i] - x0) / (x1 - x0 || 1)) * w;
    const Y = h - ((y - y0) / (y1 - y0 || 1)) * h;
    d += `${pen ? "L" : "M"}${X.toFixed(1)},${Y.toFixed(1)}`;
    pen = true;
  }
  return d;
}

// 그래프 세로축 범위, 최소 폭과 여백 보장
function extent(arrs, minSpan) {
  const v = arrs.flat().filter((x) => x != null && isFinite(x));
  if (!v.length) return [0, minSpan];
  let lo = Math.min(...v), hi = Math.max(...v);
  if (hi - lo < minSpan) { const m = (hi + lo) / 2; lo = m - minSpan / 2; hi = m + minSpan / 2; }
  const p = (hi - lo) * 0.12;
  return [lo - p, hi + p];
}

// localStorage 읽기/쓰기, 막힌 환경에서도 동작
function storeGet(k, d) { try { return localStorage.getItem(k) ?? d; } catch { return d; } }
function storeSet(k, v) { try { localStorage.setItem(k, v); } catch {} }

// 화면을 떠날 때 실행할 정리 함수
let leaving = [];
const onLeave = (fn) => leaving.push(fn);
let overview = null;

const pages = { "/": operatePage, "/data": dataPage, "/train": trainPage, "/eval": evalPage, "/sim": simPage, "/calib": calibPage };

// 주소에 맞는 화면 그리기, 이전 화면 정리
function route() {
  leaving.forEach((f) => f());
  leaving = [];
  const path = pages[location.pathname] ? location.pathname : "/";
  $$("nav a").forEach((a) => {
    const on = a.getAttribute("href") === path;
    a.classList.toggle("on", on);
    if (on) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  });
  $("#devices-toggle").hidden = path !== "/";
  const el = $("#page");
  el.innerHTML = "";
  pages[path](el);
}

// 화면 이동
function go(href) {
  history.pushState(null, "", href);
  route();
}

// 내부 링크는 새로고침 없이 화면 이동
document.addEventListener("click", (e) => {
  const a = e.target.closest("a[href^='/']");
  if (!a || e.ctrlKey || e.metaKey || e.shiftKey || a.target || a.hasAttribute("download")) return;
  e.preventDefault();
  go(a.getAttribute("href"));
});
window.addEventListener("popstate", route);

// 비상 정지 버튼과 Esc 키
$("#estop").addEventListener("click", async () => {
  if (await act("/api/estop")) toast("비상 정지: 동작을 멈추고 지금 자세를 유지합니다");
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") $("#estop").click();
});

// 조작 화면: 장치 연결, 카메라, 관절 추종, 녹화
function operatePage(el) {
  const ov = overview;
  el.innerHTML = `
  <section id="devices" class="card" aria-label="장비 구성" hidden>
    <div class="card-head"><div class="left"><div class="title">장비 구성</div>
      <div class="sub">로봇 암 2대, 카메라 ${ov.cameras.length}대</div></div>
      <div class="note">config.py 기준</div></div>
    <div class="table-wrap"><table class="tbl" style="min-width:720px"><thead><tr>
      <th>장치</th><th>역할</th><th>포트</th><th>구성</th><th>설정</th></tr></thead><tbody>
      ${ov.devices.map((d) => `<tr><td>${esc(d.name)}</td><td>${esc(d.role)}</td><td class="mono" style="color:var(--text)">${esc(d.port)}</td><td>${esc(d.spec)}</td><td>${esc(d.setting)}</td></tr>`).join("")}
    </tbody></table></div>
  </section>
  <div class="row">
    <main class="col" style="flex:999 1 640px">
      <section class="row">
        ${ov.cameras.map((c) => `
        <div class="card tight" style="flex:1 1 300px">
          <div class="card-head" style="margin-bottom:12px"><div class="title">카메라 ${esc(c)}</div><div class="note">${ov.devices.find((d) => d.name === "카메라 " + c)?.setting ?? ""}</div></div>
          <div class="media" data-cam="${esc(c)}">장치 연결 안 됨</div>
        </div>`).join("")}
      </section>
      <section class="card">
        <div class="card-head" style="align-items:center">
          <div class="left" style="align-items:center;gap:18px"><div class="title">관절 추종</div>
            <div class="note" style="display:flex;gap:14px;color:var(--text-2)">
              <span style="display:flex;align-items:center;gap:6px"><span style="width:14px;border-top:2px dashed var(--gray-line)"></span>목표각 (리더·정책)</span>
              <span style="display:flex;align-items:center;gap:6px"><span style="width:14px;height:2px;background:var(--blue)"></span>실제각</span>
            </div></div>
          <div class="seg" id="joint-tabs" role="tablist">${ov.joints.map((j) => `<button type="button" role="tab" data-j="${j}">${JOINT_KO[j]}</button>`).join("")}</div>
        </div>
        <svg class="chart" viewBox="0 0 600 150" preserveAspectRatio="none" style="height:170px" role="img" aria-label="관절 목표각과 실제각 시계열">
          ${[37, 75, 112].map((y) => `<line x1="0" y1="${y}" x2="600" y2="${y}" stroke="#EEF0F3" vector-effect="non-scaling-stroke"/>`).join("")}
          <path id="j-target" fill="none" stroke="#9CA3AF" stroke-width="1.5" stroke-dasharray="5 4" vector-effect="non-scaling-stroke"/>
          <path id="j-actual" fill="none" stroke="#2563EB" stroke-width="2" vector-effect="non-scaling-stroke"/>
        </svg>
        <div class="between note" style="margin-top:8px"><span>10초 전</span><span id="j-err">-</span><span>현재</span></div>
      </section>
      <section class="row">
        <div class="card" style="flex:1 1 300px">
          <div class="stat-label" id="lat-label">제어 주기</div>
          <div class="stat-value" id="lat-value">-</div>
          <svg class="chart" viewBox="0 0 300 80" preserveAspectRatio="none" style="height:90px" role="img" aria-label="지연 시계열">
            <line id="lat-limit" x1="0" x2="300" y1="20" y2="20" stroke="#9CA3AF" stroke-dasharray="3 4" vector-effect="non-scaling-stroke"/>
            <path id="lat-line" fill="none" stroke="#2563EB" stroke-width="2" vector-effect="non-scaling-stroke"/>
          </svg>
          <div class="note" style="margin-top:8px">점선은 제어 주기 한계 (${(1000 / ov.fps).toFixed(0)}ms)</div>
        </div>
        <div class="card" style="flex:1 1 300px">
          <div class="stat-label">그리퍼 부하</div>
          <div class="stat-value" id="load-value">-</div>
          <svg class="chart" viewBox="0 0 300 80" preserveAspectRatio="none" style="height:90px" role="img" aria-label="그리퍼 서보 부하 시계열">
            <path id="load-line" fill="none" stroke="#2563EB" stroke-width="2" vector-effect="non-scaling-stroke"/>
          </svg>
          <div class="note" style="margin-top:8px">부하가 올라가 유지되면 물체를 잡은 상태</div>
        </div>
      </section>
    </main>
    <aside class="col" style="flex:1 1 360px">
      <section class="card stack" id="dev-card"></section>
      <section class="card stack">
        <div class="between"><div class="title">시연 수집</div><div class="sub" id="rec-progress">-</div></div>
        <div class="progress"><div id="rec-bar" style="width:0"></div></div>
        <div class="field"><label for="task">작업 문장</label><input id="task" class="input" type="text"></div>
        <button id="rec-main" class="btn primary lg" type="button" disabled>녹화 시작</button>
        <div class="btn-row">
          <button class="btn flex" type="button" data-finish="success" disabled>성공</button>
          <button class="btn flex" type="button" data-finish="fail" disabled>실패</button>
          <button class="btn flex" type="button" data-finish="discard" title="이번 녹화를 버립니다" disabled>재녹화</button>
        </div>
        <div class="between note"><span id="rec-counts">-</span><span id="rec-avg"></span></div>
        <div class="note" id="rec-dataset"></div>
      </section>
      <section class="card" style="display:flex;flex-direction:column;min-height:300px;flex:1 1 auto">
        <div class="between"><div class="title">명령</div><div class="sub">연결 안 됨</div></div>
        <div style="flex:1 1 auto;display:flex;flex-direction:column;justify-content:center;padding:18px 0">
          <div class="bubble note" style="color:var(--text-2)">자연어·음성 명령은 아직 연결되지 않았습니다. 학습한 정책은 평가 화면에서 실행할 수 있습니다.</div>
        </div>
        <div class="composer">
          <label for="cmd" style="position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)">명령 입력</label>
          <input id="cmd" type="text" placeholder="명령 입력은 준비 중입니다" disabled>
          <button class="icon-btn" type="button" aria-label="음성으로 말하기" disabled><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0"/><path d="M12 18v3"/></svg></button>
          <button class="icon-btn fill" type="button" aria-label="명령 보내기" disabled><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 19V5"/><path d="M6 11l6-6 6 6"/></svg></button>
        </div>
      </section>
    </aside>
  </div>`;

  const toggle = $("#devices-toggle");
  // 장비 구성 펼치기/접기
  const setDevices = (open) => {
    $("#devices").hidden = !open;
    toggle.setAttribute("aria-expanded", String(open));
    toggle.querySelector("svg").style.transform = open ? "rotate(180deg)" : "";
    storeSet("devicesOpen", open ? "1" : "0");
  };
  setDevices(storeGet("devicesOpen", "0") === "1");
  toggle.onclick = () => setDevices($("#devices").hidden);
  onLeave(() => (toggle.onclick = null));

  let joint = storeGet("joint", "shoulder_pan");
  // 그래프 관절 선택
  const setJoint = (j) => {
    joint = j;
    storeSet("joint", j);
    $$("#joint-tabs button").forEach((b) => {
      b.classList.toggle("on", b.dataset.j === j);
      b.setAttribute("aria-selected", String(b.dataset.j === j));
    });
  };
  setJoint(ov.joints.includes(joint) ? joint : ov.joints[0]);
  $("#joint-tabs").onclick = (e) => e.target.dataset.j && setJoint(e.target.dataset.j);

  $("#task").value = ov.task;

  // 장치 카드, 상태가 바뀔 때만 다시 그림
  let devKey = "";
  const renderDev = (s) => {
    const key = `${s.mode}|${s.error}`;
    if (key === devKey) return;
    devKey = key;
    const text = {
      off: ["연결 안 됨", "팔로워, 리더, 카메라를 엽니다."],
      connecting: ["연결 중", "장치를 여는 중입니다."],
      idle: ["연결됨, 대기", "지금 자세를 유지하고 있습니다."],
      teleop: ["텔레옵 중", "리더 암을 움직이면 팔로워가 따라 합니다."],
      policy: ["정책 실행 중", "평가 화면에서 실행한 정책이 로봇을 움직입니다."],
      homing: ["시작 자세로 이동 중", ""],
    }[s.mode];
    const buttons = {
      off: `<button class="btn primary flex" data-dev="connect">장치 연결</button>`,
      connecting: `<button class="btn primary flex" disabled>연결 중</button>`,
      idle: `<button class="btn primary flex" data-dev="teleop-on">텔레옵 시작</button><button class="btn" data-dev="disconnect">연결 해제</button>`,
      teleop: `<button class="btn flex" data-dev="teleop-off">텔레옵 정지</button><button class="btn" data-dev="disconnect">연결 해제</button>`,
      policy: `<a class="btn flex" href="/eval" style="text-decoration:none">평가 화면</a><button class="btn" data-dev="disconnect">연결 해제</button>`,
      homing: `<button class="btn" data-dev="disconnect">연결 해제</button>`,
    }[s.mode];
    $("#dev-card").innerHTML = `
      <div class="between"><div class="title">장치</div><div class="sub">${esc(text[0])}</div></div>
      ${text[1] ? `<div class="note">${esc(text[1])}</div>` : ""}
      ${s.error ? `<div class="banner err">${esc(s.error)}</div>` : ""}
      <div class="btn-row">${buttons}</div>`;
  };
  // 장치 연결/해제, 텔레옵 버튼
  $("#dev-card").onclick = async (e) => {
    const b = e.target.closest("[data-dev]");
    if (!b) return;
    b.disabled = true;
    const what = b.dataset.dev;
    if (what === "connect") await act("/api/rig/connect");
    if (what === "disconnect") await act("/api/rig/disconnect");
    if (what === "teleop-on") await act("/api/teleop", { on: true });
    if (what === "teleop-off") await act("/api/teleop", { on: false });
    devKey = "";
  };

  // 카메라 스트림, 연결 중일 때만 열기
  const setCams = (live) => {
    $$(".media[data-cam]", el).forEach((m) => {
      const img = m.querySelector("img");
      if (live && !img) m.innerHTML = `<img alt="카메라 ${esc(m.dataset.cam)} 실시간 영상" src="/cam/${encodeURIComponent(m.dataset.cam)}?t=${Date.now()}">`;
      if (!live && img) { img.src = ""; m.textContent = "장치 연결 안 됨"; }
    });
  };
  onLeave(() => setCams(false));

  // 녹화 카드 상태 표시
  const renderRec = (s) => {
    const c = s.counts, r = s.recording, connected = !["off", "connecting"].includes(s.mode);
    $("#rec-progress").textContent = `${c.total} / ${c.target}`;
    $("#rec-bar").style.width = `${Math.min(100, (c.total / c.target) * 100)}%`;
    $("#rec-counts").textContent = `성공 ${c.success}, 실패 ${c.fail}`;
    $("#rec-avg").textContent = c.avg_length_s ? `평균 길이 ${c.avg_length_s.toFixed(0)}초` : "";
    $("#rec-dataset").textContent = `데이터셋 ${s.dataset}`;
    const main = $("#rec-main");
    main.classList.toggle("rec", !!r);
    if (r && r.state === "recording") {
      main.textContent = `녹화 중 ${r.elapsed_s.toFixed(1)}초 / ${s.episode_time_s}초`;
      main.disabled = true;
    } else if (r) {
      main.textContent = "시간 종료: 결과를 고르세요";
      main.disabled = true;
    } else {
      main.textContent = connected ? "녹화 시작" : "장치를 연결하면 녹화할 수 있습니다";
      main.disabled = !["idle", "teleop"].includes(s.mode);
    }
    $$("[data-finish]").forEach((b) => (b.disabled = !r));
  };
  // 녹화 시작, 성공/실패/재녹화 버튼
  $("#rec-main").onclick = async (e) => {
    e.target.disabled = true;
    e.target.textContent = "데이터셋 여는 중";
    await act("/api/record/start", { task: $("#task").value });
  };
  $$("[data-finish]").forEach((b) => (b.onclick = async () => {
    $$("[data-finish]").forEach((x) => (x.disabled = true));
    const f = b.dataset.finish;
    const r = await act("/api/record/finish", { result: f === "discard" ? null : f });
    if (r && r.episode != null) toast(`에피소드 ${pad3(r.episode)}을 ${f === "success" ? "성공" : "실패"}으로 저장했습니다`);
  }));

  // 장치 상태 주기 갱신
  let mode = "off";
  poll(async () => {
    const s = await api("/api/rig");
    mode = s.mode;
    renderDev(s);
    renderRec(s);
    setCams(!["off", "connecting"].includes(s.mode));
  }, 300);

  // 관절 추종, 지연, 그리퍼 부하 그래프 갱신
  poll(async () => {
    if (mode === "off") return;
    const t = await api(`/api/telemetry?joint=${joint}`);
    const box = { w: 600, h: 150, x0: -10, x1: 0 };
    const [y0, y1] = extent([t.actual, t.target], joint === "gripper" ? 20 : 10);
    $("#j-target").setAttribute("d", linePath(t.t, t.target, { ...box, y0, y1 }));
    $("#j-actual").setAttribute("d", linePath(t.t, t.actual, { ...box, y0, y1 }));
    const errs = t.actual.map((a, i) => (t.target[i] == null ? null : Math.abs(a - t.target[i]))).filter((x) => x != null);
    const unit = joint === "gripper" ? "%" : "°";
    $("#j-err").textContent = errs.length ? `평균 추종 오차 ${(errs.reduce((a, b) => a + b, 0) / errs.length).toFixed(1)}${unit}` : "목표각 없음 (텔레옵이나 정책이 꺼져 있음)";

    const policy = t.infer_ms.some((x) => x != null);
    const lat = policy ? t.infer_ms : t.loop_ms;
    const limit = 1000 / overview.fps;
    const top = Math.max(limit * 1.6, ...lat.filter((x) => x != null));
    $("#lat-label").textContent = policy ? "추론 지연" : "제어 주기 (한 바퀴 처리 시간)";
    const lastLat = [...lat].reverse().find((x) => x != null);
    $("#lat-value").textContent = lastLat == null ? "-" : `${lastLat.toFixed(0)}ms`;
    $("#lat-line").setAttribute("d", linePath(t.t, lat, { w: 300, h: 80, x0: -10, x1: 0, y0: 0, y1: top }));
    const ly = 80 - (limit / top) * 80;
    $("#lat-limit").setAttribute("y1", ly);
    $("#lat-limit").setAttribute("y2", ly);

    const lastLoad = [...t.load].reverse().find((x) => x != null);
    $("#load-value").textContent = lastLoad == null ? "-" : `${lastLoad.toFixed(0)}%`;
    $("#load-line").setAttribute("d", linePath(t.t, t.load, { w: 300, h: 80, x0: -10, x1: 0, y0: 0, y1: 100 }));
  }, 250);
}

// 데이터 화면: 에피소드 목록, 영상 재생, 라벨 수정
function dataPage(el) {
  el.innerHTML = `<div class="row" style="align-items:flex-start">
    <section class="card stack" style="flex:1 1 480px" id="ds-left"><div class="empty">불러오는 중</div></section>
    <section class="card stack" style="flex:1.3 1 560px" id="ds-right" hidden></section>
  </div>`;
  let repo = new URLSearchParams(location.search).get("repo");
  let detail = null, sel = null;
  const vids = [];
  let raf = 0;
  onLeave(() => { cancelAnimationFrame(raf); vids.forEach((v) => { v.pause(); v.removeAttribute("src"); v.load(); }); });

  // 데이터셋 목록과 선택한 데이터셋 불러오기
  async function load() {
    const list = await api("/api/datasets");
    if (!list.length) {
      $("#ds-left").innerHTML = `<div class="title">데이터</div><div class="empty">녹화된 데이터셋이 없습니다. 조작 화면에서 시연을 녹화하세요.</div>`;
      return;
    }
    if (!repo || !list.some((d) => d.repo === repo)) {
      const rig = await api("/api/rig");
      repo = list.some((d) => d.repo === rig.dataset) ? rig.dataset : list[0].repo;
    }
    try {
      detail = await api(`/api/dataset?repo=${encodeURIComponent(repo)}`);
    } catch (e) {
      $("#ds-left").innerHTML = `<div class="banner err">${esc(e.message)}</div>`;
      return;
    }
    renderLeft(list);
    if (detail.episodes.length) {
      const keep = detail.episodes.find((e) => e.index === sel);
      select(keep ? keep.index : detail.episodes[detail.episodes.length - 1].index, !keep);
    }
  }

  // 데이터셋 요약과 에피소드 표
  function renderLeft(list) {
    const eps = detail.episodes;
    const n = (f) => eps.filter(f).length;
    $("#ds-left").innerHTML = `
      <div class="card-head" style="margin:0;align-items:center">
        <label for="ds-select" class="title" style="position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)">데이터셋</label>
        <select id="ds-select" class="input" style="height:38px;width:auto;max-width:100%;font-weight:600">
          ${list.map((d) => `<option value="${esc(d.repo)}" ${d.repo === repo ? "selected" : ""}>${esc(d.repo)}</option>`).join("")}
        </select>
        <button class="link" id="ds-reload" type="button">새로고침</button>
      </div>
      <div class="row" style="gap:8px">
        ${[["에피소드", detail.total_episodes], ["성공", n((e) => e.result === "success")], ["실패", n((e) => e.result === "fail")], ["삭제 표시", n((e) => e.deleted)]]
          .map(([k, v]) => `<div class="stat-box"><div class="note">${k}</div><div class="v">${v}</div></div>`).join("")}
      </div>
      ${detail.unreadable ? `<div class="banner">녹화 세션이 데이터셋을 쓰는 중이라 최근 에피소드 일부를 아직 읽을 수 없습니다. 조작 화면에서 텔레옵을 정지하면 보입니다.</div>` : ""}
      <div class="table-wrap" style="max-height:560px;overflow-y:auto">
        <table class="tbl plain"><thead><tr><th>번호</th><th>결과</th><th>길이</th><th>물체 위치</th><th>메모</th></tr></thead>
        <tbody>${eps.slice().reverse().map((e) => `
          <tr class="click ${e.index === sel ? "sel" : ""} ${e.deleted ? "dim" : ""}" data-ep="${e.index}" tabindex="0">
            <td>${pad3(e.index)}</td>
            <td>${e.result === "success" ? "성공" : e.result === "fail" ? (e.keep ? "실패, 유지" : "실패") : "-"}</td>
            <td>${sec(e.length_s)}</td><td>${esc(e.position)}</td>
            <td>${[e.deleted ? "삭제 표시" : "", esc(e.memo)].filter(Boolean).join(", ")}</td>
          </tr>`).join("")}</tbody></table>
        ${eps.length ? "" : `<div class="empty">아직 에피소드가 없습니다.</div>`}
      </div>
      <div class="note">학습에는 삭제 표시하지 않은 성공 에피소드와, 데이터에 유지한 실패 에피소드가 쓰입니다.</div>`;
    $("#ds-select").onchange = (e) => { repo = e.target.value; sel = null; history.replaceState(null, "", `/data?repo=${encodeURIComponent(repo)}`); load(); };
    $("#ds-reload").onclick = load;
    $("#ds-left tbody").onclick = (e) => { const tr = e.target.closest("tr[data-ep]"); if (tr) select(+tr.dataset.ep, true); };
    $("#ds-left tbody").onkeydown = (e) => { const tr = e.target.closest("tr[data-ep]"); if (tr && e.key === "Enter") select(+tr.dataset.ep, true); };
  }

  // 라벨 저장 후 다시 불러오기
  async function label(fields) {
    if (await act("/api/label", { repo, ep: sel, ...fields })) await load();
  }

  // 에피소드 선택, 영상/그래프/라벨 표시
  function select(idx, reloadMedia) {
    sel = idx;
    $$("#ds-left tr[data-ep]").forEach((tr) => tr.classList.toggle("sel", +tr.dataset.ep === idx));
    const e = detail.episodes.find((x) => x.index === idx);
    const right = $("#ds-right");
    right.hidden = false;
    if (!reloadMedia && right.dataset.ep === String(idx)) {
      $("#ep-pos").value = e.position;
      $("#ep-memo").value = e.memo;
      return;
    }
    right.dataset.ep = String(idx);
    cancelAnimationFrame(raf);
    vids.length = 0;
    right.innerHTML = `
      <div class="between"><div class="title">에피소드 ${pad3(idx)}</div><div class="sub">${esc(e.task)}</div></div>
      <div class="row" style="gap:14px">${detail.video_keys.map((k) => `
        <div style="flex:1 1 260px;min-width:0"><div class="note" style="margin-bottom:6px">${esc(k.replace("observation.images.", "카메라 "))}</div>
        <div class="media" style="height:250px"><video muted playsinline preload="auto" data-key="${esc(k)}"></video></div></div>`).join("")}</div>
      <div style="display:flex;align-items:center;gap:16px">
        <button class="btn primary" id="ep-play" type="button" aria-label="재생" style="width:44px;padding:0"><svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M7 5v14l12-7z"/></svg></button>
        <input id="ep-seek" type="range" min="0" max="${e.length_s}" step="0.01" value="0" style="flex:1 1 auto;accent-color:var(--blue)" aria-label="재생 위치">
        <div class="note" id="ep-time" style="white-space:nowrap">0.0초 / ${sec(e.length_s)}</div>
      </div>
      <div style="background:var(--field);border-radius:8px;padding:14px 16px">
        <div class="between" style="margin-bottom:8px"><div class="title" style="font-size:13px">그리퍼 위치</div><div class="note">실선은 관측, 점선은 명령. 세로선은 현재 재생 위치</div></div>
        <svg class="chart" viewBox="0 0 600 110" preserveAspectRatio="none" style="height:110px" role="img" aria-label="그리퍼 위치 시계열">
          <line x1="0" y1="55" x2="600" y2="55" stroke="#E5E7EB" vector-effect="non-scaling-stroke"/>
          <path id="sig-action" fill="none" stroke="#9CA3AF" stroke-width="1.5" stroke-dasharray="5 4" vector-effect="non-scaling-stroke"/>
          <path id="sig-state" fill="none" stroke="#374151" stroke-width="1.5" vector-effect="non-scaling-stroke"/>
          <line id="sig-head" x1="0" x2="0" y1="0" y2="110" stroke="#2563EB" stroke-width="2" vector-effect="non-scaling-stroke"/>
        </svg>
      </div>
      <div class="grid2">
        <div class="field"><label for="ep-pos">물체 위치</label><input id="ep-pos" class="input" value="${esc(e.position)}" placeholder="예: 왼쪽 앞"></div>
        <div class="field"><label for="ep-memo">메모</label><input id="ep-memo" class="input" value="${esc(e.memo)}" placeholder="예: 들어 올리다 놓침"></div>
      </div>
      <div class="btn-row">
        <button class="btn primary" data-act="keep" type="button">데이터에 유지</button>
        <button class="btn" data-act="flip" type="button">결과 변경</button>
        <button class="btn" data-act="delete" type="button">삭제 표시</button>
        <button class="btn" data-act="next" type="button">다음 에피소드</button>
      </div>`;

    $("#ep-pos").onchange = (ev) => label({ position: ev.target.value });
    $("#ep-memo").onchange = (ev) => label({ memo: ev.target.value });
    right.querySelector(".btn-row").onclick = (ev) => {
      const a = ev.target.closest("[data-act]")?.dataset.act;
      const cur = detail.episodes.find((x) => x.index === sel);
      if (a === "keep") label({ keep: true, deleted: false });
      if (a === "flip") label({ result: cur.result === "success" ? "fail" : "success" });
      if (a === "delete") label({ deleted: true });
      if (a === "next") {
        const i = detail.episodes.findIndex((x) => x.index === sel);
        const nxt = detail.episodes[i - 1] ?? detail.episodes[detail.episodes.length - 1];
        select(nxt.index, true);
      }
    };

    // v3 데이터셋은 한 영상 파일에 여러 에피소드가 이어 붙은 구조
    $$("video", right).forEach((v) => {
      const info = e.videos[v.dataset.key];
      v.dataset.from = info.from;
      v.src = `/video?repo=${encodeURIComponent(repo)}&key=${encodeURIComponent(v.dataset.key)}&chunk=${info.chunk}&file=${info.file}`;
      v.addEventListener("loadedmetadata", () => (v.currentTime = info.from), { once: true });
      v.addEventListener("error", () => (v.parentElement.textContent = "이 브라우저에서 영상을 열 수 없습니다"), { once: true });
      vids.push(v);
    });
    const seekTo = (t) => vids.forEach((v) => (v.currentTime = +v.dataset.from + t));
    const playBtn = $("#ep-play");
    const setPlayIcon = (playing) => {
      playBtn.setAttribute("aria-label", playing ? "일시 정지" : "재생");
      playBtn.innerHTML = playing
        ? `<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M6 5h4v14H6zM14 5h4v14h-4z"/></svg>`
        : `<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M7 5v14l12-7z"/></svg>`;
    };
    playBtn.onclick = () => {
      if (!vids.length) return;
      if (vids[0].paused) {
        if (vids[0].currentTime - vids[0].dataset.from >= e.length_s - 0.05) seekTo(0);
        vids.forEach((v) => v.play().catch(() => {}));
        setPlayIcon(true);
      } else {
        vids.forEach((v) => v.pause());
        setPlayIcon(false);
      }
    };
    $("#ep-seek").oninput = (ev) => seekTo(+ev.target.value);
    // 재생 위치를 슬라이더와 그래프 세로선에 반영
    const frame = () => {
      if (vids.length) {
        let t = vids[0].currentTime - vids[0].dataset.from;
        if (t >= e.length_s) {
          vids.forEach((v) => v.pause());
          setPlayIcon(false);
          t = e.length_s;
        }
        t = Math.max(0, t);
        if (document.activeElement !== $("#ep-seek")) $("#ep-seek").value = t;
        $("#ep-time").textContent = `${t.toFixed(1)}초 / ${sec(e.length_s)}`;
        const x = (t / e.length_s) * 600;
        $("#sig-head").setAttribute("x1", x);
        $("#sig-head").setAttribute("x2", x);
      }
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);

    // 그리퍼 그래프
    api(`/api/signal?repo=${encodeURIComponent(repo)}&ep=${idx}&chunk=${e.data.chunk}&file=${e.data.file}`).then((s) => {
      if (sel !== idx) return;
      const box = { w: 600, h: 110, x0: 0, x1: e.length_s, y0: -5, y1: 105 };
      $("#sig-state").setAttribute("d", linePath(s.t, s.state, box));
      $("#sig-action").setAttribute("d", linePath(s.t, s.action, box));
    }).catch((err) => toast(err.message));
  }

  load();
}

// 학습 화면: 설정, 진행, 손실 곡선, 체크포인트
function trainPage(el) {
  const ov = overview;
  const LR = { act: "0.00001", smolvla: "0.0001", diffusion: "0.0001" };
  el.innerHTML = `<div class="row" style="align-items:flex-start">
    <div class="col" style="flex:1 1 320px">
      <section class="card stack">
        <div class="title">학습 설정</div>
        <div class="field"><label for="t-policy">기반 정책</label>
          <select id="t-policy" class="input">${Object.entries(POLICY_KO).map(([k, v]) => `<option value="${k}" ${k === ov.policy ? "selected" : ""}>${v}${k === "smolvla" ? " (smolvla_base에서 시작)" : ""}</option>`).join("")}</select></div>
        <div class="field"><label for="t-repo">데이터</label><select id="t-repo" class="input"></select>
          <div class="note" id="t-repo-note"></div></div>
        <div class="grid2">
          <div class="field"><label for="t-steps">스텝</label><input id="t-steps" class="input" type="number" min="1" value="${ov.train_defaults.steps}"></div>
          <div class="field"><label for="t-batch">배치 크기</label><input id="t-batch" class="input" type="number" min="1" value="${ov.train_defaults.batch}"></div>
          <div class="field"><label for="t-lr">학습률</label><input id="t-lr" class="input" type="text" value="${LR[ov.policy] ?? "0.0001"}"></div>
          <div class="field"><label for="t-save">저장 주기</label><input id="t-save" class="input" type="number" min="1" value="5000"></div>
        </div>
        <div class="btn-row"><button id="t-start" class="btn primary lg" style="flex:2 1 0" type="button">학습 시작</button>
          <button id="t-stop" class="btn lg" style="flex:1 1 0" type="button">중단</button></div>
      </section>
      <section class="card"><div class="title" style="margin-bottom:6px">저장된 모델</div><div id="t-ckpts" class="list"></div></section>
    </div>
    <div class="col" style="flex:999 1 600px">
      <section class="card stack">
        <div class="row" style="gap:12px 48px" id="t-summary"></div>
        <div class="progress" style="height:6px"><div id="t-bar" style="width:0"></div></div>
      </section>
      <section class="card">
        <div class="card-head"><div class="title">손실 곡선</div><div class="note">세로축은 로그 눈금</div></div>
        <svg class="chart" viewBox="0 0 600 220" preserveAspectRatio="none" style="height:240px" role="img" aria-label="학습 손실 곡선">
          ${[55, 110, 165].map((y) => `<line x1="0" y1="${y}" x2="600" y2="${y}" stroke="#EEF0F3" vector-effect="non-scaling-stroke"/>`).join("")}
          <line id="t-now" x1="0" x2="0" y1="0" y2="220" stroke="#9CA3AF" stroke-dasharray="3 4" vector-effect="non-scaling-stroke"/>
          <path id="t-loss" fill="none" stroke="#2563EB" stroke-width="2" vector-effect="non-scaling-stroke"/>
        </svg>
        <div class="between note" style="margin-top:8px"><span>0</span><span id="t-mid"></span><span id="t-end"></span></div>
      </section>
      <div class="banner err" id="t-hint" hidden></div>
      <section class="card"><div class="title" style="margin-bottom:8px">기록</div><div id="t-recent"></div>
        <details id="t-log-wrap" style="margin-top:12px"><summary class="note" style="cursor:pointer">로그</summary><pre class="log" id="t-log"></pre></details>
      </section>
    </div></div>`;

  $("#t-policy").onchange = (e) => ($("#t-lr").value = LR[e.target.value]);
  let datasets = [];
  const repoNote = () => {
    const d = datasets.find((x) => x.repo === $("#t-repo").value);
    $("#t-repo-note").textContent = d ? `에피소드 ${d.episodes}개, 학습 제외 ${d.excluded}개 (삭제 표시, 유지하지 않은 실패)` : "녹화된 데이터셋이 없습니다";
  };
  $("#t-repo").onchange = repoNote;
  // 학습 시작/중단
  $("#t-start").onclick = async (e) => {
    e.target.disabled = true;
    const ok = await act("/api/train/start", {
      policy: $("#t-policy").value, repo: $("#t-repo").value, steps: $("#t-steps").value,
      batch: $("#t-batch").value, lr: $("#t-lr").value, save_freq: $("#t-save").value,
    });
    if (ok) toast("학습을 시작했습니다. 첫 기록까지 시간이 걸릴 수 있습니다.");
  };
  $("#t-stop").onclick = () => act("/api/train/stop");

  // 학습 상태 주기 갱신, 선택 목록은 처음 한 번만 채움
  let first = true;
  poll(async () => {
    const s = await api("/api/train");
    if (first) {
      datasets = s.datasets;
      $("#t-repo").innerHTML = datasets.map((d) => `<option value="${esc(d.repo)}">${esc(d.repo)}</option>`).join("");
      repoNote();
      first = false;
    }
    const running = s.state === "running";
    $("#t-start").disabled = running || !datasets.length;
    $("#t-stop").disabled = !running;
    $$("#t-policy,#t-repo,#t-steps,#t-batch,#t-lr,#t-save").forEach((i) => (i.disabled = running));

    const total = s.total || (s.loss.length ? s.loss[s.loss.length - 1][0] : 0);
    const last = s.recent[0];
    const stateKo = { idle: "대기", running: "학습 중", done: "완료", failed: "실패", stopped: "중단됨" }[s.state];
    const rem = s.remaining_s == null ? "-" : s.remaining_s >= 3600 ? `${Math.floor(s.remaining_s / 3600)}시간 ${Math.round((s.remaining_s % 3600) / 60)}분` : `${Math.max(1, Math.round(s.remaining_s / 60))}분`;
    const mem = s.gpu.name ? `${last?.mem_gb != null ? last.mem_gb.toFixed(1) : "-"} / ${s.gpu.total_gb.toFixed(0)}GB` : "CPU";
    $("#t-summary").innerHTML = [
      [`진행 (${stateKo})`, `${num(s.step)} / ${num(total)}`], ["현재 손실", last ? last.loss.toFixed(3) : "-"],
      ["남은 시간", rem], [s.gpu.name ? "그래픽 메모리" : "연산 장치", mem],
    ].map(([k, v]) => `<div><div class="stat-label">${k}</div><div class="stat-value" style="margin:2px 0 0">${v}</div></div>`).join("");
    $("#t-bar").style.width = total ? `${Math.min(100, (s.step / total) * 100)}%` : "0";

    if (s.loss.length) {
      const xs = s.loss.map((r) => r[0]), ys = s.loss.map((r) => Math.log10(Math.max(r[1], 1e-6)));
      const [y0, y1] = extent([ys], 0.3);
      $("#t-loss").setAttribute("d", linePath(xs, ys, { w: 600, h: 220, x0: 0, x1: total || 1, y0, y1 }));
      const x = ((s.step || 0) / (total || 1)) * 600;
      $("#t-now").setAttribute("x1", x);
      $("#t-now").setAttribute("x2", x);
    } else {
      $("#t-loss").setAttribute("d", "");
    }
    $("#t-mid").textContent = total ? num(Math.round(total / 2)) : "";
    $("#t-end").textContent = total ? `${num(total)}스텝` : "";

    $("#t-recent").innerHTML = s.recent.length
      ? `<table class="tbl plain"><thead><tr><th>스텝</th><th>손실</th><th>학습률</th><th>초당 반복</th></tr></thead><tbody>
          ${s.recent.map((r) => `<tr><td style="font-weight:400">${num(r.step)}</td><td>${r.loss.toFixed(3)}</td><td>${r.lr != null ? r.lr.toExponential(1) : "-"}</td><td>${r.step_s ? (1 / r.step_s).toFixed(1) : "-"}</td></tr>`).join("")}</tbody></table>`
      : `<div class="empty">아직 기록이 없습니다.</div>`;
    $("#t-log").textContent = s.tail.join("\n");
    $("#t-hint").hidden = !s.hint;
    $("#t-hint").textContent = s.hint ?? "";
    if (s.state === "failed" && !$("#t-log-wrap").dataset.opened) { $("#t-log-wrap").open = true; $("#t-log-wrap").dataset.opened = "1"; }

    $("#t-ckpts").innerHTML = s.checkpoints.length
      ? s.checkpoints.slice(0, 8).map((c) => `<div style="display:flex;align-items:center;gap:12px">
          <div style="flex:1 1 auto;min-width:0"><div style="font-weight:600">${num(c.step)}스텝</div><div class="note" style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${esc(c.run)}">${esc(c.run)}</div></div>
          <div class="note">손실 ${c.loss != null ? c.loss.toFixed(3) : "-"}</div>
          <a href="/eval?checkpoint=${encodeURIComponent(c.path)}" style="font-weight:600;text-decoration:none">평가하기</a></div>`).join("")
      : `<div class="note" style="padding:8px 0">저장된 체크포인트가 없습니다.</div>`;
  }, 1500);
}

// 평가 화면: 모델 준비, 회차 진행, 모델별 성공률
function evalPage(el) {
  const wanted = new URLSearchParams(location.search).get("checkpoint");
  const cam = overview.cameras.find((c) => /front|top|work/i.test(c)) ?? overview.cameras[0];
  el.innerHTML = `<div class="row" style="align-items:flex-start">
    <section class="card stack" style="flex:999 1 640px" id="e-main"></section>
    <section class="card" style="flex:1 1 360px" id="e-hist"></section></div>`;
  let key = "";

  // 평가 준비 폼
  const setupHtml = (s) => {
    const opts = s.checkpoints.map((c) => `<option value="${esc(c.path)}" data-label="${esc(`${c.run.split("_")[0].toUpperCase()} ${num(c.step)}스텝`)}" ${c.path === wanted ? "selected" : ""}>${esc(c.run)} · ${num(c.step)}스텝</option>`);
    if (s.mock) opts.push(`<option value="mock" data-label="모의 정책">모의 정책 (화면 확인용)</option>`);
    const rigOk = ["idle", "teleop"].includes(s.rig);
    return `
      <div class="title">평가 준비</div>
      ${rigOk ? "" : `<div class="banner">정책이 로봇을 움직입니다. 먼저 <a href="/">조작 화면</a>에서 장치를 연결하고 비상 정지를 준비하세요.</div>`}
      ${s.error ? `<div class="banner err">${esc(s.error)}</div>` : ""}
      <div class="field"><label for="e-ckpt">모델</label>
        ${opts.length ? `<select id="e-ckpt" class="input">${opts.join("")}</select>` : `<div class="note">저장된 체크포인트가 없습니다. 학습 화면에서 먼저 학습하세요.</div>`}</div>
      <div class="field"><label for="e-task">작업 문장</label><input id="e-task" class="input" value="${esc(overview.task)}"></div>
      <div class="note">물체 위치 ${s.grid[0] * s.grid[1]}곳에서 한 번씩 실행합니다. 회차마다 최대 ${overview.episode_time_s}초 동안 정책을 실행하고, 끝나면 시작 자세로 돌아갑니다.</div>
      <div class="btn-row"><button class="btn primary lg flex" id="e-prepare" ${opts.length && rigOk && !s.loading ? "" : "disabled"}>${s.loading ? "모델 읽는 중" : "평가 준비"}</button></div>`;
  };

  // 진행 중인 평가
  const runHtml = (s) => `
    <div class="between"><div class="title">진행 중인 평가</div><div class="sub">${esc(s.session.label)}</div></div>
    <div class="row" style="gap:12px 56px" id="e-stats"></div>
    <div class="row" style="gap:20px">
      <div style="flex:1 1 320px;min-width:0"><div class="media" style="height:310px" id="e-cam">장치 연결 안 됨</div></div>
      <div style="flex:1 1 360px;min-width:0">
        <div class="between" style="margin-bottom:10px"><div class="title" style="font-size:13px">물체 위치별 결과</div><div class="note">○ 성공 × 실패</div></div>
        <div class="cells" id="e-cells" style="grid-template-columns:repeat(${s.grid[0]},1fr)"></div>
        <div class="note" style="margin-top:10px">파란 테두리 칸이 이번 회차에 물체를 놓을 위치</div>
      </div>
    </div>
    <div class="note" id="e-run"></div>
    <div style="display:flex;flex-wrap:wrap;gap:8px;align-items:flex-end">
      <button class="btn primary lg" id="e-trial" type="button">회차 시작</button>
      <button class="btn lg" data-res="1" type="button">성공</button>
      <button class="btn lg" data-res="0" type="button">실패</button>
      <div class="field" style="flex:1 1 240px"><label for="e-reason">실패 원인</label><input id="e-reason" class="input" placeholder="집기 위치 어긋남, 들어 올리다 놓침 등"></div>
    </div>
    <div><button class="link" id="e-end" type="button">평가 끝내기</button></div>`;

  // 작업 공간 카메라 스트림
  const camLive = (live) => {
    const m = $("#e-cam");
    if (!m) return;
    const img = m.querySelector("img");
    if (live && !img) m.innerHTML = `<img alt="카메라 ${esc(cam)} 실시간 영상" src="/cam/${encodeURIComponent(cam)}?t=${Date.now()}">`;
    if (!live && img) { img.src = ""; m.textContent = "장치 연결 안 됨"; }
  };
  onLeave(() => camLive(false));

  // 평가 버튼 동작
  $("#e-main").onclick = async (e) => {
    const t = e.target.closest("button");
    if (!t) return;
    if (t.id === "e-prepare") {
      const o = $("#e-ckpt").selectedOptions[0];
      t.disabled = true;
      await act("/api/eval/prepare", { checkpoint: o.value, label: o.dataset.label, task: $("#e-task").value });
      key = "";
    }
    if (t.id === "e-trial") { t.disabled = true; await act("/api/eval/trial/start"); }
    if (t.dataset.res) {
      t.disabled = true;
      if (await act("/api/eval/trial/finish", { success: t.dataset.res === "1", reason: t.dataset.res === "1" ? "" : $("#e-reason").value })) $("#e-reason").value = "";
    }
    if (t.id === "e-end") { await act("/api/eval/end"); key = ""; }
  };

  // 평가 상태 주기 갱신, 구성이 바뀔 때만 다시 그림
  poll(async () => {
    const [s, rig] = await Promise.all([api("/api/eval"), api("/api/rig")]);
    s.rig = rig.mode;
    const k = s.session ? `run|${s.session.id}` : `setup|${s.loading}|${s.error}|${s.rig}|${s.checkpoints.length}`;
    if (k !== key) {
      key = k;
      $("#e-main").innerHTML = s.session ? runHtml(s) : setupHtml(s);
    }
    if (s.session) {
      const tr = s.session.trials, N = s.grid[0] * s.grid[1];
      const ok = tr.filter((x) => x.success).length;
      const avg = tr.length ? tr.reduce((a, x) => a + x.duration_s, 0) / tr.length : null;
      $("#e-stats").innerHTML = [["회차", `${Math.min(tr.length + (s.trial_running ? 1 : 0), N)} / ${N}`], ["현재 성공", `${ok} / ${tr.length}`], ["평균 소요", avg == null ? "-" : `${avg.toFixed(1)}초`]]
        .map(([a, b]) => `<div><div class="stat-label">${a}</div><div class="stat-value" style="margin:2px 0 0">${b}</div></div>`).join("");
      $("#e-cells").innerHTML = Array.from({ length: N }, (_, i) => {
        const t = tr[i];
        if (t) return `<div class="cell ${t.success ? "ok" : "fail"}" title="${t.success ? "성공" : "실패: " + esc(t.reason)}">${t.success ? "○" : "×"}</div>`;
        return `<div class="cell ${i === tr.length ? "cur" : ""}"></div>`;
      }).join("");
      const p = rig.policy;
      $("#e-run").textContent = s.trial_running && p
        ? (p.timed_out ? `시간 종료 (${overview.episode_time_s}초). 결과를 고르세요.` : p.running ? `정책 실행 중: ${num(p.steps)}스텝, ${p.elapsed_s.toFixed(1)}초 경과` : "결과를 고르세요")
        : tr.length >= N ? "모든 위치를 평가했습니다." : "물체를 파란 테두리 칸에 놓고 회차를 시작하세요.";
      $("#e-trial").disabled = s.trial_running || tr.length >= N || !["idle", "teleop"].includes(rig.mode);
      $$("[data-res]").forEach((b) => (b.disabled = !s.trial_running));
      camLive(!["off", "connecting"].includes(rig.mode));
    }

    const hist = s.history.filter((h) => h.trials.length || (s.session && h.id === s.session.id)).reverse();
    $("#e-hist").innerHTML = `
      <div class="title">모델별 성공률</div>
      <div class="note" style="margin:2px 0 10px">같은 물체 위치 ${s.grid[0] * s.grid[1]}곳에서 한 번씩 측정</div>
      <div class="list">${hist.length ? hist.map((h) => {
        const ok = h.trials.filter((x) => x.success).length, n = h.trials.length;
        const cur = s.session && h.id === s.session.id;
        return `<div><div class="between"><div style="font-weight:600">${esc(h.label)}</div><div style="font-weight:600">${ok} / ${n}</div></div>
          <div class="note">${esc(h.id.replace(/(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2}).*/, "$1-$2-$3 $4:$5"))}${cur ? ", 평가 진행 중" : ""}</div>
          <div class="bar"><div style="width:${n ? (ok / n) * 100 : 0}%"></div></div></div>`;
      }).join("") : `<div class="note" style="padding:8px 0">아직 평가 기록이 없습니다.</div>`}</div>`;
  }, 500);
}

// 가동 범위 기록 때 관절별로 움직이는 방법
const JOINT_HOW = {
  shoulder_pan: "받침에서 팔 전체를 왼쪽 끝, 오른쪽 끝까지 돌리기",
  shoulder_lift: "어깨에서 팔을 앞으로 숙였다가 뒤로 끝까지 젖히기",
  elbow_flex: "팔꿈치를 완전히 접었다가 끝까지 펴기",
  wrist_flex: "손목을 위로 끝까지, 아래로 끝까지 꺾기",
  wrist_roll: "기록하지 않으므로 움직이지 않아도 됨",
  gripper: "집게를 완전히 닫았다가 끝까지 벌리기",
};

// 옆에서 본 SO-101 팔 그림, 원 안 숫자는 서보 번호
const ARM_SVG = (() => {
  const pts = [[60, 112], [60, 90], [130, 38], [200, 62], [226, 74], [252, 86]];
  const link = (a, b) => `<line x1="${pts[a][0]}" y1="${pts[a][1]}" x2="${pts[b][0]}" y2="${pts[b][1]}"/>`;
  const dot = ([x, y], i) => `<circle cx="${x}" cy="${y}" r="9" fill="var(--blue)"/><text x="${x}" y="${y + 3.5}" text-anchor="middle" font-size="10" font-weight="700" fill="var(--card)">${i + 1}</text>`;
  return `<svg viewBox="0 0 300 140" width="260" height="121" role="img" aria-label="SO-101 팔의 관절 위치와 서보 번호">
    <rect x="18" y="124" width="84" height="10" rx="3" fill="var(--media)"/>
    <rect x="40" y="98" width="40" height="28" rx="6" fill="var(--media)" stroke="var(--gray-line)"/>
    <g stroke="var(--gray-line)" stroke-width="8" stroke-linecap="round">${link(1, 2)}${link(2, 3)}${link(3, 4)}${link(4, 5)}
      <line x1="252" y1="86" x2="286" y2="80"/><line x1="252" y1="86" x2="278" y2="108"/></g>
    ${pts.map(dot).join("")}</svg>`;
})();

// 캘리브레이션 화면: 단계 진행과 관절별 가동 범위
function calibPage(el) {
  let arm = storeGet("calibArm", "follower");
  el.innerHTML = `<div class="row" style="align-items:flex-start">
    <section class="card stack" style="flex:1 1 340px" id="c-left"></section>
    <section class="card" style="flex:999 1 560px">
      <div class="card-head"><div class="title">관절별 가동 범위</div><div class="note">파란 막대는 기록된 범위, 세로선은 현재 위치</div></div>
      <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:12px">${ARM_SVG}
        <div class="note">옆에서 본 팔입니다. 원 안 숫자는 표의 서보 번호와 같습니다.</div></div>
      <div class="table-wrap"><table class="tbl" style="min-width:620px"><thead><tr><th>관절</th><th>서보 번호</th><th>최소</th><th style="width:34%">범위</th><th>최대</th><th>상태</th></tr></thead>
      <tbody id="c-rows"></tbody></table></div>
      <div class="note" style="margin-top:14px">범위가 좁게 기록된 관절은 끝까지 움직이지 않은 것입니다. 저장 전에 다시 움직여 주세요. 손목 회전은 한 바퀴를 모두 쓰므로 기록하지 않습니다.</div>
    </section></div>`;
  let key = "";
  // 각도 표시와 -180~180°를 막대 위치(%)로 변환
  const deg = (v) => (v == null ? "-" : `${Math.round(v)}°`);
  const pct = (v) => ((Math.max(-180, Math.min(180, v)) + 180) / 360) * 100;

  // 팔 선택과 단계 버튼
  $("#c-left").onclick = async (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    if (b.dataset.arm) { arm = b.dataset.arm; storeSet("calibArm", arm); key = ""; return; }
    b.disabled = true;
    const a = b.dataset.c;
    if (a === "start") await act("/api/calib/start", { arm });
    if (a === "middle") await act("/api/calib/middle");
    if (a === "finish") { if (await act("/api/calib/finish")) toast("캘리브레이션을 저장했습니다"); }
    if (a === "cancel") await act("/api/calib/cancel");
    key = "";
  };
  // 진행 중에 화면을 떠나면 취소
  onLeave(() => { if (["1", "2"].includes(key.split("|")[1])) api("/api/calib/cancel", {}).catch(() => {}); });

  // 캘리브레이션 상태 주기 갱신
  poll(async () => {
    const s = await api(`/api/calib?arm=${arm}`);
    const busy = s.step === 1 || s.step === 2;
    if (busy && s.arm && s.arm !== arm) arm = s.arm;
    const step = s.arm === arm ? s.step : 0;
    const k = `${arm}|${step}|${s.rig}|${s.error}|${s.info.last}`;
    if (k !== key) {
      key = k;
      const stepState = (n) => (step > n || step === 3 ? "done" : step === n ? "cur" : "");
      const tag = (n) => (step > n || step === 3 ? `<div class="note" style="margin-top:2px">완료</div>` : step === n ? `<div class="note" style="margin-top:2px;font-weight:600;color:var(--blue)">진행 중</div>` : "");
      const rigBusy = s.rig !== "off";
      const btns = {
        0: `<button class="btn primary lg" style="flex:2 1 0" data-c="start" ${rigBusy ? "disabled" : ""}>캘리브레이션 시작</button>`,
        1: `<button class="btn primary lg" style="flex:2 1 0" data-c="middle">중간 자세 확인</button><button class="btn lg flex" data-c="cancel">취소</button>`,
        2: `<button class="btn primary lg" style="flex:2 1 0" data-c="finish">범위 기록 완료</button><button class="btn lg flex" data-c="start">다시 시작</button>`,
        3: `<button class="btn lg flex" data-c="start" ${rigBusy ? "disabled" : ""}>다시 시작</button>`,
      }[step];
      $("#c-left").innerHTML = `
        <div class="title">캘리브레이션</div>
        <div class="seg wide" role="tablist">${[["follower", "팔로워 암"], ["leader", "리더 암"]].map(([v, t]) => `<button type="button" role="tab" data-arm="${v}" class="${v === arm ? "on" : ""}" aria-selected="${v === arm}" ${busy && v !== arm ? "disabled" : ""}>${t}</button>`).join("")}</div>
        <div class="kv"><div><span>포트</span><span class="mono">${esc(s.info.port)}</span></div>
          <div><span>설정 이름</span><span style="font-weight:600">${esc(s.info.id)}</span></div>
          <div><span>마지막 실행</span><span style="font-weight:600">${esc(s.info.last ?? "없음")}</span></div></div>
        ${rigBusy && !busy ? `<div class="banner">조작 화면에서 장치 연결을 해제해야 캘리브레이션할 수 있습니다.</div>` : ""}
        ${s.mock ? `<div class="banner">모의 장치 모드에서는 모터와 캘리브레이션 파일에 쓰지 않습니다.</div>` : ""}
        ${s.error ? `<div class="banner err">${esc(s.error)}</div>` : ""}
        <ol class="steps">
          <li class="${stepState(1)}"><div class="num">1</div><div><div style="font-weight:600">중간 자세 맞추기</div><div class="sub" style="color:var(--text-2)">토크가 풀립니다. 모든 관절을 가동 범위의 가운데에 두고 확인을 누릅니다.</div>${tag(1)}</div></li>
          <li class="${stepState(2)}"><div class="num">2</div><div><div style="font-weight:600">가동 범위 기록</div><div class="sub" style="color:var(--text-2)">토크가 풀린 팔을 손으로 잡고 각 관절을 하나씩 양쪽 끝까지 천천히 움직입니다. 억지로 밀지 말고 막히는 곳까지만 움직입니다. 관절별 방법은 오른쪽 표에 있고 최소와 최대가 기록됩니다.</div>${tag(2)}${step === 2 ? `<div id="c-prog" style="margin-top:10px"></div>` : ""}</div></li>
          <li class="${stepState(3)}"><div class="num">3</div><div><div style="font-weight:600">저장과 검증</div><div class="sub" style="color:var(--text-2)">설정을 저장한 뒤 조작 화면의 텔레옵으로 리더 암 동작을 따라 하는지 확인합니다.</div>${step === 3 ? `<div class="note" style="margin-top:2px">저장됨</div>` : ""}</div></li>
        </ol>
        <div class="btn-row">${btns}</div>`;
    }
    // 2단계 관절별 기록 현황
    const prog = $("#c-prog");
    if (prog) {
      const rows = s.joints.filter((j) => j.state !== "전체 회전");
      const done = rows.filter((j) => j.state === "기록됨").length;
      prog.innerHTML = `<div style="font-weight:600">기록된 관절 <span style="color:var(--blue)">${done}</span> / ${rows.length}</div>
        <div class="jchips">${rows.map((j) => `<span class="jchip ${j.state === "기록됨" ? "ok" : j.state === "범위 좁음" ? "narrow" : ""}">${j.state === "기록됨" ? "✓ " : ""}${JOINT_KO[j.name]}</span>`).join("")}</div>`;
    }
    $("#c-rows").innerHTML = s.joints.map((j) => {
      const rec = s.arm === arm && j.min != null;
      const st = s.arm === arm ? j.state : "대기";
      return `<tr class="${st === "기록됨" ? "done" : ""}"><td style="padding-top:17px;padding-bottom:17px">${JOINT_KO[j.name]}<div class="note" style="font-weight:400;margin-top:2px">${JOINT_HOW[j.name]}</div></td><td>${j.id}</td><td>${rec ? deg(j.min) : "-"}</td>
        <td><div class="range"><div class="track"></div>
          ${rec ? `<div class="fill" style="left:${pct(j.min)}%;width:${pct(j.max) - pct(j.min)}%"></div>` : ""}
          ${s.arm === arm && j.pos != null ? `<div class="now" style="left:${pct(j.pos)}%"></div>` : ""}</div></td>
        <td>${rec ? deg(j.max) : "-"}</td>
        <td style="${st === "기록됨" ? "color:var(--blue);font-weight:600" : st === "범위 좁음" ? "color:var(--text);font-weight:600" : ""}">${st}</td></tr>`;
    }).join("");
  }, 200);
}

// 시뮬레이션 화면 (아직 연결 안 됨)
function simPage(el) {
  el.innerHTML = `<div class="banner">시뮬레이션은 아직 연결되지 않았습니다. Isaac Sim(Omniverse) 서버가 준비되면 이 화면에서 실물과 시뮬레이션을 비교합니다.</div>
  <div class="row" style="align-items:flex-start">
    <div class="col" style="flex:999 1 640px">
      <section class="card">
        <div class="card-head"><div class="left"><div class="title">실물과 시뮬레이션 비교</div><div class="sub">같은 시연을 양쪽에서 재생</div></div></div>
        <div class="row" style="gap:14px">
          <div style="flex:1 1 280px"><div class="title" style="font-size:13px;margin-bottom:8px">실물</div><div class="media">[작업 공간 카메라 영상]</div></div>
          <div style="flex:1 1 280px"><div class="title" style="font-size:13px;margin-bottom:8px">Isaac Sim</div><div class="media">연결 안 됨</div></div>
        </div>
      </section>
    </div>
    <aside class="col" style="flex:1 1 360px">
      <section class="card stack">
        <div class="title">시뮬레이션 연결</div>
        <div class="field"><label for="s-srv">Omniverse 서버</label><input id="s-srv" class="input" placeholder="서버 주소" disabled></div>
        <div class="field"><label for="s-env">환경</label><input id="s-env" class="input" placeholder="디지털 트윈 환경" disabled></div>
        <div class="btn-row"><button class="btn primary flex" disabled>연결</button><button class="btn flex" disabled>시연 재생</button><button class="btn flex" disabled>정책 실행</button></div>
      </section>
      <section class="card stack">
        <div class="title">합성 데이터 생성</div>
        <div class="note">시뮬레이션이 연결되면 물체 위치, 조명, 카메라 위치를 바꿔 가며 에피소드를 만들어 데이터 화면에 합성으로 구분해 넣습니다.</div>
      </section>
    </aside>
  </div>`;
}

// 공통 설정을 받은 뒤 첫 화면 그리기
(async () => {
  try {
    overview = await api("/api/overview");
  } catch (e) {
    $("#page").innerHTML = `<div class="banner err">서버에 연결할 수 없습니다: ${esc(e.message)}</div>`;
    return;
  }
  $("#mock-badge").hidden = !overview.mock;
  route();
})();
