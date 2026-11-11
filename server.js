/**
 * 听喵说 · 本地服务
 * 架构：单文件网页 + Node 本地服务（无第三方依赖），音视频全部在本机处理，不上传任何云端。
 *
 * 流程：
 *   音频 → 暂存临时文件 → predict.py 听声音 → 返回
 *   视频 → 暂存临时文件 → 并行跑 predict.py（抽音轨听声音）+ vision.py（抽帧看画面）→ 合并 → 返回
 *
 * 降级原则（与界面的"诚实边界"一致）：
 *   视频没有音轨 / 音频读不出来 → 只要画面里看到猫，仍然返回画面结论，status = vision_only
 *   画面读不出来（损坏、罕见编码）→ 只返回声音结论
 *   两边都不行 → 500 + 明确原因
 *
 * 端口：8790（沿用户既有工作台约定顺延 8787→8788→8789→8790）
 */
const http = require('http');
const fs = require('fs');
const path = require('path');
const os = require('os');
const { spawn } = require('child_process');

const PORT = 8790;
// Python 解释器：环境变量 > 本机托管解释器 > PATH 里的 python（不写死任何绝对路径）
const PYTHON = [
  process.env.TMS_PYTHON,
  path.join(os.homedir(), '.workbuddy', 'binaries', 'python', 'envs', 'default', 'Scripts', 'python.exe'),
].find((p) => p && fs.existsSync(p)) || 'python';
const PREDICT = path.join(__dirname, 'scripts', 'predict.py');
const VISION = path.join(__dirname, 'scripts', 'vision.py');
const WEB_DIR = path.join(__dirname, 'web');

const MAX_UPLOAD = 300 * 1024 * 1024;      // 单文件上限 300MB，防止内存被撑爆
const PY_TIMEOUT = 120 * 1000;             // 单个脚本最长 120 秒

// 浏览器录音产出的 audio/webm 属于音频，靠 X-Media-Kind 区分；这些容器一律按视频处理
const AUDIO_EXT = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.opus', '.amr', '.wma', '.webm'];
const VIDEO_EXT = ['.mp4', '.mov', '.m4v', '.avi', '.mkv', '.webm', '.wmv', '.flv', '.3gp', '.mpg', '.mpeg', '.ts', '.mts'];
const ALLOW_EXT = new Set(AUDIO_EXT.concat(VIDEO_EXT));
const FORCE_VISION = new Set(['.mp4', '.mov', '.m4v', '.avi', '.mkv', '.wmv', '.flv', '.3gp', '.mpg', '.mpeg', '.ts', '.mts']);

// ══ 喵语存档（纯本地、不上云） ══
// 每次成功分析的喵语都落到 data/uploads/：原始文件 + index.jsonl（含分析结果），
// 方便日后回看，也为训练数据回流打地基。
const UPLOAD_DIR = path.join(__dirname, 'data', 'uploads');
fs.mkdirSync(UPLOAD_DIR, { recursive: true });
function _safeName(s) {
  return (String(s == null ? '' : s).replace(/[^\w.\-一-鿿]+/g, '_').slice(0, 40)) || 'audio';
}
function _stamp() {
  const d = new Date(), p = (n) => String(n).padStart(2, '0');
  return d.getFullYear() + p(d.getMonth() + 1) + p(d.getDate()) + '_' +
    p(d.getHours()) + p(d.getMinutes()) + p(d.getSeconds());
}
function persistMeow(rawName, ext, buf, result) {
  const id = _stamp() + '_' + Math.random().toString(36).slice(2, 7);
  const fname = 'call_' + id + ext;
  const fpath = path.join(UPLOAD_DIR, fname);
  fs.writeFileSync(fpath, buf);
  const rec = {
    id, ts: Date.now(),
    file: fname,
    original: rawName,
    media: (result && result.media && result.media.kind) || 'audio',
    bytes: buf.length,
    status: result ? result.status : 'pending',
    emotion: (result && result.emotion) || null,
    translation: (result && result.translation) || null,
    confidence: (result && result.confidence) || null,
    uncertain: !!(result && result.uncertain),
    uncertain_reason: (result && result.uncertain_reason) || null,
    p_meow: (result && result.gate && result.gate.p_meow != null) ? result.gate.p_meow : null,
  };
  fs.appendFileSync(path.join(UPLOAD_DIR, 'index.jsonl'), JSON.stringify(rec) + '\n');
  return 'data/uploads/' + fname;
}

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.png': 'image/png', '.jpg': 'image/jpeg', '.svg': 'image/svg+xml',
};

function sendJSON(res, code, obj) {
  res.writeHead(code, { 'Content-Type': 'application/json; charset=utf-8' });
  res.end(JSON.stringify(obj));
}

// 从 Python 报错里提取关键行，去掉 ffmpeg 版本号等噪音
function cleanErr(s) {
  const lines = String(s || '').split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  const key = lines
    .filter((l) => /Error|无法|失败|not recognised|RuntimeError|ModuleNotFound|超时/i.test(l))
    .slice(-2).join(' | ');
  return (key || lines.slice(-1)[0] || String(s || '')).slice(0, 240);
}

// Python 脚本把结果打在一行 JSON 上，取最后一行（前面可能有 warnings）
function lastJSON(out) {
  const lines = String(out || '').trim().split(/\r?\n/).filter(Boolean);
  for (let i = lines.length - 1; i >= 0; i--) {
    try { return JSON.parse(lines[i]); } catch (e) { /* 继续往前找 */ }
  }
  return null;
}

// 跑一个 Python 脚本，永远 resolve（不 reject），带超时保护
function runPy(script, file) {
  return new Promise((resolve) => {
    let child;
    try {
      child = spawn(PYTHON, [script, file, '--json'], { cwd: path.join(__dirname, 'scripts') });
    } catch (e) {
      return resolve({ code: -1, out: '', err: '启动失败: ' + e.message });
    }
    let out = '', err = '', done = false;
    const tm = setTimeout(() => {
      try { child.kill(); } catch (e) { /* ignore */ }
      finish({ code: -1, out, err: err + '\n处理超时（超过 ' + (PY_TIMEOUT / 1000) + ' 秒）' });
    }, PY_TIMEOUT);
    function finish(r) {
      if (done) return;
      done = true;
      clearTimeout(tm);
      resolve(r);
    }
    child.stdout.on('data', (d) => (out += d.toString('utf-8')));
    child.stderr.on('data', (d) => (err += d.toString('utf-8')));
    child.on('error', (e) => finish({ code: -1, out, err: err + ' ' + e.message }));
    child.on('close', (code) => finish({ code, out, err }));
  });
}

const server = http.createServer((req, res) => {
  if (req.url !== '/health' && req.url !== '/favicon.ico') {
    console.log(`[${new Date().toLocaleTimeString()}] ${req.method} ${req.url}`);
  }

  // ---- 推理接口（音频 / 视频） ----
  if (req.url === '/translate' && req.method === 'POST') {
    const chunks = [];
    let size = 0, aborted = false;
    req.on('data', (c) => {
      size += c.length;
      if (size > MAX_UPLOAD) {
        aborted = true;
        sendJSON(res, 413, { ok: false, error: '文件太大了（超过 300MB），截一段更短的视频或音频再试。' });
        req.destroy();
        return;
      }
      chunks.push(c);
    });
    req.on('error', (e) => { if (!aborted) sendJSON(res, 500, { ok: false, error: '收包失败: ' + e.message }); });
    req.on('end', () => {
      if (aborted) return;
      const buf = Buffer.concat(chunks);
      if (!buf.length) return sendJSON(res, 400, { ok: false, error: '没有收到文件内容' });

      const rawName = decodeURIComponent(req.headers['x-filename'] || 'audio.wav');
      let ext = path.extname(rawName).toLowerCase();
      if (!ALLOW_EXT.has(ext)) ext = '.wav';        // 未知类型仍按 wav 试，保持旧行为
      const kindHint = String(req.headers['x-media-kind'] || '').toLowerCase();

      // 要不要跑视觉：明确说是视频，或者是只可能是视频的容器（.webm 交给 hint 判断）
      const wantVision = kindHint === 'video' || FORCE_VISION.has(ext);

      const tmp = path.join(os.tmpdir(), 'cat_' + Date.now() + ext);
      fs.writeFileSync(tmp, buf);
      const t0 = Date.now();
      console.log(`   ↳ ${rawName}  ${(buf.length / 1048576).toFixed(1)}MB  ${wantVision ? '视频（听 + 看）' : '音频（只听）'}`);

      Promise.all([
        runPy(PREDICT, tmp),
        wantVision ? runPy(VISION, tmp) : Promise.resolve(null),
      ]).then(([pr, vr]) => {
        fs.unlink(tmp, () => {});

        // ---- 声音侧 ----
        let audio = null, audioErr = null;
        if (pr.code === 0) {
          audio = lastJSON(pr.out);
          if (!audio) audioErr = '声音结果解析失败: ' + cleanErr(pr.err);
        } else {
          audioErr = cleanErr(pr.err);
        }

        // ---- 画面侧 ----
        let vision = null, visionErr = null;
        if (vr) {
          if (vr.code === 0) {
            vision = lastJSON(vr.out);
            if (!vision) visionErr = '画面结果解析失败';
          } else {
            visionErr = cleanErr(vr.err);
          }
        }

        const audioOk = !!(audio && audio.ok);
        const sawCat = !!(vision && vision.ok && vision.has_cat);

        // 两边都没结论 → 明确报错，让界面照实说
        if (!audioOk && !sawCat) {
          // 损坏/读不出的文件：优先用画面侧那句更准确的说明（它知道文件能不能解析）
          const why = (vision && vision.ok === false && vision.error)
            ? vision.error
            : ((audio && audio.message) || audioErr || '没能处理这个文件');
          return sendJSON(res, 500, {
            ok: false,
            error: why,
            // audio 已经返回了结构化结果时，它的 error 就是给人看的那句话，
            // 原始 stderr 只留在服务端日志里，不要再漏到界面上
            err: audio ? '' : (audioErr || visionErr || ''),
          });
        }

        const out = Object.assign({}, audio || {}, {
          ok: true,
          media: {
            kind: wantVision ? 'video' : 'audio',
            name: rawName,
            bytes: buf.length,
            elapsed_ms: Date.now() - t0,
          },
          vision: vision || null,
          vision_error: visionErr || null,
        });

        // 视频没有声音（或声音读不出）但画面里看到猫 → 只报画面
        if (!audioOk) {
          out.status = 'vision_only';
          out.message = '这段视频里没有声音，或者声音我读不出来 —— 那我就只说说画面里看到的。';
        }
        if (!out.note) out.note = '趣味解读 · 仅供娱乐，不构成任何诊断';

        // 把这次喵语持久化存档（纯本地，不上云）：文件 + 带分析结果的索引
        try { out.archive = persistMeow(rawName, ext, buf, out); }
        catch (e) { console.error('喵语存档失败（不影响本次翻译）:', e.message); }
        console.log(`   ↳ 完成 ${Date.now() - t0}ms  status=${out.status || '?'}  画面=${vision ? (vision.has_cat ? '有猫' : '无猫') : '未分析'}  存档=${out.archive || '无'}`);
        sendJSON(res, 200, out);
      });
    });
    return;
  }

  // ---- 健康检查 ----
  if (req.url === '/health') return sendJSON(res, 200, { ok: true, service: 'cat-translator', port: PORT });

  // ---- 喵语档案馆：存档列表 + 音频回放（纯本地 data/uploads/）----
  if (req.url.split('?')[0] === '/archive' && req.method === 'GET') {
    const idx = path.join(UPLOAD_DIR, 'index.jsonl');
    let items = [];
    try {
      if (fs.existsSync(idx)) {
        items = fs.readFileSync(idx, 'utf8').trim().split('\n').filter(Boolean)
          .map((l) => { try { return JSON.parse(l); } catch (e) { return null; } })
          .filter(Boolean).reverse();   // 最新在前
      }
    } catch (e) { /* 读失败返回空列表 */ }
    return sendJSON(res, 200, { ok: true, count: items.length, items });
  }
  if (req.url.startsWith('/archive/audio/')) {
    const raw = req.url.split('?')[0].slice('/archive/audio/'.length);
    const fpath = path.join(UPLOAD_DIR, path.basename(decodeURIComponent(raw)));  // basename 防目录穿越
    if (!fpath.startsWith(UPLOAD_DIR) || !fs.existsSync(fpath)) { res.writeHead(404); return res.end('not found'); }
    const AMIME = { '.wav': 'audio/wav', '.mp3': 'audio/mpeg', '.m4a': 'audio/mp4', '.aac': 'audio/aac',
      '.ogg': 'audio/ogg', '.opus': 'audio/ogg', '.flac': 'audio/flac', '.amr': 'audio/amr', '.webm': 'audio/webm',
      '.mp4': 'video/mp4', '.mov': 'video/quicktime', '.m4v': 'video/mp4', '.avi': 'video/x-msvideo',
      '.mkv': 'video/x-matroska', '.wmv': 'video/x-ms-wmv', '.3gp': 'video/3gpp', '.ts': 'video/mp2t' };
    fs.readFile(fpath, (e, data) => {
      if (e) { res.writeHead(404); return res.end('not found'); }
      res.writeHead(200, { 'Content-Type': AMIME[path.extname(fpath).toLowerCase()] || 'application/octet-stream',
        'Accept-Ranges': 'none' });
      res.end(data);
    });
    return;
  }

  // ---- 静态文件 ----
  let rel = req.url.split('?')[0];
  if (rel === '/') rel = '/index.html';
  const file = path.join(WEB_DIR, path.normalize(rel).replace(/^([/\\])+/, ''));
  if (!file.startsWith(WEB_DIR)) { res.writeHead(403); return res.end('forbidden'); }
  fs.readFile(file, (e, data) => {
    if (e) { res.writeHead(404); return res.end('not found'); }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream' });
    res.end(data);
  });
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(`🐱 听喵说已启动: http://127.0.0.1:${PORT}`);
  console.log(`   听懂: ${PREDICT}`);
  console.log(`   看懂: ${VISION}`);
});
