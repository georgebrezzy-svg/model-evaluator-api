
// ---------- server.mjs (Node 18+, ES modules) ----------
import express from "express";
import cors from "cors";
import fetch from "node-fetch";
import crypto from "crypto";

// ── ENV ───────────────────────────────────────────────────────────────────────
const LIGHT_MODE = process.env.LIGHT_MODE === "1";          // ← toggle mock mode
const EVALUATOR_API_KEY = process.env.EVALUATOR_API_KEY;    // required for /evaluate
const CLOUDINARY_CLOUD_NAME = process.env.CLOUDINARY_CLOUD_NAME || "";
const CLOUDINARY_API_KEY = process.env.CLOUDINARY_API_KEY || "";
const CLOUDINARY_API_SECRET = process.env.CLOUDINARY_API_SECRET || "";
const ADMIN_TOKEN = process.env.ADMIN_TOKEN || "CHANGE_ME";
const REFERENCE_FOLDERS_JSON = process.env.REFERENCE_FOLDERS_JSON || "";
const HF_MODEL = process.env.HF_MODEL || "Xenova/clip-vit-base-patch32";
const TRANSFORMERS_CACHE = process.env.TRANSFORMERS_CACHE || ""; // optional
const MAX_REFS_PER_FOLDER = Number(process.env.MAX_REFS_PER_FOLDER || 40);
const MAX_EMBEDS_CONCURRENCY = Number(process.env.MAX_EMBEDS_CONCURRENCY || 2);

// ── Lazy load transformers ONLY when needed (heavy) ──────────────────────────
let _tf = null;
async function getTF() {
  if (!_tf) {
    _tf = await import("@xenova/transformers");
    // optional cache dir (improves restarts on Render)
    if (TRANSFORMERS_CACHE) _tf.env.cacheDir = TRANSFORMERS_CACHE;
    _tf.env.allowLocalModels = false;
    _tf.env.backends.onnx.wasm.wasmPaths =
      "https://cdn.jsdelivr.net/npm/@xenova/transformers/dist/";
  }
  return _tf;
}

// ── App bootstrap (light) ─────────────────────────────────────────────────────
const app = express();
app.use(cors());
app.use(express.json({ limit: "2mb" }));

app.get("/ping", (req, res) => res.json({ ok: true, mode: LIGHT_MODE ? "light" : "heavy" }));
app.get("/", (_req, res) => res.send("OK"));

// ── Utils ────────────────────────────────────────────────────────────────────
const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
const basicAuth = "Basic " + Buffer.from(`${CLOUDINARY_API_KEY}:${CLOUDINARY_API_SECRET}`).toString("base64");

const parseMeas = (mStr) => {
  if (!mStr || typeof mStr !== "string") return null;
  const nums = mStr.match(/\d+(\.\d+)?/g);
  if (!nums || nums.length < 3) return null;
  return { b: Number(nums[0]), w: Number(nums[1]), h: Number(nums[2]) };
};

const cosine = (a, b) => {
  let dot = 0, na = 0, nb = 0;
  const L = Math.min(a.length, b.length);
  for (let i = 0; i < L; i++) {
    const x = a[i], y = b[i];
    dot += x * y; na += x * x; nb += y * y;
  }
  return (na && nb) ? dot / Math.sqrt(na * nb) : 0;
};

const heightScoreFactory = ({ min, target, max }) => (h) => {
  if (!h || !Number(h)) return 0;
  const x = Number(h);
  if (x <= min) return 0;
  if (x >= max) return 1;
  let s = (x - min) / (max - min);
  const span = (max - min) / 2;
  const bonus = Math.exp(-Math.pow((x - target) / (span / 2), 2)) * 0.15;
  return clamp(s + bonus);
};

// ── Cloudinary helpers (light) ────────────────────────────────────────────────
async function listRootFolders() {
  const url = `https://api.cloudinary.com/v1_1/${CLOUDINARY_CLOUD_NAME}/folders`;
  const r = await fetch(url, { headers: { Authorization: basicAuth } });
  if (!r.ok) throw new Error("cloudinary_list_folders_failed:" + r.status);
  const j = await r.json();
  return j.folders?.map(f => f.name) || [];
}
async function searchFolderAssets(folderName) {
  const url = `https://api.cloudinary.com/v1_1/${CLOUDINARY_CLOUD_NAME}/resources/search`;
  const body = { expression: `resource_type:image AND folder="${folderName}"`, max_results: 500 };
  const r = await fetch(url, {
    method: "POST",
    headers: { Authorization: basicAuth, "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!r.ok) throw new Error("cloudinary_search_failed:" + (await r.text()));
  const j = await r.json();
  return (j.resources || []).map(x => x.secure_url);
}
function inferGenderFromFolder(name) {
  const n = name.toLowerCase();
  if (n.includes("female")) return "female";
  if (n.includes("male")) return "male";
  return "unknown";
}

// ── Embeddings (lazy transformers use) ────────────────────────────────────────
async function fetchImageBuffer(url) {
  // downscale via Cloudinary if possible
  let normalized = url;
  try {
    const marker = "/image/upload/";
    const idx = url.indexOf(marker);
    if (idx !== -1 && !url.includes("/image/upload/f_jpg")) {
      normalized = url.slice(0, idx + marker.length) + "f_jpg,q_auto,w_512,c_limit/" + url.slice(idx + marker.length);
    }
  } catch (_) {}
  const r = await fetch(normalized);
  if (!r.ok) throw new Error(`image_fetch_failed:${r.status}`);
  const ab = await r.arrayBuffer();
  return new Uint8Array(ab);
}

async function embedOne(url) {
  const { pipeline } = await getTF();                    // lazy
  const buf = await fetchImageBuffer(url);
  const extractor = await pipeline("feature-extraction", HF_MODEL);
  const out = await extractor(buf, { pooling: "mean", normalize: true });
  return Float32Array.from(out.data);
}

// ── Reference centroids (memory-light) ────────────────────────────────────────
const clusters = []; // [{ label, gender, size, centroid: Float32Array }]

async function discoverReferenceFolders() {
  if (REFERENCE_FOLDERS_JSON.trim()) {
    try { return JSON.parse(REFERENCE_FOLDERS_JSON); }
    catch { throw new Error("Invalid REFERENCE_FOLDERS_JSON"); }
  }
  const names = await listRootFolders();
  return names.filter(n => /^reference\s/i.test(n));
}

async function loadClusters() {
  if (!CLOUDINARY_CLOUD_NAME || !CLOUDINARY_API_KEY || !CLOUDINARY_API_SECRET) {
    throw new Error("Cloudinary credentials missing");
  }
  const folderNames = await discoverReferenceFolders();
  const out = [];

  for (const fname of folderNames) {
    const urlsAll = await searchFolderAssets(fname);
    if (!urlsAll.length) continue;

    const urls = urlsAll.slice(0, MAX_REFS_PER_FOLDER);
    const gender = inferGenderFromFolder(fname);

    let centroid = null;
    let count = 0;

    const queue = [...urls];
    const workers = Array(Math.min(MAX_EMBEDS_CONCURRENCY, urls.length)).fill(0).map(async () => {
      while (queue.length) {
        const u = queue.shift();
        try {
          const v = await embedOne(u);
          if (!centroid) centroid = new Float32Array(v.length);
          for (let i = 0; i < v.length; i++) centroid[i] += v[i];
          count++;
        } catch (e) {
          console.error("embed_fail", fname, u, e.message);
        }
      }
    });
    await Promise.all(workers);

    if (count > 0) {
      for (let i = 0; i < centroid.length; i++) centroid[i] /= count;
      out.push({ label: fname, gender, size: count, centroid });
    }
    global.gc?.();
  }

  clusters.length = 0;
  clusters.push(...out);
  console.log("Loaded centroids:", clusters.map(c => `${c.label}(${c.size})`).join(", "));
}

// ── Admin endpoints ───────────────────────────────────────────────────────────
app.get("/admin/reload_refs", async (req, res) => {
  if (req.query.token !== ADMIN_TOKEN) return res.status(401).json({ error: "Unauthorized" });
  if (LIGHT_MODE) return res.json({ ok: true, skipped: "LIGHT_MODE" });
  try {
    await loadClusters();
    res.json({ ok: true, clusters: clusters.map(c => ({ label: c.label, gender: c.gender, size: c.size })) });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: String(e?.message || e) });
  }
});

app.get("/admin/test_embed", async (req, res) => {
  if (req.query.token !== ADMIN_TOKEN) return res.status(401).json({ error: "Unauthorized" });
  if (LIGHT_MODE) return res.json({ ok: true, skipped: "LIGHT_MODE" });
  try {
    const url = req.query.url;
    if (!url) return res.status(400).json({ error: "missing ?url=" });
    const vec = await embedOne(url);
    res.json({ ok: true, dim: vec.length, model: HF_MODEL });
  } catch (e) {
    res.status(500).json({ ok: false, error: String(e?.message || e) });
  }
});

// ── Evaluate (with LIGHT_MODE short-circuit) ──────────────────────────────────
app.post("/evaluate", async (req, res) => {
  try {
    // Auth (same as before)
    const auth = req.headers.authorization || "";
    if (!EVALUATOR_API_KEY || !auth.startsWith("Bearer ") || auth.slice(7) !== EVALUATOR_API_KEY) {
      return res.status(401).json({ error: "Unauthorized" });
    }

    const { photos = [], gender, height_cm, age, measurements } = req.body || {};
    if (!Array.isArray(photos) || photos.length === 0) {
      return res.status(400).json({ error: "`photos` must be a non-empty array of URLs" });
    }

    // FAST, LOW-MEMORY RETURN
    if (LIGHT_MODE) {
      const firstTwo = photos.slice(0, 2);
      const details_text = firstTwo.join("; ");
      const detailsString = `photos=${photos.length}, gender=${gender ?? "n/a"}, h=${height_cm ?? "n/a"}, age=${age ?? "n/a"}, meas=${measurements ?? "n/a"}`;

      return res.status(200).json({
        decision: "review",
        confidence: 0.75,
        reason: "Endpoint OK; mock (LIGHT_MODE).",
        details: detailsString,        // keep for Bubble compatibility
        details_text,
        face_similarity: 0.5,
        face_cluster: "none"
      });
    }

    // ── REAL logic (same scoring as before, trimmed for brevity) ──────────────
    const PREFS = {
      female: { height: { min: 172, target: 177, max: 182 }, meas_target: { b: 82, w: 60, h: 88 }, meas_tol: { b: 6, w: 4, h: 6 } },
      male:   { height: { min: 182, target: 186, max: 191 },
                meas_bands: { b: { min: 88, ideal: [90, 94], max: 96 }, w: { min: 71, ideal: [73, 77], max: 82 }, h: { min: 88, ideal: [90, 94], max: 96 } } }
    };
    const GEN = (gender || "female").toLowerCase().startsWith("m") ? "male" : "female";
    const P   = PREFS[GEN];

    const heightScore = heightScoreFactory(P.height);
    const m = parseMeas(measurements);

    const bandScore = (val, band) => {
      if (val == null || !band) return 0;
      const v = Number(val);
      const { min, ideal, max } = band;
      if (v <= min || v >= max) return 0;
      if (v >= ideal[0] && v <= ideal[1]) return 1;
      if (v < ideal[0]) return (v - min) / (ideal[0] - min);
      return (max - v) / (max - ideal[1]);
    };

    let measScore = 0;
    if (GEN === "male" && P.meas_bands && m) {
      const sB = bandScore(m.b, P.meas_bands.b);
      const sW = bandScore(m.w, P.meas_bands.w);
      const sH = bandScore(m.h, P.meas_bands.h);
      measScore = clamp((sB + 1.3 * sW + sH) / 3.3);
    } else if (m && P.meas_target && P.meas_tol) {
      const dxB = Math.abs(m.b - P.meas_target.b) / P.meas_tol.b;
      const dxW = Math.abs(m.w - P.meas_target.w) / P.meas_tol.w;
      const dxH = Math.abs(m.h - P.meas_target.h) / P.meas_tol.h;
      measScore = clamp((1 - dxB + 1.2 * (1 - dxW) + (1 - dxH)) / 3.2);
    }

    // Face similarity vs loaded centroids
    let faceScore = 0.5, faceCluster = "none", faceReason = "face similarity neutral";
    try {
      if (clusters.length) {
        const photoVecs = [];
        for (const url of photos.slice(0, 5)) {
          try { photoVecs.push(await embedOne(url)); } catch (e) { console.error("app_embed_fail", url, e.message); }
        }
        if (photoVecs.length) {
          const dim = photoVecs[0].length;
          const avg = new Float32Array(dim);
          for (const v of photoVecs) for (let i = 0; i < dim; i++) avg[i] += v[i];
          for (let i = 0; i < dim; i++) avg[i] /= photoVecs.length;

          let maxSim = -1, bestLabel = "none";
          for (const c of clusters) {
            const s = cosine(avg, c.centroid);
            if (s > maxSim) { maxSim = s; bestLabel = c.label; }
          }
          faceScore = clamp((maxSim + 1) / 2);
          faceCluster = bestLabel;
          if (faceScore >= 0.70) faceReason = "face matches reference look";
          else if (faceScore >= 0.55) faceReason = "some similarity to reference look";
          else faceReason = "low similarity to reference look";
        } else {
          faceReason = "no valid photos to analyze";
        }
      } else {
        faceReason = "no reference faces loaded";
      }
    } catch (e) {
      console.error("face_similarity_error", e.message);
      faceReason = "face similarity unavailable";
      faceScore = 0.5;
    }

    const nPhotos = photos.length;
    const photoBoost = clamp((nPhotos - 2) * 0.04, 0, 0.12);
    const hash = crypto.createHash("sha256").update(photos.join("|")).digest("hex");
    const n = parseInt(hash.slice(0, 6), 16);
    const tinyNoise = (n % 100) / 10000;

    let confidence =
      0.40 * measScore +
      0.25 * heightScore(height_cm) +
      0.30 * faceScore +
      0.05 * (nPhotos ? 0.6 : 0) +
      photoBoost +
      tinyNoise;
    confidence = clamp(confidence);

    const decision = confidence >= 0.70 ? "proceed" : confidence >= 0.45 ? "review" : "reject";

    const reasons = [];
    if (height_cm) {
      const hs = heightScore(height_cm);
      if (hs >= 0.75) reasons.push("height in editorial sweet spot");
      else if (hs >= 0.45) reasons.push("height within range");
      else reasons.push("height below preferred range");
    }
    if (m) {
      if (measScore >= 0.75) reasons.push("measurements in ideal window");
      else if (measScore >= 0.45) reasons.push("measurements within acceptable band");
      else reasons.push("measurements outside preferred band");
    }
    reasons.push(`${faceReason} (${faceCluster})`);

    const reason = reasons.join("; ");
    const details = `photos=${nPhotos}, gender=${gender ?? "n/a"}, h=${height_cm ?? "n/a"}, age=${age ?? "n/a"}, meas=${measurements ?? "n/a"}`;

    return res.json({
      decision, confidence, reason, details,
      face_similarity: Number(faceScore.toFixed(3)),
      face_cluster: faceCluster
    });
  } catch (e) {
    console.error(e);
    return res.status(500).json({ error: "server_error", message: String(e?.message || e) });
  }
});

// Optional simple mock (kept for testing)
app.post("/evaluate_mock", (req, res) => {
  const b = req.body ?? {};
  let photos = Array.isArray(b.photos) ? b.photos
              : (typeof b.photos === "string" ? b.photos.split(",") : []);
  photos = photos.map(s => String(s).trim()).filter(Boolean);
  const details_text = photos.slice(0, 2).join("; ");
  res.json({
    decision: "review",
    confidence: 0.75,
    reason: "Endpoint OK; mock evaluation response.",
    details: photos.slice(0,2).map(u => ({ url: u, desc: "provided", best_similarity: 0 })),
    details_text,
    parsed_measurements_cm: null
  });
});

// ── Start server ──────────────────────────────────────────────────────────────
const PORT = process.env.PORT || 10000;
app.listen(PORT, () => {
  console.log(`Model Evaluator listening on :${PORT} ${LIGHT_MODE ? "(LIGHT_MODE)" : "(HEAVY)"}`);
  // background warm-up (skip in light mode)
  setTimeout(async () => {
    if (LIGHT_MODE) return console.log("Warm-up skipped (LIGHT_MODE).");
    try {
      const { pipeline } = await getTF();
      const extractor = await pipeline("feature-extraction", HF_MODEL);
      await extractor(new Uint8Array([0, 1, 2, 3]), { pooling: "mean", normalize: true }).catch(() => {});
      console.log("Model warm-up complete");
    } catch (e) {
      console.log("Warm-up skipped:", e?.message || e);
    }
  }, 1500);
});
// ---------- end ----------
