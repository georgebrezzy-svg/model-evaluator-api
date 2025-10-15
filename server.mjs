import express from "express";
import cors from "cors";
import crypto from "crypto";
import fetch from "node-fetch";

/* ========= ENV (Render → Settings → Environment) ========= */
const KEY              = process.env.EVALUATOR_API_KEY;         // Bubble auth
const HF_KEY           = process.env.HUGGINGFACE_API_KEY;       // hf_...
const CLOUD_NAME       = process.env.CLOUDINARY_CLOUD_NAME;     // e.g. dphw7jvyg
const CLOUD_API_KEY    = process.env.CLOUDINARY_API_KEY;
const CLOUD_API_SECRET = process.env.CLOUDINARY_API_SECRET;
const ADMIN_TOKEN      = process.env.ADMIN_TOKEN || "CHANGE_ME";
/* Optional */
const REFERENCE_FOLDERS_JSON = process.env.REFERENCE_FOLDERS_JSON || "";
const HF_MODEL_ENV     = process.env.HF_MODEL || "";            // e.g. sentence-transformers/clip-ViT-B-32

/* ========= App ========= */
const app = express();
app.use(express.json({ limit: "8mb" }));
app.use(cors({ origin: "*" })); // tighten to your Bubble domain(s) for production

/* ========= Utils ========= */
const clamp = (x, a=0, b=1) => Math.max(a, Math.min(b, x));
const basicAuth = "Basic " + Buffer.from(`${CLOUD_API_KEY}:${CLOUD_API_SECRET}`).toString("base64");

const parseMeas = (mStr) => {
  if (!mStr || typeof mStr !== "string") return null;
  const nums = mStr.match(/\d+(\.\d+)?/g);
  if (!nums || nums.length < 3) return null;
  return { b: Number(nums[0]), w: Number(nums[1]), h: Number(nums[2]) };
};

const cosine = (a, b) => {
  let dot=0, na=0, nb=0;
  const L = Math.min(a.length, b.length);
  for (let i=0;i<L;i++){ const x=a[i], y=b[i]; dot+=x*y; na+=x*x; nb+=y*y; }
  return (na && nb) ? dot / Math.sqrt(na*nb) : 0;
};

const heightScoreFactory = ({min, target, max}) => (h) => {
  if (!h || !Number(h)) return 0;
  const x = Number(h);
  if (x <= min) return 0;
  if (x >= max) return 1;
  let s = (x - min) / (max - min);
  const span = (max - min) / 2;
  const bonus = Math.exp(-Math.pow((x - target) / (span/2), 2)) * 0.15;
  return clamp(s + bonus);
};

/* ========= Embeddings (HF) ========= */
const embedCache = new Map();  // url -> Float32Array
const clusters   = [];         // [{label, gender, urls:[...], vecs:[Float32Array]}]

// Preferred models (we’ll try in order; you can force one with HF_MODEL)
const HF_MODELS = [
  HF_MODEL_ENV || "sentence-transformers/clip-ViT-B-32",
  "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
  "openai/clip-vit-base-patch32"
];

// We’ll try ALL three Inference API routes; different accounts enable different ones
const HF_ROUTES = (modelId) => [
  `https://api-inference.huggingface.co/pipeline/image-feature-extraction/${modelId}`,
  `https://api-inference.huggingface.co/pipeline/feature-extraction/${modelId}`,
  `https://api-inference.huggingface.co/models/${modelId}`
];

/* Downscale Cloudinary images to keep HF happy (w_512, jpg, q_auto) */
async function fetchImageBuffer(url) {
  let resized = url;
  try {
    const marker = "/image/upload/";
    const idx = url.indexOf(marker);
    if (idx !== -1 && !url.includes("/image/upload/f_jpg")) {
      resized = url.slice(0, idx + marker.length) + "f_jpg,q_auto,w_512,c_limit/" + url.slice(idx + marker.length);
    }
  } catch (_) { /* noop */ }

  const r = await fetch(resized);
  if (!r.ok) throw new Error(`image_fetch_failed:${r.status}`);
  const ab = await r.arrayBuffer();
  return Buffer.from(ab);
}

// HF sometimes returns [tokens][dim]; average over tokens.
// Sometimes it returns a single vector [dim].
function poolToVector(jsonOut) {
  if (Array.isArray(jsonOut) && Array.isArray(jsonOut[0])) {
    const rows = jsonOut.length, cols = jsonOut[0].length;
    const v = new Float32Array(cols);
    for (let i=0;i<rows;i++){
      const row = jsonOut[i];
      for (let j=0;j<cols;j++) v[j] += row[j];
    }
    for (let j=0;j<cols;j++) v[j] /= rows;
    return v;
  }
  return Float32Array.from(jsonOut);
}

/* Try a (route,model) with small retry/backoff on 5xx */
async function tryEmbed(route, buf) {
  const maxAttempts = 3;
  let last;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    const resp = await fetch(route, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${HF_KEY}`,
        "Content-Type": "application/octet-stream",
        Accept: "application/json",
        "X-Wait-For-Model": "true"
      },
      body: buf
    });
    if (resp.ok) {
      const data = await resp.json();
      return poolToVector(data);
    }
    const txt = await resp.text();
    last = new Error(`hf_${resp.status}_${route}:${txt || "error"}`);
    if ([500, 502, 503, 504].includes(resp.status) && attempt < maxAttempts) {
      await new Promise(r => setTimeout(r, 400 * attempt));
      continue;
    }
    throw last;
  }
  throw last;
}

async function getEmbedding(url) {
  if (embedCache.has(url)) return embedCache.get(url);
  if (!HF_KEY) throw new Error("HUGGINGFACE_API_KEY missing");

  const buf = await fetchImageBuffer(url);
  let lastErr;

  for (const model of HF_MODELS) {
    const routes = HF_ROUTES(model);
    for (const route of routes) {
      try {
        const vec = await tryEmbed(route, buf);
        embedCache.set(url, vec);
        return vec;
      } catch (e) {
        lastErr = e;
        console.error("hf_try_fail", model, e.message);
      }
    }
  }
  throw lastErr || new Error("hf_all_endpoints_failed");
}

/* ========= Cloudinary admin helpers ========= */
async function listRootFolders() {
  const url = `https://api.cloudinary.com/v1_1/${CLOUD_NAME}/folders`;
  const r = await fetch(url, { headers: { Authorization: basicAuth } });
  if (!r.ok) throw new Error("cloudinary_list_folders_failed:" + r.status);
  const j = await r.json();
  return j.folders?.map(f => f.name) || [];
}

async function searchFolderAssets(folderName) {
  const url = `https://api.cloudinary.com/v1_1/${CLOUD_NAME}/resources/search`;
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

async function discoverReferenceFolders() {
  if (REFERENCE_FOLDERS_JSON.trim()) {
    try { return JSON.parse(REFERENCE_FOLDERS_JSON); }
    catch { throw new Error("Invalid REFERENCE_FOLDERS_JSON"); }
  }
  const names = await listRootFolders();
  return names.filter(n => /^reference\s/i.test(n)); // starts with "Reference "
}

async function loadClusters() {
  if (!CLOUD_NAME || !CLOUD_API_KEY || !CLOUD_API_SECRET) {
    throw new Error("Cloudinary credentials missing");
  }
  const folderNames = await discoverReferenceFolders();
  const out = [];
  for (const fname of folderNames) {
    const urls = await searchFolderAssets(fname);
    if (!urls.length) continue;
    const gender = inferGenderFromFolder(fname);
    const vecs = [];
    for (const u of urls) {
      try { vecs.push(await getEmbedding(u)); }
      catch (e) { console.error("embed_fail", fname, u, e.message); }
    }
    if (vecs.length) out.push({ label: fname, gender, urls, vecs });
  }
  clusters.length = 0;
  clusters.push(...out);
  console.log("Loaded clusters:", clusters.map(c => `${c.label}(${c.vecs.length})`).join(", "));
}

/* ========= Admin endpoints ========= */
app.get("/admin/reload_refs", async (req, res) => {
  if (req.query.token !== ADMIN_TOKEN) return res.status(401).json({ error: "Unauthorized" });
  try {
    await loadClusters();
    res.json({
      ok: true,
      clusters: clusters.map(c => ({ label: c.label, gender: c.gender, size: c.vecs.length }))
    });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: String(e.message || e) });
  }
});

// Test a single image quickly (debug)
app.get("/admin/test_embed", async (req, res) => {
  if (req.query.token !== ADMIN_TOKEN) return res.status(401).json({ error: "Unauthorized" });
  try {
    const url = req.query.url;
    if (!url) return res.status(400).json({ error: "missing ?url=" });
    const vec = await getEmbedding(url);
    res.json({ ok: true, dim: vec.length, models_tried: HF_MODELS });
  } catch (e) {
    res.status(500).json({ ok: false, error: String(e?.message || e) });
  }
});

/* ========= Health ========= */
app.get("/", (_, res) => res.send("OK"));

/* ========= Evaluate ========= */
app.post("/evaluate", async (req, res) => {
  try {
    const auth = req.headers.authorization || "";
    if (!KEY || !auth.startsWith("Bearer ") || auth.slice(7) !== KEY) {
      return res.status(401).json({ error: "Unauthorized" });
    }

    const { photos = [], gender, height_cm, age, measurements } = req.body || {};
    if (!Array.isArray(photos) || photos.length === 0) {
      return res.status(400).json({ error: "`photos` must be a non-empty array of URLs" });
    }

    // Preferences (female target/tolerance; male bands)
    const PREFS = {
      female: {
        height: { min: 172, target: 177, max: 182 },
        meas_target: { b: 82, w: 60, h: 88 },
        meas_tol:    { b: 6,  w: 4,  h: 6  }
      },
      male: {
        height: { min: 182, target: 186, max: 191 },
        meas_bands: {
          b: { min: 88, ideal: [90, 94], max: 96 },
          w: { min: 71, ideal: [73, 77], max: 82 },
          h: { min: 88, ideal: [90, 94], max: 96 }
        }
      }
    };

    const GEN = (gender || "female").toLowerCase().startsWith("m") ? "male" : "female";
    const P   = PREFS[GEN];
    const heightScore = heightScoreFactory(P.height);

    // Measurements score
    const m = parseMeas(measurements);
    const bandScore = (val, band) => {
      if (val == null || !band) return 0;
      const v = Number(val);
      const { min, ideal, max } = band;
      if (v <= min || v >= max) return 0;
      if (v >= ideal[0] && v <= ideal[1]) return 1;
      if (v <  ideal[0]) return (v - min) / (ideal[0] - min);
      return (max - v) / (max - ideal[1]);
    };
    let measScore = 0;
    if (GEN === "male" && P.meas_bands && m) {
      const sB = bandScore(m.b, P.meas_bands.b);
      const sW = bandScore(m.w, P.meas_bands.w);
      const sH = bandScore(m.h, P.meas_bands.h);
      measScore = clamp((sB + 1.3*sW + sH) / 3.3);
      const leanBonus =
        (m.w <= P.meas_bands.w.ideal[1] && m.b <= P.meas_bands.b.ideal[1] && m.h <= P.meas_bands.h.ideal[1]) ? 0.03 : 0;
      measScore = clamp(measScore + leanBonus);
    } else if (m && P.meas_target && P.meas_tol) {
      const dxB = Math.abs(m.b - P.meas_target.b) / P.meas_tol.b;
      const dxW = Math.abs(m.w - P.meas_target.w) / P.meas_tol.w;
      const dxH = Math.abs(m.h - P.meas_target.h) / P.meas_tol.h;
      const sB = clamp(1 - dxB);
      const sW = clamp(1 - dxW);
      const sH = clamp(1 - dxH);
      measScore = clamp((sB + 1.2*sW + sH) / 3.2);
    } else {
      measScore = 0;
    }

    // Face similarity
    let faceScore = 0.5, faceCluster = "none", faceReason = "face similarity neutral";
    try {
      if (clusters.length) {
        const pool = clusters.some(c => c.gender !== "unknown")
          ? clusters.filter(c => c.gender === GEN || c.gender === "unknown")
          : clusters;

        const photoVecs = [];
        for (const url of photos) {
          try { photoVecs.push(await getEmbedding(url)); }
          catch (e) { console.error("app_embed_fail", url, e.message); }
        }

        let maxSim = -1, bestLabel = "none";
        for (const pv of photoVecs) {
          for (const c of pool) {
            for (const rv of c.vecs) {
              const s = cosine(pv, rv);
              if (s > maxSim) { maxSim = s; bestLabel = c.label; }
            }
          }
        }
        if (maxSim >= -1) {
          faceScore = clamp((maxSim + 1) / 2);
          faceCluster = bestLabel;
          if (faceScore >= 0.70) faceReason = "face matches reference look";
          else if (faceScore >= 0.55) faceReason = "some similarity to reference look";
          else faceReason = "low similarity to reference look";
        }
      } else {
        faceReason = "no reference faces loaded";
      }
    } catch (e) {
      console.error("face_similarity_error", e.message);
      faceReason = "face similarity unavailable";
      faceScore = 0.5;
    }

    // Photo count + tiny noise
    const nPhotos = photos.length;
    const photoBoost = clamp((nPhotos - 2) * 0.04, 0, 0.12);
    const hash = crypto.createHash("sha256").update(photos.join("|")).digest("hex");
    const n = parseInt(hash.slice(0, 6), 16);
    const tinyNoise = (n % 100) / 10000;

    // Combine
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
    if (nPhotos >= 3) reasons.push("enough digitals provided");
    else if (nPhotos < 2) reasons.push("insufficient digitals");
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

/* ========= Boot ========= */
const port = process.env.PORT || 10000;
app.listen(port, async () => {
  console.log("Evaluator running on :" + port);
  try {
    await loadClusters();
  } catch (e) {
    console.error("Initial reference load failed:", e.message);
  }
});
