
import express from "express";
import cors from "cors";
import crypto from "crypto";
import fetch from "node-fetch";
import { pipeline, env as tEnv } from "@xenova/transformers";

/* ======== ENV (Render → Settings → Environment) ========
Required:
  EVALUATOR_API_KEY
  CLOUDINARY_CLOUD_NAME
  CLOUDINARY_API_KEY
  CLOUDINARY_API_SECRET
  ADMIN_TOKEN
Optional:
  REFERENCE_FOLDERS_JSON   (JSON array of exact folder names)
  HF_MODEL                 (default 'Xenova/clip-vit-base-patch32')
  TRANSFORMERS_CACHE       (e.g. '/opt/render/project/.cache/transformers')
  MAX_REFS_PER_FOLDER      (default 40)
  MAX_EMBEDS_CONCURRENCY   (default 2)
======================================================== */

const KEY              = process.env.EVALUATOR_API_KEY;
const CLOUD_NAME       = process.env.CLOUDINARY_CLOUD_NAME;
const CLOUD_API_KEY    = process.env.CLOUDINARY_API_KEY;
const CLOUD_API_SECRET = process.env.CLOUDINARY_API_SECRET;
const ADMIN_TOKEN      = process.env.ADMIN_TOKEN || "CHANGE_ME";
const REFERENCE_FOLDERS_JSON = process.env.REFERENCE_FOLDERS_JSON || "";
const CLIP_MODEL_ID    = process.env.HF_MODEL || "Xenova/clip-vit-base-patch32";
const MAX_REFS_PER_FOLDER = Number(process.env.MAX_REFS_PER_FOLDER || 40);
const MAX_EMBEDS_CONCURRENCY = Number(process.env.MAX_EMBEDS_CONCURRENCY || 2);

// Optional: cache directory for model weights (survives restarts on Render)
if (process.env.TRANSFORMERS_CACHE) tEnv.cacheDir = process.env.TRANSFORMERS_CACHE;
tEnv.allowLocalModels = false;
tEnv.backends.onnx.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/@xenova/transformers/dist/";

const app = express();
app.use(express.json({ limit: "2mb" }));
app.use(cors({ origin: "*" })); // tighten to Bubble domain in prod
app.get("/ping", (req, res) => res.json({ ok: true }));


/* ================= Utils ================= */
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

/* ============ Embeddings (local CLIP) ============ */
let clipExtractorPromise = null;
async function getClipExtractor() {
  if (!clipExtractorPromise) {
    clipExtractorPromise = pipeline("feature-extraction", CLIP_MODEL_ID);
  }
  return clipExtractorPromise;
}

/* Downscale via Cloudinary to keep memory low */
async function fetchImageBuffer(url) {
  let resized = url;
  try {
    const marker = "/image/upload/";
    const idx = url.indexOf(marker);
    if (idx !== -1 && !url.includes("/image/upload/f_jpg")) {
      resized = url.slice(0, idx + marker.length) + "f_jpg,q_auto,w_512,c_limit/" + url.slice(idx + marker.length);
    }
  } catch(_) {}
  const r = await fetch(resized);
  if (!r.ok) throw new Error(`image_fetch_failed:${r.status}`);
  const ab = await r.arrayBuffer();
  return Buffer.from(ab);
}

async function embedOne(url) {
  const buf = await fetchImageBuffer(url);
  const extractor = await getClipExtractor();
  const out = await extractor(buf, { pooling: "mean", normalize: true });
  return Float32Array.from(out.data);
}

/* ======= Cloudinary helpers ======= */
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

/* ======= Memory-light reference loading (centroids only) ======= */
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
  if (!CLOUD_NAME || !CLOUD_API_KEY || !CLOUD_API_SECRET) {
    throw new Error("Cloudinary credentials missing");
  }
  const folderNames = await discoverReferenceFolders();
  const out = [];

  for (const fname of folderNames) {
    const urlsAll = await searchFolderAssets(fname);
    if (!urlsAll.length) continue;

    // sample first MAX_REFS_PER_FOLDER (or randomize if you prefer)
    const urls = urlsAll.slice(0, MAX_REFS_PER_FOLDER);
    const gender = inferGenderFromFolder(fname);

    // Streaming centroid: sum vectors then divide → no need to store each vec
    let centroid = null;
    let count = 0;

    // tiny concurrency limiter
    const queue = [...urls];
    const workers = Array(Math.min(MAX_EMBEDS_CONCURRENCY, urls.length)).fill(0).map(async () => {
      while (queue.length) {
        const u = queue.shift();
        try {
          const v = await embedOne(u);
          if (!centroid) centroid = new Float32Array(v.length);
          for (let i=0;i<v.length;i++) centroid[i] += v[i];
          count++;
        } catch (e) {
          console.error("embed_fail", fname, u, e.message);
        }
      }
    });
    await Promise.all(workers);

    if (count > 0) {
      for (let i=0;i<centroid.length;i++) centroid[i] /= count;
      out.push({ label: fname, gender, size: count, centroid });
    }
    // important: let GC reclaim buffers from this folder before moving on
    global.gc?.();
  }

  clusters.length = 0;
  clusters.push(...out);
  console.log("Loaded centroids:", clusters.map(c => `${c.label}(${c.size})`).join(", "));
}

/* =============== Admin endpoints =============== */
app.get("/admin/reload_refs", async (req, res) => {
  if (req.query.token !== ADMIN_TOKEN) return res.status(401).json({ error: "Unauthorized" });
  try {
    await loadClusters();
    res.json({
      ok: true,
      clusters: clusters.map(c => ({ label: c.label, gender: c.gender, size: c.size }))
    });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: String(e.message || e) });
  }
});

app.get("/admin/test_embed", async (req, res) => {
  if (req.query.token !== ADMIN_TOKEN) return res.status(401).json({ error: "Unauthorized" });
  try {
    const url = req.query.url;
    if (!url) return res.status(400).json({ error: "missing ?url=" });
    const vec = await embedOne(url);
    res.json({ ok: true, dim: vec.length, model: CLIP_MODEL_ID });
  } catch (e) {
    res.status(500).json({ ok: false, error: String(e?.message || e) });
  }
});

/* ================= Evaluate ================= */
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

    // Preferences
    const PREFS = {
      female: {
        height: { min: 172, target: 177, max: 182 },
        meas_target: { b: 82, w: 60, h: 88 },
        meas_tol:    { b: 6,  w: 4,  h: 6 }
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
    const m = parseMeas(measurements);

    // male band scoring
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
    }

    // Face similarity vs centroids (average applicant vector vs folder centroids)
    let faceScore = 0.5, faceCluster = "none", faceReason = "face similarity neutral";
    try {
      if (clusters.length) {
        // Embed applicant photos (sequential to keep RAM low)
        const photoVecs = [];
        for (const url of photos.slice(0, 5)) { // hard cap 5 photos for RAM
          try { photoVecs.push(await embedOne(url)); }
          catch (e) { console.error("app_embed_fail", url, e.message); }
        }
        if (photoVecs.length) {
          const dim = photoVecs[0].length;
          const avg = new Float32Array(dim);
          for (const v of photoVecs) for (let i=0;i<dim;i++) avg[i] += v[i];
          for (let i=0;i<dim;i++) avg[i] /= photoVecs.length;

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

    // extras
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
}); app.get("/", (_req, res) => res.send("OK")); app.get("/admin/warmup", async (req, res) => {
  if (req.query.token !== ADMIN_TOKEN) return res.status(401).json({ error: "Unauthorized" });
  try {
    // just load the model once; no Cloudinary fetch
    const extractor = await (await import("@xenova/transformers")).pipeline("feature-extraction", CLIP_MODEL_ID);
    // touch it once so weights are fully cached
    await extractor(new Uint8Array([0,1,2,3]), { pooling: "mean", normalize: true }).catch(() => {});
    return res.json({ ok: true, warmed: true, model: CLIP_MODEL_ID });
  } catch (e) {
    return res.status(500).json({ ok: false, error: String(e?.message || e) });
  }
});



/* ================= Boot ================= */
const port = process.env.PORT || 10000;
app.post("/evaluate_mock", (req, res) => {
  const body = req.body ?? {};
  let { photos = [], gender = "", height_cm = "", age = "", measurements = "" } = body;

  if (typeof photos === "string") {
    photos = photos.split(",").map(s => s.trim()).filter(Boolean);
  }
  const detailsArr = Array.isArray(photos) ? photos.slice(0, 2) : [];
  const details_text = detailsArr.join("; ");

  return res.status(200).json({
    decision: "review",
    confidence: 0.75,
    reason: "Endpoint OK; mock evaluation response.",
    details: detailsArr.map(u => ({ url: u, desc: "provided", best_similarity: 0.0 })),
    details_text,
    parsed_measurements_cm: null
  });
});

app.listen(port, () => {
  console.log("Evaluator running on :" + port, "(lightweight mode)");
  // background warm-up (doesn't block requests)
  setTimeout(async () => {
    try {
      const { pipeline } = await import("@xenova/transformers");
      const extractor = await pipeline("feature-extraction", CLIP_MODEL_ID);
      // tiny dummy call to ensure wasm+weights are cached
      await extractor(new Uint8Array([0,1,2,3]), { pooling: "mean", normalize: true }).catch(() => {});
      console.log("Model warm-up complete");
    } catch (e) {
      console.log("Model warm-up skipped:", e?.message || e);
    }
  }, 2000);
});
