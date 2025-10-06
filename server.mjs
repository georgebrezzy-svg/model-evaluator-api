import express from "express";
import cors from "cors";
import crypto from "crypto";
import fetch from "node-fetch";

const app = express();
app.use(express.json({ limit: "5mb" }));
app.use(cors({ origin: "*" })); // tighten to your Bubble domain later

const KEY = process.env.EVALUATOR_API_KEY;

// Health check
app.get("/", (_, res) => res.send("OK"));

app.post("/evaluate", async (req, res) => {
  try {
    // 1) Auth
    const auth = req.headers.authorization || "";
    if (!KEY || !auth.startsWith("Bearer ") || auth.slice(7) !== KEY) {
      return res.status(401).json({ error: "Unauthorized" });
    }

    // 2) Validate input
    const { photos = [], gender, height_cm, age, measurements } = req.body || {};
    if (!Array.isArray(photos) || photos.length === 0) {
      return res.status(400).json({ error: "`photos` must be a non-empty array of URLs" });
    }

    // 3) Optional reachability check (non-fatal if blocked by CDN)
    try {
      await fetch(photos[0], { method: "HEAD" });
    } catch (_) {
      // ignore
    }

    // 4) Preference-aware scoring (rule-based for now)
    // ---- CONFIG: edit these to your taste/market ----
    const PREFS = {
      female: {
        height_cm: { min: 172, target: 177, max: 182 }, // sweet spot
        meas_target: { b: 82, w: 60, h: 88 },           // target digitals
        meas_tolerance: { b: 6, w: 4, h: 6 }            // +/- range that still scores well
      },
      male: {
        height_cm: { min: 182, target: 186, max: 192 },
        meas_target: { b: 96, w: 76, h: 94 },
        meas_tolerance: { b: 8, w: 6, h: 8 }
      }
    };
    const GENDER = (gender || "female").toLowerCase().startsWith("m") ? "male" : "female";
    const P = PREFS[GENDER];

    const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));

    // Parse "82-60-88" → {b:82,w:60,h:88}
    const parseMeas = (mStr) => {
      if (!mStr || typeof mStr !== "string") return null;
      const nums = mStr.match(/\d+(\.\d+)?/g);
      if (!nums || nums.length < 3) return null;
      return { b: Number(nums[0]), w: Number(nums[1]), h: Number(nums[2]) };
    };

    // Height score: 0..1 within min..max, with bonus near target
    const heightScore = (h) => {
      if (!h || !Number(h)) return 0;
      const x = Number(h);
      if (x <= P.height_cm.min) return 0;
      if (x >= P.height_cm.max) return 1;
      // base linear within min..max
      let s = (x - P.height_cm.min) / (P.height_cm.max - P.height_cm.min);
      // bell bonus around target
      const span = (P.height_cm.max - P.height_cm.min) / 2;
      const t = P.height_cm.target;
      const bonus = Math.exp(-Math.pow((x - t) / (span / 2), 2)) * 0.15; // up to +0.15
      return clamp(s + bonus);
    };

    // Measurements score: closeness to target within tolerance bands
    const m = parseMeas(measurements);
    const measScore = (() => {
      if (!m) return 0;
      const dxB = Math.abs(m.b - P.meas_target.b) / P.meas_tolerance.b;
      const dxW = Math.abs(m.w - P.meas_target.w) / P.meas_tolerance.w;
      const dxH = Math.abs(m.h - P.meas_target.h) / P.meas_tolerance.h;
      // each term: 1 at target, falls off as it exceeds tolerance
      const sB = clamp(1 - dxB);
      const sW = clamp(1 - dxW);
      const sH = clamp(1 - dxH);
      // emphasize waist a bit more
      return clamp((sB + 1.2 * sW + sH) / 3.2);
    })();

    // Photo signals
    const photoCount = photos.length;
    const photoCountBoost = clamp((photoCount - 2) * 0.04, 0, 0.12); // +4% per photo beyond 2, max +12%

    // keep tiny deterministic noise so ties spread (hash of URLs)
    const hash = crypto.createHash("sha256").update(photos.join("|")).digest("hex");
    const n = parseInt(hash.slice(0, 6), 16);
    const tinyNoise = (n % 100) / 10000; // 0..0.0099

    // Combine (weights you can tweak)
    let confidence =
      0.55 * measScore +
      0.35 * heightScore(height_cm) +
      0.10 * (photoCount > 0 ? 0.6 : 0) +
      photoCountBoost +
      tinyNoise;
    confidence = clamp(confidence);

    // Decision thresholds (tweakable)
    const decision = confidence >= 0.70 ? "proceed" : confidence >= 0.45 ? "review" : "reject";

    // Reasons (human-readable)
    const reasons = [];
    if (height_cm) {
      const hs = heightScore(height_cm);
      if (hs >= 0.75) reasons.push("height in editorial sweet spot");
      else if (hs >= 0.45) reasons.push("height within range");
      else reasons.push("height below preferred range");
    }
    if (m) {
      if (measScore >= 0.75) reasons.push("measurements close to target");
      else if (measScore >= 0.45) reasons.push("measurements acceptable");
      else reasons.push("measurements outside preferred band");
    }
    if (photoCount >= 3) reasons.push("enough digitals provided");
    if (photoCount < 2) reasons.push("insufficient digitals");

    const reason = reasons.join("; ");
    const details = `photos=${photos.length}, gender=${gender ?? "n/a"}, h=${height_cm ?? "n/a"}, age=${age ?? "n/a"}, meas=${measurements ?? "n/a"}`;

    return res.json({ decision, confidence, reason, details });
  } catch (e) {
    console.error(e);
    return res.status(500).json({ error: "server_error" });
  }
});

const port = process.env.PORT || 10000;
app.listen(port, () => console.log("Evaluator running on :" + port));
