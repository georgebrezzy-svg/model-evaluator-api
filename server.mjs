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

    // 3) Optional reachability check
    try {
      await fetch(photos[0], { method: "HEAD" });
    } catch (_) {
      // ignore if HEAD blocked by CDN; not fatal
    }

    // 4) TEMP scoring: deterministic but varied by photo URLs
    const hash = crypto.createHash("sha256").update(photos.join("|")).digest("hex");
    const n = parseInt(hash.slice(0, 8), 16); // 32-bit int from hash
    let confidence = (n % 10000) / 10000;     // 0.0000..0.9999

    // Gentle nudges (demo only)
    if (height_cm && Number(height_cm) >= 175) confidence = Math.max(confidence, 0.35);
    if (age && Number(age) < 19) confidence = Math.min(confidence + 0.05, 0.99);

    const decision = confidence >= 0.65 ? "proceed" : confidence >= 0.35 ? "review" : "reject";
    const reason = "Demo: score derived from hashed photo URLs.";
    const details = `photos=${photos.length}, gender=${gender ?? "n/a"}, h=${height_cm ?? "n/a"}, age=${age ?? "n/a"}, meas=${measurements ?? "n/a"}`;

    return res.json({ decision, confidence, reason, details });
  } catch (e) {
    console.error(e);
    return res.status(500).json({ error: "server_error" });
  }
});

const port = process.env.PORT || 10000;
app.listen(port, () => console.log("Evaluator running on :" + port));
