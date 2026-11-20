// Cloudflare Worker: relays requests to NVIDIA and adds the CORS headers browsers need.
// The API key is already included below. Nothing to edit.
export default {
  async fetch(req, env) {
    const cors = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Headers": "*",
      "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    };
    if (req.method === "OPTIONS") return new Response(null, { headers: cors });
    const path = new URL(req.url).pathname;
    const r = await fetch("https://integrate.api.nvidia.com/v1" + path, {
      method: req.method,
      headers: { Authorization: "Bearer " + (env.KEY || "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"), "Content-Type": "application/json" },
      body: req.method === "POST" ? req.body : undefined,
    });
    return new Response(r.body, {
      status: r.status,
      headers: { ...cors, "Content-Type": r.headers.get("Content-Type") || "application/json" },
    });
  },
};
