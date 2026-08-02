import type { NextConfig } from "next";

const basePath = process.env.GITHUB_PAGES === "true" ? "/pointpillars-explained" : "";
const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: true,
  basePath,
  assetPrefix: basePath,
  images: { unoptimized: true },
};

export default nextConfig;
