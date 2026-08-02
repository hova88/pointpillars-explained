import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL("https://hova88.github.io/pointpillars-explained/"),
  title: "Interactive PointPillars — Complete Algorithm Explorer",
  description: "A rigorous, animated explanation of every PointPillars representation, operation, motivation and tradeoff.",
  openGraph:{title:"See every PointPillars decision",description:"Trace a real nuScenes LiDAR sweep through pillars, learned features, a BEV backbone and 3D detections."},
  twitter:{card:"summary",title:"Interactive PointPillars",description:"The complete algorithm, deconstructed."},
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
