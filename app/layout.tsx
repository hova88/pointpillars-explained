import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL("https://hova88.github.io/pointpillars-explained/"),
  title: "PointPillars Geometry Lab",
  description: "A focused 3D explanation of pillar construction, feature decoration, BEV encoding and anchor-based detection.",
  openGraph:{title:"PointPillars Geometry Lab",description:"Follow one real LiDAR frame from points to pillars, centers, offsets, anchors and detections."},
  twitter:{card:"summary",title:"PointPillars Geometry Lab",description:"One LiDAR frame. Every geometric decision."},
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
