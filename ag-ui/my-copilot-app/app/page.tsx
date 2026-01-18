"use client";
import { useFrontendTool } from "@copilotkit/react-core";
import { CopilotKitCSSProperties, CopilotSidebar } from "@copilotkit/react-ui";
import { useState } from "react";

export default function Page() {
  const [themeColor, setThemeColor] = useState("#6366f1");

  useFrontendTool({
    name: "set_theme_color",
    parameters: [
      {
        name: "theme_color",
        description: "The theme color to set. Make sure to pick nice colors.",
        required: true,
      },
    ],
    handler({ theme_color }) {
      setThemeColor(theme_color)
    }
  })

  return (
    <main
      style={
        { "--copilot-kit-primary-color": themeColor } as CopilotKitCSSProperties
      }
    >
      <h1>Your App</h1>
      <CopilotSidebar />
    </main>
  );
}
