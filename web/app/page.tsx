import type { Metadata } from "next"
import DashboardView from "@/components/dashboard-view"

export const metadata: Metadata = {
  title: "Methane Emissions Dashboard",
  description: "Monitor and compare methane emissions and weather variables across facilities",
}

export default function Home() {
  return <DashboardView />
}
