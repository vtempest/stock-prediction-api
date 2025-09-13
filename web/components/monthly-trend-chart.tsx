"use client"

import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from "recharts"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"

export default function MonthlyTrendChart() {
  // Generate sample data for the last 24 months
  const generateMonthlyData = () => {
    const data = []
    const now = new Date()
    const twoYearsAgo = new Date()
    twoYearsAgo.setFullYear(now.getFullYear() - 2)

    // Generate monthly data points for 2 years (24 months)
    for (let i = 0; i < 24; i++) {
      const date = new Date(twoYearsAgo)
      date.setMonth(twoYearsAgo.getMonth() + i)

      // Add some randomness and seasonal patterns
      const seasonalFactor = Math.sin((i / 12) * Math.PI * 2) * 0.3 + 1

      data.push({
        month: date.toISOString().split("T")[0].substring(0, 7),
        facility1: Math.round((Math.random() * 20000 + 80000) * seasonalFactor),
        facility2: Math.round((Math.random() * 18000 + 70000) * seasonalFactor),
      })
    }

    return data
  }

  const monthlyData = generateMonthlyData()

  return (
    <ChartContainer
      config={{
        facility1: {
          label: "Facility 1",
          color: "hsl(var(--chart-1))",
        },
        facility2: {
          label: "Facility 2",
          color: "hsl(var(--chart-2))",
        },
      }}
    >
      <BarChart
        accessibilityLayer
        data={monthlyData}
        margin={{
          left: 5,
          right: 5,
          top: 5,
          bottom: 5,
        }}
      >
        <CartesianGrid vertical={false} strokeDasharray="3 3" />
        <XAxis
          dataKey="month"
          tickLine={false}
          axisLine={false}
          tickMargin={10}
          tickFormatter={(value) => {
            const date = new Date(value + "-01")
            return `${date.toLocaleString("default", { month: "short" })} ${date.getFullYear()}`
          }}
          tick={{ fontSize: 12 }}
          minTickGap={30}
        />
        <YAxis
          tickLine={false}
          axisLine={false}
          tickMargin={10}
          tickFormatter={(value) => `${(value / 1000).toFixed(0)}k`}
          tick={{ fontSize: 12 }}
        />
        <ChartTooltip
          content={
            <ChartTooltipContent
              formatter={(value) => `${value.toLocaleString()} BTU`}
              labelFormatter={(label) => {
                const date = new Date(label + "-01")
                return date.toLocaleDateString("en-US", {
                  year: "numeric",
                  month: "long",
                })
              }}
            />
          }
        />
        <Bar dataKey="facility1" fill="var(--color-facility1)" radius={[4, 4, 0, 0]} />
        <Bar dataKey="facility2" fill="var(--color-facility2)" radius={[4, 4, 0, 0]} />
      </BarChart>
    </ChartContainer>
  )
}
